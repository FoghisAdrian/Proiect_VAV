import argparse
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def ensure_parent_folder(path):
    parent = Path(path).parent
    if str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)


def resize_keep_aspect(frame, target_width):
    if target_width is None or target_width <= 0:
        return frame

    h, w = frame.shape[:2]

    if w <= target_width:
        return frame

    scale = target_width / float(w)
    new_h = int(h * scale)

    return cv2.resize(frame, (target_width, new_h), interpolation=cv2.INTER_AREA)


def transform_to_grayscale(img: np.ndarray):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_gaussian_blur(img: np.ndarray):
    kernel = np.array([
        [1, 4, 7, 4, 1],
        [4, 16, 26, 16, 4],
        [7, 26, 41, 26, 7],
        [4, 16, 26, 16, 4],
        [1, 4, 7, 4, 1]
    ], dtype=np.float32) / 273.0

    return cv2.filter2D(img, -1, kernel)


def white_top_hat_enhancer(img, size=21):
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))

    erosion = cv2.erode(img, kernel)
    opening = cv2.dilate(erosion, kernel)

    top_hat = cv2.subtract(img, opening)
    return top_hat


def apply_sobel_edge_detector(img: np.ndarray):
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=np.float32)

    gx = cv2.filter2D(img.astype(np.float32), -1, kernel_x)
    gy = cv2.filter2D(img.astype(np.float32), -1, kernel_y)

    gradient_magnitude = cv2.magnitude(gx, gy)
    gradient_normalized = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    return gradient_normalized


def normalize_hazard_map(hazard_map):
    return cv2.normalize(hazard_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def build_binary_from_hazard_map(hazard_map, threshold_bias=0):
    normalized = normalize_hazard_map(hazard_map)

    otsu_threshold, _ = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    effective_threshold = int(np.clip(otsu_threshold + threshold_bias, 0, 255))

    _, binary = cv2.threshold(
        normalized,
        effective_threshold,
        255,
        cv2.THRESH_BINARY
    )

    return normalized, binary, effective_threshold


def clean_binary_mask(binary):
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    cleaned = cv2.dilate(cleaned, kernel_small, iterations=1)

    return cleaned


def detect_lane_like_segments(binary_mask):
    h, w = binary_mask.shape[:2]

    lines = cv2.HoughLinesP(
        binary_mask,
        rho=1,
        theta=np.pi / 180,
        threshold=25,
        minLineLength=max(20, int(w * 0.06)),
        maxLineGap=max(12, int(w * 0.03))
    )

    if lines is None:
        return []

    kept = []

    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in line]

        dx = x2 - x1
        dy = y2 - y1

        if dx == 0:
            slope = 999.0
        else:
            slope = dy / float(dx)

        mid_y = (y1 + y2) / 2.0
        length = np.hypot(dx, dy)

        if mid_y < h * 0.35:
            continue

        if length < max(18, w * 0.035):
            continue

        if abs(slope) < 0.12:
            continue

        if abs(slope) > 10.0:
            continue

        kept.append((x1, y1, x2, y2))

    return kept


def x_at_y_for_segment(x1, y1, x2, y2, y_target):
    if y2 == y1:
        return (x1 + x2) / 2.0

    t = (y_target - y1) / float(y2 - y1)
    return x1 + t * (x2 - x1)


def split_segments_left_right(segments, width, height):
    left_segments = []
    right_segments = []

    center_x = width / 2.0
    y_bottom = height - 1

    for x1, y1, x2, y2 in segments:
        xb = x_at_y_for_segment(x1, y1, x2, y2, y_bottom)

        if xb < center_x:
            left_segments.append((x1, y1, x2, y2))
        else:
            right_segments.append((x1, y1, x2, y2))

    return left_segments, right_segments


def fit_line_from_segments(segments):
    if not segments:
        return None

    pts = []

    for x1, y1, x2, y2 in segments:
        pts.append([x1, y1])
        pts.append([x2, y2])

    pts = np.array(pts, dtype=np.float32).reshape(-1, 1, 2)

    if len(pts) < 2:
        return None

    vx, vy, x0, y0 = cv2.fitLine(
        pts,
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    )

    vx = float(vx)
    vy = float(vy)
    x0 = float(x0)
    y0 = float(y0)

    if abs(vy) < 1e-5:
        return None

    return vx, vy, x0, y0


def x_from_fitline(line, y):
    vx, vy, x0, y0 = line
    return x0 + ((y - y0) * vx / vy)


def build_adaptive_road_corridor(binary_mask, segments):
    h, w = binary_mask.shape[:2]

    segment_mask = np.zeros_like(binary_mask)

    for x1, y1, x2, y2 in segments:
        cv2.line(segment_mask, (x1, y1), (x2, y2), 255, 3)

    left_segments, right_segments = split_segments_left_right(segments, w, h)

    left_line = fit_line_from_segments(left_segments)
    right_line = fit_line_from_segments(right_segments)

    corridor_mask = None

    if left_line is not None and right_line is not None:
        y_top = int(h * 0.45)
        y_bottom = h - 1

        lx_top = int(np.clip(x_from_fitline(left_line, y_top), 0, w - 1))
        lx_bottom = int(np.clip(x_from_fitline(left_line, y_bottom), 0, w - 1))
        rx_top = int(np.clip(x_from_fitline(right_line, y_top), 0, w - 1))
        rx_bottom = int(np.clip(x_from_fitline(right_line, y_bottom), 0, w - 1))

        if lx_top < rx_top and lx_bottom < rx_bottom:
            polygon = np.array([
                [lx_bottom, y_bottom],
                [lx_top, y_top],
                [rx_top, y_top],
                [rx_bottom, y_bottom]
            ], dtype=np.int32)

            corridor_mask = np.zeros_like(binary_mask)
            cv2.fillPoly(corridor_mask, [polygon], 255)

            corridor_mask = cv2.dilate(
                corridor_mask,
                cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21)),
                iterations=1
            )

    return segment_mask, corridor_mask


def remove_small_components(mask, min_area=35):
    cleaned = np.zeros_like(mask)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if area >= min_area:
            cv2.drawContours(cleaned, [contour], -1, 255, thickness=cv2.FILLED)

    return cleaned


def build_final_road_marking_mask(binary_mask, segment_mask, corridor_mask):
    h, w = binary_mask.shape[:2]

    if corridor_mask is not None:
        candidate_mask = cv2.bitwise_and(binary_mask, corridor_mask)
    else:
        candidate_mask = binary_mask.copy()

    combined = cv2.bitwise_or(candidate_mask, segment_mask)

    combined = remove_small_components(combined, min_area=25)

    lower_weight_mask = np.zeros_like(combined)
    lower_weight_mask[int(h * 0.25):, :] = 255
    combined = cv2.bitwise_and(combined, lower_weight_mask)

    combined = cv2.morphologyEx(
        combined,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
        iterations=2
    )

    combined = cv2.dilate(
        combined,
        cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
        iterations=1
    )

    return combined


def stylize_hud_overlay(original_bgr, road_mask):
    overlay = original_bgr.copy()

    fill_layer = np.zeros_like(original_bgr)
    fill_layer[road_mask > 0] = (0, 220, 90)

    glow = cv2.GaussianBlur(road_mask, (0, 0), sigmaX=6, sigmaY=6)
    glow_layer = np.zeros_like(original_bgr)
    glow_layer[:, :, 1] = glow
    glow_layer[:, :, 2] = (glow * 0.18).astype(np.uint8)

    overlay = cv2.addWeighted(overlay, 1.0, glow_layer, 0.32, 0)
    overlay = cv2.addWeighted(overlay, 0.85, fill_layer, 0.15, 0)

    contours, _ = cv2.findContours(
        road_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(overlay, contours, -1, (90, 255, 130), 2)

    final_blend = cv2.addWeighted(original_bgr, 0.58, overlay, 0.42, 0)

    return final_blend


def process_frame(original_img, threshold_bias=0):
    gray = transform_to_grayscale(original_img)

    denoised = apply_gaussian_blur(gray)

    bright_hazards = white_top_hat_enhancer(denoised, size=21)
    enhanced_bright = cv2.multiply(bright_hazards, 2)

    road_boundaries = apply_sobel_edge_detector(denoised)

    hazard_map = cv2.addWeighted(
        enhanced_bright,
        1.0,
        road_boundaries,
        0.7,
        0
    )

    normalized_hazard, binary_mask, used_threshold = build_binary_from_hazard_map(
        hazard_map,
        threshold_bias=threshold_bias
    )

    cleaned_binary = clean_binary_mask(binary_mask)

    segments = detect_lane_like_segments(cleaned_binary)

    segment_mask, corridor_mask = build_adaptive_road_corridor(cleaned_binary, segments)

    final_road_mask = build_final_road_marking_mask(
        cleaned_binary,
        segment_mask,
        corridor_mask
    )

    hud_display = stylize_hud_overlay(original_img, final_road_mask)

    return gray, normalized_hazard, final_road_mask, hud_display, used_threshold


def build_live_debug_view(original_img, gray, hazard_map, final_mask, hud_display, used_threshold):
    h, w = original_img.shape[:2]

    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    hazard_bgr = cv2.cvtColor(hazard_map, cv2.COLOR_GRAY2BGR)

    gray_bgr = cv2.resize(gray_bgr, (w, h))
    hazard_bgr = cv2.resize(hazard_bgr, (w, h))
    hud_display = cv2.resize(hud_display, (w, h))

    top = np.hstack([original_img, gray_bgr])
    bottom = np.hstack([hazard_bgr, hud_display])

    debug_view = np.vstack([top, bottom])

    cv2.putText(debug_view, "1. Original", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(debug_view, "2. Grayscale + Gaussian", (w + 15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(debug_view, "3. Top-Hat + Sobel Hazard Map", (15, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(debug_view, "4. Final HUD Overlay", (w + 15, h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.putText(
        debug_view,
        f"Adaptive threshold: {used_threshold}",
        (w + 15, h + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 255, 200),
        2
    )

    return debug_view


def show_image_results(gray, hazard_map, final_mask, hud_display):
    cv2.imshow("1. Input Grayscale", gray)
    cv2.imshow("2. Top-Hat + Sobel Hazard Map", hazard_map)
    cv2.imshow("3. Final Road Marking Mask", final_mask)
    cv2.imshow("4. Augmented AR HUD Display", hud_display)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_image(input_path, output_path=None, resize_width=None, show=True, threshold_bias=0):
    original_img = cv2.imread(input_path)

    if original_img is None:
        raise FileNotFoundError(f"Could not find image at {input_path}")

    original_img = resize_keep_aspect(original_img, resize_width)

    gray, hazard_map, final_mask, hud_display, used_threshold = process_frame(
        original_img,
        threshold_bias=threshold_bias
    )

    if output_path:
        ensure_parent_folder(output_path)
        cv2.imwrite(output_path, hud_display)
        print(f"Saved image result to: {output_path}")

    if show:
        show_image_results(gray, hazard_map, final_mask, hud_display)


def process_video(input_path, output_path=None, resize_width=None, show=True, live_debug=False, threshold_bias=0):
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 25

    writer = None
    frame_index = 0
    window_name = "Fog-Cutter Hazard Enhancer"

    if show:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while True:
        ret, original_img = cap.read()

        if not ret:
            break

        original_img = resize_keep_aspect(original_img, resize_width)

        gray, hazard_map, final_mask, hud_display, used_threshold = process_frame(
            original_img,
            threshold_bias=threshold_bias
        )

        if live_debug:
            live_display = build_live_debug_view(
                original_img,
                gray,
                hazard_map,
                final_mask,
                hud_display,
                used_threshold
            )
        else:
            live_display = hud_display

        if writer is None and output_path:
            ensure_parent_folder(output_path)

            h, w = hud_display.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (w, h)
            )

            if not writer.isOpened():
                cap.release()
                raise RuntimeError(f"Could not create output video: {output_path}")

        if writer is not None:
            writer.write(hud_display)

        if show:
            lh, lw = live_display.shape[:2]
            max_preview_width = 1280

            if lw > max_preview_width:
                scale = max_preview_width / float(lw)
                preview_h = int(lh * scale)

                preview = cv2.resize(
                    live_display,
                    (max_preview_width, preview_h),
                    interpolation=cv2.INTER_AREA
                )
            else:
                preview = live_display

            ph, pw = preview.shape[:2]
            cv2.resizeWindow(window_name, pw, ph)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == 27:
                break

        frame_index += 1

        if frame_index % 50 == 0:
            print(f"Processed {frame_index} frames...")

    cap.release()

    if writer is not None:
        writer.release()
        print(f"Saved video result to: {output_path}")

    cv2.destroyAllWindows()


def process_folder(input_folder, output_folder, resize_width=None, show=False, live_debug=False, threshold_bias=0):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in input_folder.rglob("*") if p.is_file()])

    for file_path in files:
        suffix = file_path.suffix.lower()
        relative = file_path.relative_to(input_folder)

        if suffix in IMAGE_EXTENSIONS:
            output_path = output_folder / relative.with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Processing image: {file_path}")

            process_image(
                str(file_path),
                str(output_path),
                resize_width=resize_width,
                show=show,
                threshold_bias=threshold_bias
            )

        elif suffix in VIDEO_EXTENSIONS:
            output_path = output_folder / relative.with_suffix(".mp4")
            output_path.parent.mkdir(parents=True, exist_ok=True)

            print(f"Processing video: {file_path}")

            process_video(
                str(file_path),
                str(output_path),
                resize_width=resize_width,
                show=show,
                live_debug=live_debug,
                threshold_bias=threshold_bias
            )


def detect_input_type(path):
    path_obj = Path(path)

    if path_obj.is_dir():
        return "folder"

    suffix = path_obj.suffix.lower()

    if suffix in IMAGE_EXTENSIONS:
        return "image"

    if suffix in VIDEO_EXTENSIONS:
        return "video"

    raise ValueError(f"Unsupported input type: {path}")


def main():
    parser = argparse.ArgumentParser(description="Fog-Cutter Hazard Enhancer")

    parser.add_argument("--input", required=True, help="Path to an image, video, or folder.")
    parser.add_argument("--output", default=None, help="Path where the final HUD result will be saved.")
    parser.add_argument(
        "--resize-width",
        type=int,
        default=640,
        help="Resize frames to this width before processing. Use 0 to keep original size."
    )
    parser.add_argument(
        "--threshold-bias",
        type=int,
        default=0,
        help="Negative = more green detections, positive = stricter filtering."
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open preview window.")
    parser.add_argument(
        "--live-debug",
        action="store_true",
        help="Show original, grayscale, hazard map and final HUD in one live window."
    )

    args = parser.parse_args()

    input_type = detect_input_type(args.input)
    resize_width = None if args.resize_width == 0 else args.resize_width
    show = not args.no_show

    if input_type == "image":
        output_path = args.output or "outputs/image_hud_result.png"

        process_image(
            args.input,
            output_path,
            resize_width=resize_width,
            show=show,
            threshold_bias=args.threshold_bias
        )

    elif input_type == "video":
        output_path = args.output or "outputs/video_hud_result.mp4"

        process_video(
            args.input,
            output_path,
            resize_width=resize_width,
            show=show,
            live_debug=args.live_debug,
            threshold_bias=args.threshold_bias
        )

    elif input_type == "folder":
        output_folder = args.output or "outputs"

        process_folder(
            args.input,
            output_folder,
            resize_width=resize_width,
            show=show,
            live_debug=args.live_debug,
            threshold_bias=args.threshold_bias
        )


if __name__ == "__main__":
    main()