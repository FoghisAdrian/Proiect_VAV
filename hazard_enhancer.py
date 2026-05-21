import numpy as np
import cv2

def transform_to_grayscale(img: np.ndarray):
    h, w, _ = img.shape
    new_img = np.zeros((h, w), dtype=np.uint8)

    for i in range(h):
        for j in range(w):
            b, g, r = img[i, j]
            new_img[i, j] = int(0.299 * r + 0.587 * g + 0.114 * b)
    return new_img


def manual_convolution(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad = kh // 2
    padded_img = np.pad(img, pad, mode='edge')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded_img[i:i + kh, j:j + kw]
            output[i, j] = np.sum(region * kernel)

    return output.astype(np.uint8)


def manual_morphology(img, size=5, op='erosion'):
    h, w = img.shape
    pad = size // 2
    padded_img = np.pad(img, pad, mode='edge')
    output = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            region = padded_img[i:i + size, j:j + size]
            if op == 'erosion':
                output[i, j] = np.min(region)
            else:
                output[i, j] = np.max(region)

    return output


def apply_gaussian_blur(img: np.ndarray):

    kernel = np.array([[1, 4, 7, 4, 1],
                       [4, 16, 26, 16, 4],
                       [7, 26, 41, 26, 7],
                       [4, 16, 26, 16, 4],
                       [1, 4, 7, 4, 1]]) / 273.0
    return manual_convolution(img, kernel)


def white_top_hat_enhancer(img, size=15):
    erosion = manual_morphology(img, size, 'erosion')
    opening = manual_morphology(erosion, size, 'dilation')

    top_hat = cv2.subtract(img, opening)
    return top_hat


def apply_sobel_edge_detector(img: np.ndarray):
    kernel_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]], dtype=np.float32)

    kernel_y = np.array([[-1, -2, -1],
                         [0, 0, 0],
                         [1, 2, 1]], dtype=np.float32)

    img_float = img.astype(np.float64)

    h, w = img.shape
    padded_img = np.pad(img_float, 1, mode='edge')

    gx = np.zeros_like(img_float)
    gy = np.zeros_like(img_float)

    for i in range(h):
        for j in range(w):
            region = padded_img[i:i + 3, j:j + 3]
            gx[i, j] = np.sum(region * kernel_x)
            gy[i, j] = np.sum(region * kernel_y)

    gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)

    gradient_normalized = np.clip(gradient_magnitude, 0, 255).astype(np.uint8)

    return gradient_normalized


def get_bev_homography_matrix(img_shape):
    h, w = img_shape[:2]

    src_pts = np.float32([
        [int(w * 0.35), int(h * 0.48)],
        [int(w * 0.65), int(h * 0.48)],
        [int(w * 0.88), int(h * 0.95)],
        [int(w * 0.12), int(h * 0.95)]
    ])

    dst_pts = np.float32([
        [int(w * 0.25), 0],
        [int(w * 0.75), 0],
        [int(w * 0.75), h],
        [int(w * 0.25), h]
    ])

    A = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1, 0, 0, 0, x * xp, y * xp, xp])
        A.append([0, 0, 0, -x, -y, -1, x * yp, y * yp, yp])
    A = np.array(A)

    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    return H / H[2, 2]


def apply_backward_mapping_bev(img, H):
    h, w = img.shape[:2]
    H_inv = np.linalg.inv(H)
    warped = np.zeros_like(img)

    for i in range(h):
        for j in range(w):
            dst_vector = np.array([j, i, 1.0])
            src_projected = H_inv @ dst_vector

            src_x = src_projected[0] / src_projected[2]
            src_y = src_projected[1] / src_projected[2]

            if 0 <= src_x < w - 1 and 0 <= src_y < h - 1:
                warped[i, j] = img[int(src_y), int(src_x)]

    return warped


def generate_hud_overlay(original_bgr, hazard_map):
    output_display = original_bgr.copy()

    _, binary_mask = cv2.threshold(hazard_map, 45, 255, cv2.THRESH_BINARY)

    output_display[binary_mask > 0] = [0, 255, 0]

    final_blend = cv2.addWeighted(original_bgr, 0.6, output_display, 0.4, 0)

    return final_blend


if __name__ == "__main__":
    input_path = "data/016.png"
    original_img = cv2.imread(input_path)

    if original_img is None:
        print(f"Error: Could not find image at {input_path}")
    else:
        gray = transform_to_grayscale(original_img)

        denoised = apply_gaussian_blur(gray)

        bright_hazards = white_top_hat_enhancer(denoised, size=21)
        enhanced_bright = cv2.multiply(bright_hazards, 2)

        road_boundaries = apply_sobel_edge_detector(denoised)
        hazard_map = cv2.addWeighted(enhanced_bright, 1.0, road_boundaries, 0.7, 0)

        print("Computing spatial bird's eye warping projection...")
        H_matrix = get_bev_homography_matrix(hazard_map.shape)
        bev_hazard_map = apply_backward_mapping_bev(hazard_map, H_matrix)

        hud_display = generate_hud_overlay(original_img, hazard_map)

        cv2.imshow("1. Input Grayscale", gray)
        cv2.imshow("2. Perspective Hazard Profile", hazard_map)
        cv2.imshow("3. Top-Down BEV Safety Track Map", bev_hazard_map)
        cv2.imshow("4. Augmented AR HUD Display", hud_display)

        print("Pipeline Step 4 complete. Press any key to log results.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
