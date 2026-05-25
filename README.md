# Fog-Cutter Hazard Enhancer

This project processes low-visibility road images and videos using basic image processing filters.

## Test videos

Test videos are stored in:

```text
data/videos
```

## Output folder

Before running the project, create an output folder:

```cmd
mkdir outputs
```

Output videos are not stored in Git. The `outputs/` folder is ignored.

## Install dependencies

```cmd
pip install opencv-python numpy
```

## Run all test videos

```cmd
python hazard_enhancer.py --input "data/videos/IMG_9848.mov" --output "outputs/IMG_9848_fogcutter.mp4" --resize-width 640 --live-debug --threshold-bias -5
python hazard_enhancer.py --input "data/videos/GX010154(a).mp4" --output "outputs/GX010154_fogcutter.mp4" --resize-width 640 --live-debug --threshold-bias -5
python hazard_enhancer.py --input "data/videos/WhatsApp Video 2025-01-08 at 07.58.55 (1).mp4" --output "outputs/whatsapp_fogcutter.mp4" --resize-width 640 --live-debug --threshold-bias -5
```

Press `q` or `Esc` to stop the live preview.
