import cv2
import numpy as np
import argparse
import json
from pathlib import Path


def load_image_paths(path: str):
    p = Path(path)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
    elif p.is_file():
        return [p]
    else:
        raise FileNotFoundError(f"Path not found: {path}")


def hsv_fire_smoke_masks(bgr_img):
    """Return fire_mask, smoke_mask given a BGR image."""
    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    # Fire: lower hue, strong saturation and value
    fire_lower = np.array([0, 50, 50])
    fire_upper = np.array([25, 255, 255])
    fire_mask = cv2.inRange(hsv, fire_lower, fire_upper)

    # Smoke: low saturation, mid-to-high value (light gray/white)
    smoke_lower = np.array([0, 0, 80])
    smoke_upper = np.array([180, 60, 255])
    smoke_mask = cv2.inRange(hsv, smoke_lower, smoke_upper)

    # Optional: clean masks a bit
    kernel = np.ones((3, 3), np.uint8)
    fire_mask = cv2.morphologyEx(fire_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    smoke_mask = cv2.morphologyEx(smoke_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    return fire_mask, smoke_mask


def contours_to_detections(mask, cls_name, min_area=500):
    """Convert a binary mask to bounding-box detections."""
    detections = []

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = mask.shape[:2]

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        x1, y1 = x, y
        x2, y2 = x + bw, y + bh

        # Simple confidence: normalized area
        conf = float(area / (w * h))
        conf = float(min(max(conf * 10, 0.1), 0.99))  # stretch a bit

        detections.append([int(x1), int(x2), int(y1), int(y2), round(conf, 2), cls_name])

    return detections


def annotate_image(bgr_img, detections):
    """Draw bounding boxes and labels on the image."""
    annotated = bgr_img.copy()
    for x1, x2, y1, y2, conf, cls_name in detections:
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{cls_name} {int(conf * 100)}%"
        cv2.putText(annotated, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return annotated


def process_image(img_path: Path, outdir: Path, show: bool = False):
    bgr = cv2.imread(str(img_path))
    if bgr is None:
        print(f"Warning: could not read {img_path}")
        return None

    fire_mask, smoke_mask = hsv_fire_smoke_masks(bgr)

    fire_dets = contours_to_detections(fire_mask, "fire")
    smoke_dets = contours_to_detections(smoke_mask, "smoke")
    detections = fire_dets + smoke_dets

    # Save masks
    base = img_path.stem
    cv2.imwrite(str(outdir / f"fire_mask_{base}.png"), fire_mask)
    cv2.imwrite(str(outdir / f"smoke_mask_{base}.png"), smoke_mask)

    # Annotated image
    annotated = annotate_image(bgr, detections)
    cv2.imwrite(str(outdir / f"annotated_{base}.png"), annotated)

    # JSON output per image
    out_json = {
        "image": str(img_path),
        "detections": detections,
    }
    with open(outdir / f"detections_{base}.json", "w") as f:
        json.dump(out_json, f, indent=2)

    # Print JSON to console (per assignment style)
    print(json.dumps(out_json))

    if show:
        cv2.imshow("Original", bgr)
        cv2.imshow("Fire mask", fire_mask)
        cv2.imshow("Smoke mask", smoke_mask)
        cv2.imshow("Annotated", annotated)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            cv2.destroyAllWindows()

    return out_json


def main():
    parser = argparse.ArgumentParser(description="HSV-based Fire & Smoke Detection (Mini Project 2)")
    parser.add_argument("--path", required=True,
                        help="Path to an image file or a directory containing images.")
    parser.add_argument("--outdir", default="results",
                        help="Directory to save masks, annotations, and JSON outputs.")
    parser.add_argument("--show", action="store_true",
                        help="Show windows with original, masks, and annotated outputs.")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    img_paths = load_image_paths(args.path)

    for img_path in img_paths:
        process_image(img_path, outdir, show=args.show)

    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
