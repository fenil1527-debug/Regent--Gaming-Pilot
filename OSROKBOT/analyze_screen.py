
import argparse
from pathlib import Path

import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def analyze_screenshot(image_path: str) -> None:
    path = Path(image_path)

    if not path.is_file():
        print(f"Error: Image not found: {path}")
        return

    image = cv2.imread(str(path))

    if image is None:
        print("Error: Could not read the image.")
        return

    height, width = image.shape[:2]

    print(f"\nOriginal resolution: {width} x {height}")

    # Enlarge small screenshots before OCR
    scale = max(1, 1200 // max(width, height))
    enlarged = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    # Convert to grayscale
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)

    # Improve contrast and create a binary image
    gray = cv2.equalizeHist(gray)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Save processed image for visual inspection
    processed_path = path.with_name(
        f"{path.stem}_processed.png"
    )
    cv2.imwrite(str(processed_path), binary)

    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 11",
    ]

    for config in configs:
        print(f"\n--- OCR configuration: {config} ---")
        text = pytesseract.image_to_string(binary, config=config)
        print(text.strip() or "No text detected.")

    print(f"\nProcessed image saved: {processed_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Screenshot image path")
    args = parser.parse_args()

    analyze_screenshot(args.image)
