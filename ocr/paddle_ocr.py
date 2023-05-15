import os
import sys

from paddleocr import PaddleOCR, draw_ocr
from PIL import Image


def main(img_path, font_path):
    try:
        # Initialize OCR model
        ocr = PaddleOCR(use_angle_cls=True, lang="en")

        # Check if input image file exists
        if not os.path.isfile(img_path):
            raise ValueError(f"{img_path} does not exist")

        # Check if font path is valid
        if not os.path.isfile(font_path):
            raise ValueError(f"{font_path} does not exist")

        # Perform OCR on input image
        result = ocr.ocr(img_path, cls=True)
        image = Image.open(img_path).convert("RGB")

        # Get OCR results
        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]

        # Get rotation of text
        rotation = result[0][1][2]
        print(f"Text rotation: {rotation} degrees")
        # Draw OCR results on image
        im_show = draw_ocr(image, boxes, txts, scores, font_path=font_path)
        im_show = Image.fromarray(im_show)

        # Generate output path
        output_path = os.path.splitext(img_path)[0] + "_result.jpg"

        # Save output image
        im_show.save(output_path)
        print(f"OCR results saved to {output_path}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Check if image path was passed as command line argument
    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_path>")
        sys.exit()

    img_path = sys.argv[1]
    font_path = "./simfang.ttf"
    main(img_path, font_path)
