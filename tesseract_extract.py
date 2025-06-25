from PIL import Image
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Adjust path

def extract_text(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh)
    return text

if __name__ == "__main__":
    result = extract_text("hackathon_input_image/input_image.jpeg")
    with open("donut_output.txt", "w", encoding="utf-8") as f:
        f.write(result)
    print("✅ Tesseract OCR output saved to donut_output.txt")
