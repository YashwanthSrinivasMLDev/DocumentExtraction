import os
import json
from typing import Dict, List, Any
import pytesseract
from PIL import Image
from pypdf import PdfReader
from io import BytesIO


class DocumentExtractor:
    """
    A class for simple document extraction using pypdf and pytesseract.
    This approach is highly reliable and avoids complex dependencies.
    """

    def __init__(self):
        """
        Initializes the document extractor.
        """
        print("Initializing DocumentExtractor with pypdf and pytesseract...")
        # Check if tesseract is in PATH
        try:
            pytesseract.get_tesseract_version()
            print("Tesseract OCR is available.")
        except pytesseract.TesseractNotFoundError:
            print("Tesseract is not found. Please ensure it's installed and in PATH.")
            raise
        print("DocumentExtractor initialized successfully.")

    def extract_from_document(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts text from a PDF file using pypdf and performs OCR on images.

        Args:
            file_path (str): The path to the document file.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted text per page.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at {file_path}")

        print(f"Processing document: {file_path}")

        extracted_data = {
            "file_path": file_path,
            "pages": []
        }

        try:
            reader = PdfReader(file_path)
            for page_number, page in enumerate(reader.pages):
                page_text = page.extract_text()

                # Check for images and perform OCR if necessary
                if not page_text.strip():
                    try:
                        print(f"No text extracted on page {page_number + 1}, attempting OCR on images...")
                        page_text = self._extract_text_from_page_images(page)
                    except Exception as e:
                        print(f"OCR failed for page {page_number + 1}: {e}")
                        page_text = ""

                extracted_data["pages"].append({
                    "page_number": page_number + 1,
                    "text": page_text
                })

        except Exception as e:
            print(f"Error during PDF processing: {e}")
            raise

        return extracted_data

    def _extract_text_from_page_images(self, page) -> str:
        """
        Helper function to extract text from images on a single PDF page using Tesseract OCR.
        """
        text_from_images = []
        for image_file_obj in page.images:
            try:
                # Read the image data
                image_data = image_file_obj.data
                image = Image.open(BytesIO(image_data))

                # Perform OCR using Tesseract
                ocr_text = pytesseract.image_to_string(image)
                text_from_images.append(ocr_text)
            except Exception as e:
                print(f"Failed to process an image with OCR: {e}")

        return "\n".join(text_from_images)


def save_to_json(data: Dict[str, Any], output_path: str):
    """
    Saves a dictionary to a JSON file.
    """
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Extraction result saved to {output_path}")


if __name__ == "__main__":
    sample_pdf_path = "sample.pdf"
    if not os.path.exists(sample_pdf_path):
        print("Please place a PDF file named 'sample.pdf' in the same directory.")
        with open(sample_pdf_path, "w") as f:
            f.write("This is a dummy PDF file.")
        print("A dummy file 'sample.pdf' has been created.")
        exit()

    extractor = DocumentExtractor()
    extracted_info = extractor.extract_from_document(sample_pdf_path)
    save_to_json(extracted_info, "extracted_data.json")
    print("\n--- Summary of Extracted Data ---")
    for page in extracted_info["pages"]:
        print(f"Page {page['page_number']}:")
        print(f"  - Text: {page['text'][:100]}...")