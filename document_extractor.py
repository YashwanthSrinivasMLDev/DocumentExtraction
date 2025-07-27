import os
import json
from PIL import Image
from typing import Dict, List, Any
from pathlib import Path
import cv2

# Import deepdoctection
import deepdoctection as dd

# Import the necessary pipeline components directly
from deepdoctection.dataprocessing.layout import LayoutDetection
from deepdoctection.ocr import TesseractOCR

# Import other necessary libraries
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, Table, Text


class DocumentExtractor:
    """
    A class to orchestrate an intermediate-to-advanced document extraction pipeline
    using open-source tools.
    """

    def __init__(self, model_name: str = "deepdoctection/layoutlmv3-base-finetuned-funsd"):
        """
        Initializes the document extraction pipeline.

        Args:
            model_name (str): The Hugging Face model to use for layout and information extraction.
        """
        print("Initializing DocumentExtractor...")
        self.model_name = model_name

        # We no longer need this variable as we're not using a config file
        self.config_file = None

        self._initialize_pipeline()
        print("DocumentExtractor initialized successfully.")

    def _initialize_pipeline(self):
        """
        Sets up the deepdoctection pipeline using the modern API.
        """
        # The modern approach is to directly instantiate a DoctectionPipe and
        # pass the components to it.
        # We import LayoutDetection and TesseractOCR directly, so we can use them
        # without the incorrect dd.dd_layout prefix.
        self.dd_pipeline = dd.DoctectionPipe(
            # A layout analysis model for detecting elements like text, titles, tables, etc.
            pipeline_component_1=LayoutDetection(self.model_name),

            # An OCR component to convert image text to machine-readable text.
            pipeline_component_2=TesseractOCR()
        )

    def _create_default_config(self):
        """
        This method is no longer needed with the modern API.
        """
        pass

    def extract_from_document(self, file_path: str) -> Dict[str, Any]:
        """
        Processes a single document and extracts all available information.

        Args:
            file_path (str): The path to the document file.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted information.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at {file_path}")

        print(f"Processing document: {file_path}")

        try:
            analyzed_doc = self.dd_pipeline.analyze(path=file_path)
            return self._format_output(analyzed_doc)
        except Exception as e:
            print(f"Error processing with deepdoctection: {e}. Falling back to unstructured.")
            return self._fallback_unstructured_extraction(file_path)

    def _format_output(self, analyzed_doc) -> Dict[str, Any]:
        """
        Formats the output from the deepdoctection pipeline into a structured dictionary.
        """
        extracted_data = {
            "file_path": analyzed_doc.get_info().path,
            "page_count": len(analyzed_doc.pages),
            "elements": []
        }

        for page in analyzed_doc.pages:
            page_elements = []
            # Extract text blocks
            for text_block in page.text:
                page_elements.append({
                    "type": "text",
                    "text": text_block.get_text(),
                    "bounding_box": text_block.bounding_box
                })

            # Extract tables
            for table in page.tables:
                page_elements.append({
                    "type": "table",
                    "table_data": table.get_table_csv(),
                    "bounding_box": table.bounding_box
                })

            # Extract headers, titles, and other layout elements
            for element in page.layouts:
                page_elements.append({
                    "type": element.category_name,  # e.g., 'title', 'figure', 'list'
                    "text": element.get_text(),
                    "bounding_box": element.bounding_box
                })

            extracted_data["elements"].append({
                "page_number": page.page_number,
                "data": page_elements
            })

        return extracted_data

    def _fallback_unstructured_extraction(self, file_path: str) -> Dict[str, Any]:
        """
        A simpler fallback method using unstructured-io.
        """
        elements = partition_pdf(filename=file_path, strategy="hi_res")

        fallback_data = {
            "file_path": file_path,
            "elements": []
        }

        current_page_number = -1
        page_data = []

        for element in elements:
            if element.metadata.page_number is not None and element.metadata.page_number != current_page_number:
                if page_data:
                    fallback_data["elements"].append({
                        "page_number": current_page_number,
                        "data": page_data
                    })
                current_page_number = element.metadata.page_number
                page_data = []

            element_dict = {
                "type": "unknown",
                "text": element.text,
                "bounding_box": element.metadata.bbox
            }
            if isinstance(element, Title):
                element_dict["type"] = "title"
            elif isinstance(element, Table):
                element_dict["type"] = "table"
                element_dict["table_data"] = element.text
            elif isinstance(element, Text):
                element_dict["type"] = "text"

            page_data.append(element_dict)

        if page_data:
            fallback_data["elements"].append({
                "page_number": current_page_number,
                "data": page_data
            })

        return fallback_data


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
    for page in extracted_info["elements"]:
        print(f"Page {page['page_number']}:")
        for element in page["data"]:
            if element["type"] == "title":
                print(f"  - Title: {element['text']}")
            elif element["type"] == "table":
                print(f"  - Found a table with data:\n{element['table_data']}")
            elif element["type"] == "text":
                print(f"  - Text: {element['text'][:50]}...")