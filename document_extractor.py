import os
import json
from typing import Dict, List, Any
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title, Table, Text, ListItem, NarrativeText, Header, Footer


class DocumentExtractor:
    """
    A class to orchestrate document extraction using the simpler unstructured library.
    """

    def __init__(self):
        """
        Initializes the document extractor.
        No complex pipeline to initialize here!
        """
        print("Initializing DocumentExtractor with unstructured...")
        print("DocumentExtractor initialized successfully.")

    def extract_from_document(self, file_path: str) -> Dict[str, Any]:
        """
        Processes a single document and extracts all available information using unstructured.

        Args:
            file_path (str): The path to the document file.

        Returns:
            Dict[str, Any]: A dictionary containing the extracted information.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document not found at {file_path}")

        print(f"Processing document with unstructured: {file_path}")

        # Use unstructured's fast strategy to avoid deep learning dependencies
        elements = partition_pdf(filename=file_path, strategy="fast")

        return self._format_output(elements, file_path)

    def _format_output(self, elements: List[Any], file_path: str) -> Dict[str, Any]:
        """
        Formats the output from the unstructured library into a structured dictionary.
        """
        extracted_data = {
            "file_path": file_path,
            "elements": []
        }

        current_page_number = -1
        page_data = []

        for element in elements:
            if element.metadata.page_number is not None and element.metadata.page_number != current_page_number:
                if page_data:
                    extracted_data["elements"].append({
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
                element_dict["text"] = element.text
            elif isinstance(element, NarrativeText):
                element_dict["type"] = "text"
            elif isinstance(element, ListItem):
                element_dict["type"] = "list_item"
            elif isinstance(element, Header):
                element_dict["type"] = "header"
            elif isinstance(element, Footer):
                element_dict["type"] = "footer"

            page_data.append(element_dict)

        if page_data:
            extracted_data["elements"].append({
                "page_number": current_page_number,
                "data": page_data
            })

        return extracted_data


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
                print(f"  - Found a table with data:\n{element['text']}")
            elif element["type"] == "text":
                print(f"  - Text: {element['text'][:50]}...")