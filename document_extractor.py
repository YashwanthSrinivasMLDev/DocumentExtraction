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

        # Use unstructured's hi_res strategy to get detailed output
        elements = partition_pdf(filename=file_path, strategy="hi_res")

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
            # Check for page breaks
            if element.metadata.page_number is not None and element.metadata.page_number != current_page_number:
                if page_data:
                    extracted_data["elements"].append({
                        "page_number": current_page_number,
                        "data": page_data
                    })
                current_page_number = element.metadata.page_number
                page_data = []

            # Format each element
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

        # Append the last page's data
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