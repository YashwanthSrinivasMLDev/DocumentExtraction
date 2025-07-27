import streamlit as st
import json
import os
from document_extractor import DocumentExtractor


# We need to instantiate the class once.
@st.cache_resource
def get_extractor():
    """Returns a singleton instance of the DocumentExtractor."""
    return DocumentExtractor()


# Set up the Streamlit page
st.set_page_config(page_title="Document Extraction App", layout="wide")

st.title("Document Extraction App")
st.write("Upload a PDF to extract its content.")

# File uploader
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file:
    # Save the uploaded file to a temporary location
    temp_dir = "temp_pdfs"
    os.makedirs(temp_dir, exist_ok=True)
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Get the extractor instance
    extractor = get_extractor()

    # Create a button to trigger the extraction
    if st.button("Start Extraction"):
        st.info("Extraction started...")
        try:
            # The extraction logic
            extracted_info = extractor.extract_from_document(file_path)

            st.success("Extraction complete!")

            # Display the summary of the extraction
            st.subheader("Extraction Summary")

            # The output now uses the 'pages' key, not 'elements'
            for page in extracted_info['pages']:
                st.write(f"**Page {page['page_number']}:**")
                st.code(page['text'][:200] + "...")

            # Show a download button for the full JSON
            json_string = json.dumps(extracted_info, indent=4)
            st.download_button(
                label="Download JSON",
                data=json_string,
                file_name=f"{uploaded_file.name}_extracted.json",
                mime="application/json"
            )

        except Exception as e:
            st.error(f"An error occurred during extraction: {e}")
        finally:
            # Clean up the temporary file
            os.remove(file_path)