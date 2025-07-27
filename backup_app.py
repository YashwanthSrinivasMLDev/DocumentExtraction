import streamlit as st
import json
from document_extractor import DocumentExtractor

#dummy comment
# We need to instantiate the class once.
# Streamlit caches objects by default, but we can be explicit.
@st.cache_resource
def get_extractor():
    """Returns a singleton instance of the DocumentExtractor."""
    return DocumentExtractor()


# Set up the Streamlit page
st.set_page_config(
    page_title="Document Extraction AI",
    page_icon="📄",
    layout="wide"
)

st.title("Document Extraction AI")
st.subheader("Extract structured data from your PDFs and images.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a document...", type=["pdf", "jpg", "jpeg", "png"])

if uploaded_file is not None:
    # A Streamlit file uploader returns a BytesIO object.
    # We'll save it to a temporary file to be processed by deepdoctection.
    with open("temp_doc.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Get the extractor instance
    extractor = get_extractor()

    # Create a button to trigger the extraction
    if st.button("Start Extraction"):
        with st.spinner("Processing document... This may take a few moments."):
            try:
                # Call the extraction method on the temporary file
                extracted_data = extractor.extract_from_document("temp_doc.pdf")

                st.success("Extraction complete!")

                # Display the results
                st.subheader("Extracted Content")

                # Use st.json to display the full, structured output
                with st.expander("View Raw JSON Output"):
                    st.json(extracted_data)

                # Now, display a user-friendly summary
                st.subheader("Summary")
                for page_data in extracted_data["elements"]:
                    st.write(f"**Page {page_data['page_number']}**")
                    for element in page_data["data"]:
                        if element["type"] == "title":
                            st.info(f"**Title:** {element['text']}")
                        elif element["type"] == "text":
                            st.write(f"**Text:** {element['text'][:200]}...")  # Display first 200 chars
                        elif element["type"] == "table":
                            st.success(f"**Table Found:**")
                            st.code(element['table_data'])  # Display table data as code

            except Exception as e:
                st.error(f"An error occurred during extraction: {e}")