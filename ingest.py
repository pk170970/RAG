import os  # reads the API key and Qdrant URL from environment variables
from pathlib import Path  # checks and handles the PDF path

from dotenv import load_dotenv  # loads settings from the .env file
# from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings  # creates Gemini text embeddings
from langchain_qdrant import QdrantVectorStore  # stores LangChain documents and embeddings in Qdrant
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits page text into overlapping chunks
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "course_documents-2"

def main():
    load_dotenv()

    project_folder = Path(__file__).resolve().parent
    pdf_path = project_folder / 'materials' / 'ncc_book.pdf'

    if not pdf_path.is_file():
        raise FileNotFoundError(f'PDF not found: {pdf_path}')

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('Add Gemini api key to your .env file')

    # Load the file in the python program into pages arrays
    docs = PyPDFLoader(file_path=pdf_path).load()
    # Pdf laoder read the pdf structure and extracts, text usually one page at a time. PyPDFLoader uses the pypdf library underneath. It output is usally langchain list of document object each with page_content and metadata(source filename, page index) 
    # Secdon page may be img, text extraction can return empty text for that page. OCR need to read it.

    #Split the document into chunks
    #splitter gets the text from each page and break long text into pieces near a configured size.
    #for example: page with 2500 characters and chunk size 1000 might becomes several chunks. a shorter page may stay as one chunk. as code splits the each page seperately, so each chunk keeps that page metadata.
    splitter = RecursiveCharacterTextSplitter(
       chunk_size = 1500, #characters not tokens
       chunk_overlap = 300
    )

    chunks = splitter.split_documents(documents=docs)

    #Vector embeddings
    #use this embedding with causion: if you free plan expired, then some content don't get stored in db or not converted into vectors so result may get affected.
    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model=EMBEDDING_MODEL,
    #     task_type="RETRIEVAL_DOCUMENT",
    #     google_api_key=api_key,
    # )
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


    QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url='http://localhost:6333',
        collection_name = COLLECTION_NAME
    )
main()