from dotenv import load_dotenv
import os
from langchain_google_genai import ( #connect langchain to gemini
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings
)   
from langchain_qdrant import QdrantVectorStore # search your qudarnt collection
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "course_documents-2"
CHAT_MODEL = 'gemini-3.5-flash-lite'

from pypdf import PdfReader

def get_printed_page(doc):
    pdf_page_index = int(doc.metadata["page"])
    source = doc.metadata["source"]
    text = PdfReader(source).pages[pdf_page_index].extract_text() or ""
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    return first_line if first_line.isdigit() else "unknown"


def main():
    load_dotenv()

    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        raise RuntimeError('Add Gemini api key to your .env file')

    question = input('Ask a question about your pdf: ').strip()
    if not question:
        print('Please enter the question')
        return

    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model=EMBEDDING_MODEL,
    #     task_type='RETRIEVAL_QUERY',
    #     google_api_key = api_key,
    # )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    ) 
    # model hosted on Hugging Face → run by Sentence Transformers → connected to your RAG code through LangChain.

    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url='http://localhost:6333'
    )

    matching_chunks = vector_store.similarity_search(query=question, k = 8)
    print(matching_chunks[0])

    if not matching_chunks:
        print("I couldn't find relevant information in ingested documents")
        return

    context = []
    source_pages = []

    for doc in matching_chunks:
        book_page = get_printed_page(doc)
        context.append(f"[Printed book page {book_page}]\n{doc.page_content}")
        source_pages.append(book_page)


    PROMPT = f'''
    You are an expert AI assistant who takes the user query and relevant information from vector database and reply the user based on that available context retrived from the pdf file along with the page content and page number.
    Do not invent page citations.
    Do not answer if something is not present in context.

    Rule: Cite only printed book page labels shown in the context. Do not invent page numbers.
    After you send the response, attach the printed page number for the results every time.

    Context:
    {context}

    Question:
    {question}
    '''

    chatModel = ChatGoogleGenerativeAI(
        model = CHAT_MODEL,
        google_api_key = api_key,
        temperature=0
    )
    response = chatModel.invoke(PROMPT)
    print('\nAnswer:')
    print(response.text)

if __name__ == "__main__":
    main()