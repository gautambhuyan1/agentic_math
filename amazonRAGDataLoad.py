import os
#from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv


def load_and_store_in_vectorDb2():
  
  sourceData = PyPDFLoader("./1005618655.pdf")

  # Define the text splitter
  splitData = RecursiveCharacterTextSplitter(
    #separators=["\n\n", "\n", " ", ""],
    separators=[" "],
    chunk_size=500,
    chunk_overlap=10
  )

  # Load and split the PDF content
  #cleanedData = sourceData.load_and_split(text_splitter=splitData)
  #cleanedData = sourceData.load()
  cleanedData = sourceData.load_and_split(text_splitter=splitData)

  # Debug: Print the number of chunks and sample content
  print(f"#####Number of chunks: {len(cleanedData)}")
  print("#####Sample chunk content:", cleanedData[0].page_content if cleanedData else "No chunks")

  # Initialize the embedding model
  dataEmbeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

  # Initialize VectorstoreIndexCreator
  dataIndex = VectorstoreIndexCreator(
    text_splitter=splitData,
    embedding=dataEmbeddings,
    vectorstore_cls=FAISS
  )

  # Create the index from the loader
  indexDb = dataIndex.from_loaders([sourceData])
  return indexDb

def load_and_store_in_vectorDb():
  # Verify PDF file exists
    pdf_path = "./1005618655.pdf"
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    # Load the PDF with UnstructuredPDFLoader for better handling of scanned PDFs
    try:
        #sourceData = UnstructuredPDFLoader(pdf_path, mode="elements")
        sourceData = PyPDFLoader(pdf_path, mode="elements")
        cleanedData = sourceData.load()
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return None

    # Define the text splitter
    splitData = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=500,  # Increased for better context
        chunk_overlap=50
    )

    # Split the loaded documents
    splitDocs = splitData.split_documents(cleanedData)

    # Debug: Print the number of chunks and sample content
    print(f"##### Number of chunks: {len(splitDocs)}")
    print("##### Sample chunk content:", splitDocs[0].page_content if splitDocs else "No chunks")

    if not splitDocs:
        print("Error: No text extracted from PDF. Check if the PDF is scanned or empty.")
        return None

    # Initialize the embedding model
    dataEmbeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Initialize VectorstoreIndexCreator
    dataIndex = VectorstoreIndexCreator(
        text_splitter=splitData,
        embedding=dataEmbeddings,
        vectorstore_cls=FAISS
    )

    # Create the index from documents (not loaders)
    indexDb = dataIndex.from_documents(splitDocs)

    return indexDb


def code_LLM():
  llm = init_chat_model(
    "anthropic:claude-3-5-haiku-latest"
  )
  '''
  llm = Bedrock(
    credentials_profile_name='default',
    model_id='anthropic:claude-3-5-haiku-latest',
    model_kwargs={
      "max_tokens_to_sample":300,
      "temperature": 0.1,
      "top_p": 0.9
    }
  )
  '''
  return llm

def code_RAG_response(index, query):
  ragLlm = code_LLM()
  codeRAGQuery = index.query(question=query, llm=ragLlm)
  return codeRAGQuery

load_dotenv()

dbIndex = load_and_store_in_vectorDb2()

userQuery = input("Enter your query")

response = code_RAG_response(dbIndex, userQuery)

print("Response: ", response)
  
