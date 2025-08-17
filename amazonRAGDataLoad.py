import os
from langchain.documentloaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms.bedrock import Bedrock

def load_and_store_in_vectorDb():
  
  sourceData = PyPDFLoader("source")
  
  #cleanedData = sourceData.load_and_split()
  
  splitData = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10)
  
  dataEmbeddings = BedrockEmbeddings(credentials_profile_name='default', model_id='amazon.titan-embed-txt-v1')
  
  dataIndex = VectorstoreIndexCreator(text_splitter=splitData, embeddings=dataEmbeddings, vectorstore_cls=FAISS)
  
  indexDb = dataIndex.from_loaders([sourceData])
  
  #print("Length of data = ", len(cleanedData))
  
  #print("Sample data = ", cleanedData[0])

  return indexDb

def code_LLM():
  llm = Bedrock(
    credentials_profile_name='default',
    model_id='claude....',
    model_kwargs={
      "max_tokens_to_sample":300,
      "temperature": 0.1,
      "top_p": 0.9
    }
  )
  return llm

def code_RAG_response(index, query):
  ragLlm = code_LLM()
  codeRAGQuery = index.query(question=query, llm=ragLlm)
  return codeRAGQuery

dbIndex = load_and_store_in_vector_db()

userQuery = input("Enter your query")

response = code_RAG_response(dbIndex, userQuery)

print("Response: ", response)
  

