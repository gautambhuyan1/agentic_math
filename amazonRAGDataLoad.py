import os
from langchain.documentloaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


sourceData = PyPDFLoader("source")

#cleanedData = sourceData.load_and_split()

splitData = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=100, chunk_overlap=10, sourceData
#print("Length of data = ", len(cleanedData))

print("Sample data = ", cleanedData[0])
