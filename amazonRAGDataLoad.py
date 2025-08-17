import os
from langchain.documentloaders import PyPDFLoader

sourceData = PyPDFLoader("source")

cleanedData = sourceData.load_and_split()

print("Length of data = ", len(cleanedData))

print("Sample data = ", cleanedData[0])
