from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")

    loader = TextLoader("/Users/surabhichanchal/Documents/Langchain/Project_Langchain/mediumblog1.txt" , encoding = "utf-8")
    documents = loader.load()

    print("Splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key = os.environ.get("OPENAI_API_KEY"))

    print("embedding....")

    PineconeVectorStore.from_documents(texts , embeddings , index_name = os.environ.get("INDEX_NAME"))
    print("Pinecone vector store created")

    print("done")




