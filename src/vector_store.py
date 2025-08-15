from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from utils.logger import logging
from utils.custom_exception import CustomException
import sys

load_dotenv()

class VectorStore:
    def __init__(self, resume_texts, qdrant_url:str="http://localhost:6333"):
        self.resume_texts = resume_texts
        self.qdrant_url = qdrant_url
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    def cleanup_collections(self):
        client = QdrantClient(url=self.qdrant_url, prefer_grpc=False)

        if client.collection_exists(collection_name="ResumeCollections"):
            client.delete_collection(collection_name="ResumeCollections")

        client.create_collection(
            collection_name="ResumeCollections",
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
        )
                

    def load_and_store(self):
        doc = [Document(page_content = text, metadata={"source": "direct_input"}) for text in self.resume_texts]
        
        splitter = CharacterTextSplitter(
            chunk_size =1000,
            chunk_overlap = 0
        )

        t = splitter.split_documents(doc)

        qdrant = Qdrant.from_documents(
            doc,
            self.embedding,
            url = self.qdrant_url,
            prefer_grpc= False,
            collection_name = "ResumeCollections"
        )

    def run(self):
        try:
            self.cleanup_collections()
            logging.info("Database cleanedup")


            logging.info("Data ingestion started")
            self.load_and_store()

            logging.info("Vector DB ready for use")

        except Exception as e:
            raise CustomException(e, sys)


# if __name__=="__main__":
#     sample_texts = [
#                 "The quick brown fox jumps over the lazy dog.",
#                 "Artificial Intelligence is transforming the way businesses operate.",
#                 "A steaming cup of coffee is the perfect start to a rainy morning.",
#                 "Mount Everest is the highest mountain in the world, located in the Himalayas.",
#                 "Python is a versatile programming language used in data science, AI, and web development.",
#                 "My name is Nilesh Nandan"
#             ]
    
#     a = VectorStore(sample_texts)
#     a.run()
        



