from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient, models
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from utils.logger import logging
from utils.custom_exception import CustomException
import sys
import uuid

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
        doc = [Document(page_content = text, metadata={"source": "direct_input", "candidate_id": str(uuid.uuid4())}) for text in self.resume_texts]
        
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


if __name__=="__main__":
    sample_texts = [
                        
                """John Doe
                Email: john.doe@example.com
                Phone: +91 9876543210

                Summary:
                Data Scientist with 5 years of experience in machine learning, NLP, and big data analytics.

                Skills:
                - Python, R, SQL
                - TensorFlow, PyTorch
                - AWS, GCP
                - Data Visualization (Tableau, Power BI)

                Experience:
                Senior Data Scientist | XYZ Corp | 2020 - Present
                - Built predictive models to improve customer retention by 15%
                - Designed ETL pipelines for large-scale data processing""",

                """Priya Sharma
                Email: priya.sharma@example.com
                Phone: +91 9123456780

                Summary:
                Full Stack Developer with expertise in building scalable web applications and cloud-native solutions.

                Skills:
                - JavaScript, TypeScript, Python
                - React, Node.js, Django
                - AWS, Docker, Kubernetes
                - PostgreSQL, MongoDB

                Experience:
                Software Engineer | ABC Technologies | 2019 - Present
                - Developed an e-commerce platform serving 1M+ monthly users
                - Optimized backend APIs, reducing response time by 40%""",

                """Rajesh Kumar
                Email: rajesh.kumar@example.com
                Phone: +91 9988776655

                Summary:
                Data Engineer specializing in building data pipelines and real-time analytics systems.

                Skills:
                - Python, Scala, Java
                - Apache Spark, Kafka
                - Airflow, dbt
                - Azure, AWS

                Experience:
                Data Engineer | DataFlow Inc | 2021 - Present
                - Designed real-time streaming pipelines for IoT data
                - Reduced ETL job runtime from 2 hours to 20 minutes"""

            ]
    
    a = VectorStore(sample_texts)
    a.run()
        



