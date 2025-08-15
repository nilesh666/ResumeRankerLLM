from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from src.prompt_template import get_ranker_prompt
from dotenv import load_dotenv
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from collections import defaultdict
from qdrant_client import QdrantClient
import os

load_dotenv()

api_key = os.getenv("GROQ_API")


class ResumeRanker:
    def __init__(self, qdrant_url="http://localhost:6333", collection_name="ResumeCollections"):
        self.embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        self.llm = ChatGroq(api_key = api_key, model_name="llama-3.1-8b-instant", temperature=0) 
        self.prompt = get_ranker_prompt()
        self.vector_db = QdrantVectorStore(
            client = QdrantClient(url="http://localhost:6333"),
            collection_name=collection_name,
            embedding=self.embedding
        )

    def retrieve_and_group(self, jd, top_k=10):
        r = self.vector_db.as_retriever(search_kwargs={"k": top_k})
        docs = r.invoke(jd)

        grp = defaultdict(list)
        for doc in docs:
            cid = doc.metadata.get("candidate_id", "unknown")
            grp[cid].append(doc.page_content)

        merged_resumes = [
            f"Candidate ID: {cid} \n\n" + "\n".join(chunks) for cid, chunks in grp.items()
        ]

        return "\n\n".join(merged_resumes)
    
    def rank_resumes(self, jd):
        resumes_text = self.retrieve_and_group(jd)
        chain = self.prompt | self.llm

        return chain.invoke(
            {"job_description": jd, 
             "resume_list": resumes_text
             })

if __name__ == "__main__":
    ranker = ResumeRanker()

    job_desc = "Looking for a Python developer with experience in data science, machine learning, and cloud deployment."
    results = ranker.rank_resumes(job_desc)

    print(results.content)
