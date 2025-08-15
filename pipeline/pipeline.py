from src.ranker import ResumeRanker
from src.vector_store import VectorStore
from src.prompt_template import get_ranker_prompt
from utils.custom_exception import CustomException
from utils.logger import logging
import sys 

class RAGchain:
    def __init__(self, jd, resumes):
        self.jd = jd
        self.resumes=resumes

    def pipeline(self):
        try:
            logging.info("Pipeline initializing")

            vecotor_store = VectorStore(self.resumes)
            vecotor_store.run()

            logging.info("Ranking started")
            ranker = ResumeRanker()
            results = ranker.rank_resumes(self.jd)

            logging.info("Pipeline finished ranking the resumes")
            return results.content
                
        except Exception as e:
            raise CustomException(e, sys)
