from langchain.prompts import PromptTemplate

def get_ranker_prompt():
    template="""
        You are an expert technical recruiter with deep knowledge of resume screening and candidate evaluation.

        Your task: 
        Given a job description and multiple candidate resumes, analyze how well each resume matches the job requirements.
        Evaluate based on skills, experience, education, and relevance to the role.

        Instructions:
        1. Score each resume on a scale from 0 to 10 (10 = perfect match).
        2. Provide a brief justification for each score.
        3. Output the results in descending order of score.

        Job Description:
        {job_description}

        Candidate Resumes:
        {resume_list}

        Expected Output Format:
        [
        {{
            "candidate_id": "<unique_resume_identifier>",
            "score": <numeric_score>,
            "justification": "<short reason for the score>"
        }},
        ...
        ]
        """
    
    return PromptTemplate(template=template, input_variables=["job_description", "resume_list"])