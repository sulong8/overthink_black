
SPLIT_QUESTION = """
Extract all explicit premises (facts/assumptions) and question from the following problem.  
    Do not solve the problem or provide any answer.  
    Return them strictly as a JSON list and a question of strings.  
Example:
'''json
    "premises": [...],
    "question": "..."
'''
Problem:
{question}
"""
