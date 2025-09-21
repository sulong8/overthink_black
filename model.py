from typing import List
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv
import os
from prompts import *
import json
import asyncio
import re
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")


class TargetModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    def generate(self, prompt):
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        chat_usage = None
        if hasattr(completion, "usage") and completion.usage is not None:
            chat_usage = completion.usage
        return completion.choices[0].message.content, chat_usage
    def agenerate(self, prompts: List[str], n_samples=1):
        async def _agenerate(prompt):
            tasks = [
                self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False,
             ) for _ in range(n_samples)
            ]
            completions = await asyncio.gather(*tasks)
            contents = [completions[i].choices[0].message.content for i in range(n_samples)]
            chat_usages = [completions[i].usage for i in range(n_samples) if hasattr(completions[i], "usage") and completions[i].usage is not None]
            return contents, chat_usages if len(chat_usages) > 0 else None
        async def run():
            tasks = [_agenerate(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks)
            return results
        return asyncio.run(run())
    

class AttackerModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    def generate(self, prompt) -> str:
        """生成攻击性问题"""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content

    async def agenerate(self, prompt) -> str:
        """异步生成攻击性问题"""
        completion = await self.async_client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content

    def split_question(self, questions):
        async def run():
            tasks = [self.asplit_question(q) for q in questions]
            return await asyncio.gather(*tasks)
        return asyncio.run(run())

    async def asplit_question(self, question):
        """异步将问题拆分成前提与提问"""
        prompt = SPLIT_QUESTION.format(question=question)
        response :str = await self.agenerate(prompt)
        clean_text = re.sub(r"^```json|```$", "", response.strip(), flags=re.MULTILINE).strip()
        res = json.loads(clean_text)
        return res

    
    

    
    
    