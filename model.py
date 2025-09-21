from typing import List, Dict
import openai
from openai import OpenAI, AsyncOpenAI
import time
from dotenv import load_dotenv
import os
from prompts import *
import re
import asyncio
from fastchat.conversation import Conversation
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
    def agenerate(self, prompts: List[str]):
        async def _agenerate(prompt):
            completion = await self.async_client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                stream=False
            )
            chat_usage = None
            if hasattr(completion, "usage") and completion.usage is not None:
                chat_usage = completion.usage
            return completion.choices[0].message.content, chat_usage
        async def run():
            tasks = [_agenerate(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks)
            return results
        return asyncio.run(run())

class AttackerModel:
    def __init__(self, model_name):
        self.model_name = model_name
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    
    def generate(self, prompt) -> str:
        """生成攻击性问题"""
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return completion.choices[0].message.content

    def split_question(self, question):
        """将问题拆分成前提与提问"""
        premises, ask = None, None
        prompt = SPLIT_QUESTION.format(question)
        response :str = self.generate(prompt)
        # Extract text between <premise> tags
        premise_match = re.search(r'<premise>(.*?)</premise>', response, re.DOTALL)
        if premise_match:
            premises = premise_match.group(1).strip().split("\n")
        ask_match = re.search(r'<ask>(.*?)</ask>', response, re.DOTALL)
        if ask_match:
            ask = ask_match.group(1).strip()
        return premises, ask

class DeepSeek:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 10000

    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.model_name = "deepseek-chat"
        self.max_tokens = 8192

    def generate(self, conv: Conversation,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        for _ in range(self.API_MAX_RETRY):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv.to_openai_api_messages(),
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    timeout=self.API_TIMEOUT
                )
                end_time = time.time()
                sum_time = end_time - start_time
                break
            except openai.OpenAIError as e:
                print(f"An error occurred: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return response.to_json(), sum_time

    def batched_generate(self,
                         conv_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0, ):
        texts = []
        times = []

        for conv in conv_list:
            text, sum_time = self.generate(conv, max_n_tokens, temperature, top_p)
            texts.append(text)
            times.append(float(sum_time))
        return texts, times

class Qwen:
    API_RETRY_SLEEP = 10
    API_ERROR_OUTPUT = "$ERROR$"
    API_QUERY_SLEEP = 0.5
    API_MAX_RETRY = 5
    API_TIMEOUT = 10000

    def __init__(self):
        BASE_URL = os.getenv("Qwen_BASE_URL")
        API_KEY = os.getenv("Qwen_API_KEY")
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        self.model_name = "qwen3-235b-a22b-instruct-2507"
        self.max_tokens = 8192

    def generate(self, conv: Conversation,
                 max_n_tokens: int,
                 temperature: float,
                 top_p: float):
        for _ in range(self.API_MAX_RETRY):
            try:
                start_time = time.time()
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=conv.to_openai_api_messages(),
                    max_tokens=self.max_tokens,
                    temperature=temperature,
                    timeout=self.API_TIMEOUT
                )
                end_time = time.time()
                sum_time = end_time - start_time
                break
            except openai.OpenAIError as e:
                print(f"An error occurred: {e}")
                time.sleep(self.API_RETRY_SLEEP)
            time.sleep(self.API_QUERY_SLEEP)
        return response.to_json(), sum_time

    def batched_generate(self,
                         conv_list: List[List[Dict]],
                         max_n_tokens: int,
                         temperature: float,
                         top_p: float = 1.0, ):
        texts = []
        times = []

        for conv in conv_list:
            text, sum_time = self.generate(conv, max_n_tokens, temperature, top_p)
            texts.append(text)
            times.append(float(sum_time))
        return texts, times