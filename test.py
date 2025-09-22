from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
import asyncio
load_dotenv()
BASE_URL = os.getenv("BASE_URL")
API_KEY = os.getenv("API_KEY")
async_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

async def agenerate(prompt: str, n_samples=1):
    tasks = [
        async_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        stream=False,
    ) for _ in range(n_samples)
    ]
    completions = await asyncio.gather(*tasks)
    contents = [completions[i].choices[0].message.content for i in range(n_samples)]
    chat_usages = [completions[i].usage for i in range(n_samples) if hasattr(completions[i], "usage") and completions[i].usage is not None]
    return contents, chat_usages if len(chat_usages) > 0 else None

if __name__ == "__main__":
    res = asyncio.run(agenerate("1 + 1 = ?", n_samples=3))
    print(res)