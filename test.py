import asyncio, os
from utils.llm import get_llm_client
from utils.logger import Logger
import argparse

async def main():
    args = argparse.Namespace(
        provider="ollama",
        model="llama3.2:latest",
        api=None,
        use_api=False,
        thinking=False,
    )
    logger = Logger(log_file="./tmp_ollama_test.log")
    client = get_llm_client(args, logger)
    resp = await client.generate_completion([
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":"Say 'ok' only."},
    ])
    print(resp["content"])
    print(resp["usage"])

asyncio.run(main())
