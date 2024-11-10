import getpass
import os
from langchain_openai import ChatOpenAI
from Tools import Tools

class Chain:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("TRANSFORMERS_QA_MODEL")
        self.llm = ChatOpenAI(api_key=api_key, model=model, temperature=0)
    
    def get_llm(self):
        return self.llm

    def get_llm_with_tools(self):
        tools = Tools()
        llm_with_tools = self.llm.bind_tools(tools.get_all_tools())
        return llm_with_tools
