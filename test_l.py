# import os
# os.environ["OPENAI_API_BASE"] = "http://localhost:1234/v1"
# os.environ["OPENAI_API_KEY"] = "test"

# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import PromptTemplate
# from langchain_core.output_parsers import StrOutputParser

# # choose model name exactly as LM Studio exposes it (check LM Studio UI)
# llm = ChatOpenAI(model="qwen/qwen3-4b-thinking-2507", temperature=0.2)  

# prompt = PromptTemplate(input_variables=["q"], template="Q: {q}\nA:")
# chain = prompt | llm | StrOutputParser()

# # print(chain.run("Explain recursion in 2 sentences."))

# print(chain.invoke("Who are you?"))


from langchain_core.language_models import BaseLanguageModel
from typing import Optional, List, Any, Mapping
import requests


class LMStudioLLM(BaseLanguageModel):
    """LangChain-compatible LLM wrapper for LM Studio."""

    def __init__(self, endpoint: str = "http://localhost:1234/v1/chat/completions",
                 model: str = "qwen/qwen3-4b-2507",
                 temperature: float = 0.7):
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "lmstudio"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "stream": False
        }

        try:
            response = requests.post(self.endpoint, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            raise ValueError(f"Error calling LM Studio API: {e}")

llm = LMStudioLLM()

prompt = "Explain the difference between supervised and unsupervised learning."
response = llm.invoke(prompt)

print(response)
