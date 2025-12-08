from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import dotenv

dotenv.load_dotenv()

ollama_llm = ChatOllama(
    base_url=os.getenv("OLLAMA_BASE_URL"), model=os.getenv("OLLAMA_LLM_MODEL")
)

gemini_llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_LLM_MODEL"), google_api_key=os.getenv("GOOGLE_API_KEY")
)


def get_llm_instance(provider: str = "ollama"):
    return gemini_llm if provider.strip().lower() == "gemini" else ollama_llm


if __name__ == "__main__":

    print("--" * 30 + "\n\n")
    print(ollama_llm.invoke("How good is ollama?"))
    print("\n\n" + "--" * 30 + "\n\n")
    print(ollama_llm.invoke("How good is Gemini?"))
