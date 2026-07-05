"""
Görev: OpenAI Testi (Salı - Meryem)
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY .env dosyasında yok!"

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

if __name__ == "__main__":
    soru = "1 dilim tam buğday ekmeği yaklaşık kaç gram karbonhidrat içerir?"
    cevap = llm.invoke(soru)
    print("🤖 Yanıt:")
    print(cevap.content)
    