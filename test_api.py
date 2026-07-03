import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("❌ OPENAI_API_KEY bulunamadı! .env dosyasını kontrol et.")

print("✅ API key yüklendi, bağlantı test ediliyor...")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
response = llm.invoke("Merhaba! Sen kimsin, tek cümlede tanıt.")

print("🤖 Model yanıtı:")
print(response.content)
