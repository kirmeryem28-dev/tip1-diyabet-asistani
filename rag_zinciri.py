"""
Görev: Beyin ve Hafıza Birleşimi - İlk RAG Zinciri (Çarşamba - Meryem)
Vektör veritabanından gelen bilgiyi + System Prompt'u birleştirip
OpenAI'ye gönderen zinciri kurar.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY .env dosyasında yok!"

# 1) Vektör veritabanını aç (dün oluşturduğun ./chroma_db)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings,
    collection_name="karbonhidrat_koleksiyonu",
)
retriever = vectordb.as_retriever(search_kwargs={"k": 3})  # en yakın 3 kayıt

# 2) System Prompt'u şablona koy
SYSTEM_PROMPT = """Sen "KarbonAsistan" adında, Tip 1 Diyabetli bireylere ve ailelerine 
karbonhidrat sayımında yardımcı olan bir yapay zeka asistanısın.

KURALLAR:
1. SADECE sana verilen bağlamdaki bilgiyi kullan. Bağlam dışına ASLA çıkma.
2. Bağlamda cevap yoksa şunu söyle: 
   "Bu konuda elimdeki kaynaklarda net bir bilgi bulamadım, lütfen diyetisyeninize danışın."
   ASLA tahmin yürütme veya uydurma.
3. Yanıtların kısa, net ve sade olsun. Tıbbi jargon kullanma.
4. Karbonhidrat değerini verirken porsiyon/ölçüyü de belirt.
5. Her yanıtının en altına şu uyarıyı MUTLAKA ekle:
   "⚠️ Bu bir tıbbi tavsiye değildir, doktorunuza/diyetisyeninize danışın."

BAĞLAM:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{soru}"),
])

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

def baglami_birlestir(dokumanlar):
    """Retriever'ın bulduğu Document'ları tek bir metin bloğuna çevirir."""
    return "\n\n".join(d.page_content for d in dokumanlar)

# 3) Zinciri kur: soru -> retriever -> prompt -> llm -> metin
rag_zinciri = (
    {"context": retriever | baglami_birlestir, "soru": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    test_sorulari = [
        "1 dilim ekmek kaç gram karbonhidrat içerir?",
        "Big Mac yedim, ne kadar karbonhidrat almış olabilirim?",
        "Uzaydaki astronotlar ne yer?",  
        "hamburger yedim,kaç karbonhidrat almışımdır?"
        
    ]

    for soru in test_sorulari:
        print(f"\n🙋 Soru: {soru}")
        cevap = rag_zinciri.invoke(soru)
        print(f"🤖 Cevap: {cevap}")
        print("-" * 60)