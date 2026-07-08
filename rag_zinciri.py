"""
Görev: Beyin ve Hafıza Birleşimi - İlk RAG Zinciri (Çarşamba - Meryem)
Vektör veritabanından gelen bilgiyi + System Prompt'u birleştirip
OpenAI'ye gönderen zinciri kurar.
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY .env dosyasında yok!"

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    SYSTEM_PROMPT = f.read()
    
# 1) Vektör veritabanını aç (dün oluşturduğun ./chroma_db)
import streamlit as st
from vektor_veritabani_olustur import (
    pdf_chunklarini_hazirla,
    json_besinlerini_documentlere_cevir,
    vektor_veritabani_olustur,
)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

@st.cache_resource(show_spinner="Veritabanı hazırlanıyor...")
def vektordb_getir():
    """
    Cloud ortamında chroma_db diske kalıcı değil (veya hiç yok).
    Bu yüzden her uygulama başlangıcında (ilk kullanıcı için) veritabanını
    JSON'dan yeniden inşa ediyoruz. st.cache_resource sayesinde bu işlem
    uygulama ayakta kaldığı sürece SADECE 1 KEZ çalışır, her soru için değil.
    """
    mevcut_db = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings,
        collection_name="karbonhidrat_koleksiyonu",
    )

    # Koleksiyon boşsa (ilk çalıştırma / cloud'da chroma_db yok) sıfırdan kur
    if mevcut_db._collection.count() == 0:
        pdf_chunklari = pdf_chunklarini_hazirla(pdf_klasoru="data/")
        json_documentleri = json_besinlerini_documentlere_cevir(
            "karbonhidrat_veritabani_birlesik.json"
        )
        mevcut_db = vektor_veritabani_olustur(pdf_chunklari, json_documentleri)

    return mevcut_db

vectordb = vektordb_getir()

# 1) Semantic (anlam bazlı) retriever -- dolaylı ifadeler, eş anlamlılar için güçlü
semantic_retriever = vectordb.as_retriever(search_kwargs={"k": 5})

# 2) BM25 (anahtar kelime bazlı) retriever -- tam/yakın kelime eşleşmeleri için güçlü
#    BM25Retriever kendi metin indeksini ister, bu yüzden Chroma'daki tüm kayıtları
#    tekrar Document nesnesine çevirip ona veriyoruz.
tum_kayitlar = vectordb.get(include=["documents", "metadatas"])
belgeler = tum_kayitlar.get("documents") or []
metalar = tum_kayitlar.get("metadatas") or [{}] * len(belgeler)
bm25_dokumanlari = [
    Document(page_content=icerik, metadata=(meta if meta else {}))
    for icerik, meta in zip(belgeler, metalar)
    if icerik  # boş string'leri atla
]
bm25_retriever = BM25Retriever.from_documents(bm25_dokumanlari)
bm25_retriever.k = 5

# 3) İkisini birleştir. Semantic'e daha yüksek ağırlık veriyoruz (0.7) çünkü saf
#    kelime eşleşmesi (BM25) "orta" gibi ortak kelimelerle yanlış kaydı (ör. "elma"
#    yerine "elma kompostosu") öne çıkarabiliyor. BM25'in ağırlığı (0.3) sadece
#    semantic'in hiç bulamadığı durumlarda (typo, "dolma" gibi) devreye girsin diye.
retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, semantic_retriever],
    weights=[0.3, 0.7],
)

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
5. Bağlamda birbirine benzer birden fazla kayıt varsa bu sırayla uygula:
   ADIM 1 — Sorudaki ürün/marka adına TAM eşleşen kaydı bul:
   - "Whopper" sorulmuşsa → "Whopper Jr." değil, "Whopper Sandviç" kaydını seç (53g)
   - "patates" sorulmuşsa → "patates püresi" değil, "pişmiş patates" kaydını seç (20g)
   - "elma" sorulmuşsa → "Elma Kompostosu" değil, "Elma (Meyve Grubu)" kaydını seç (15g)
   Genel kural: sorudaki kelime "Jr.", "püresi", "kompostosu" gibi ek kelime içermiyorsa,
   o ek kelimeyi içeren varyantları SEÇME.
   ADIM 2 — Sorudaki boy/ölçü bağlamda yoksa (ör. "orta boy elma" ama kayıtta "küçük boy"):
   mevcut en yakın kaydı kullan ve "elimdeki kayıtta küçük boy için 15g geçiyor" de.
   "net bilgi bulamadım" ASLA deme — boy farkı "bilgi yokluğu" değildir.
6. Miktar/birim dönüşümü gerektiren sorularda (ör. "500 ml", "2 bardak", "1.5 porsiyon",
   "yarım simit"):
   a) Bağlamda o birime EN YAKIN TEKİL kaydı bul.
      - "yarım simit" → 1/4 simit kaydını bul, 2 ile çarp (1/4 × 2 = 1/2)
      - "2 su bardağı ayran" → 1 su bardağı (200cc) kaydını bul, 2 ile çarp
      - "500 ml kola" → 100cc kaydını bul, 5 ile çarp
   b) Hesaplamayı adım adım göster: "[birim]'de [X]g ise [istenilen miktar]'da [X × çarpan] = [sonuç]g"
   c) Çarpım sonucunu tekrar kontrol et; sonuç mantıklı görünmüyorsa "net hesaplayamıyorum,
      diyetisyeninize danışın" de.
   d) Asla "ama bu miktar şu birim için geçerlidir" gibi kendini çelen bir cümle kurma.
      Tek bir net sayı ver.
7. Her yanıtının en altına şu uyarıyı MUTLAKA ekle:
   "⚠️ Bu bir tıbbi tavsiye değildir, doktorunuza/diyetisyeninize danışın."

BAĞLAM:
{context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{soru}"),
])

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    timeout=30,        
    max_retries=2,     
)

def baglami_birlestir(dokumanlar):
    """Retriever'ın bulduğu Document'ları numaralandırarak tek bir metin bloğuna çevirir."""
    parcalar = []
    for i, d in enumerate(dokumanlar, 1):
        parcalar.append(f"[Kayıt {i}]\n{d.page_content}")
    return "\n\n".join(parcalar)

# 3) Zinciri kur: soru -> retriever -> prompt -> llm -> metin
rag_zinciri = (
    {"context": retriever | baglami_birlestir, "soru": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def cevap_al(soru: str) -> str:
    """
    app.py'nin çağırdığı ana fonksiyon: kullanıcının sorusunu alır,
    RAG zincirinden geçirir ve metin cevabı döner.
    """
    return rag_zinciri.invoke(soru)

if __name__ == "__main__":
    test_sorulari = [
        "1 orta boy elma kaç gram karbonhidrat içerir?",
        "1 orta boy elma kaç gram karbonhidrat içerir?",
        "1 orta boy elma kaç gram karbonhidrat içerir?",
        "1 orta boy elma kaç gram karbonhidrat içerir?",
        "elma kaç karbonhidrat",
    ]
    for soru in test_sorulari:
        print(f"\n🙋 Soru: {soru}")
        cevap = rag_zinciri.invoke(soru)
        print(f"🤖 Cevap: {cevap}")
        print("-" * 60)
        
           
             