"""
Tip 1 Diyabet Karbonhidrat Sayım Asistanı
Görev: Anlamsal Arama Testi (Çarşamba - Eylül & Meryem eşli)

Bu script, dün oluşturduğumuz ./chroma_db veritabanını açıp
örnek sorularla test eder. Amaç: veritabanının doğru besin
kayıtlarını "anlamca" bulup bulamadığını görmek.
"""

import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY .env dosyasında bulunamadı!"


def veritabanini_yukle(kayit_yolu="./chroma_db"):
    """Diskte kayıtlı olan Chroma veritabanını (yeniden embedding yapmadan) açar."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectordb = Chroma(
        persist_directory=kayit_yolu,
        embedding_function=embeddings,
        collection_name="karbonhidrat_koleksiyonu",
    )
    return vectordb


def arama_yap(vectordb, soru, kac_sonuc=3):
    """Verilen soruya en yakın anlamdaki kayıtları getirir."""
    sonuclar = vectordb.similarity_search(soru, k=kac_sonuc)
    print(f"\n🔍 Soru: '{soru}'")
    print("-" * 50)
    for i, sonuc in enumerate(sonuclar, 1):
        print(f"{i}) {sonuc.page_content}")
        print(f"   [kaynak: {sonuc.metadata.get('kaynak', 'bilinmiyor')}]")
    print()


if __name__ == "__main__":
    vectordb = veritabanini_yukle()

    # Deneme soruları -- istediğin gibi çoğaltabilirsin
    test_sorulari = [
        "1 dilim ekmek kaç gram karbonhidrat içerir?",
        "elma",
        "mercimek çorbası",
        "1 su bardağı süt",
        "big mac",
        "whopper",
        "cornetto dondurma",
]
    

    for soru in test_sorulari:
        arama_yap(vectordb, soru)
