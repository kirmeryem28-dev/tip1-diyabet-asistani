"""
Tip 1 Diyabet Karbonhidrat Sayım Asistanı
Görev: Vektör Veritabanı Oluşturma (Çarşamba - Eylül)

Bu script, karbonhidrat_veritabani_birlesik.json içindeki her besini
LangChain Document nesnesine çevirir, OpenAI embedding'leriyle
vektörleştirir ve ./chroma_db klasörüne kalıcı olarak kaydeder.

ÖNEMLİ: JSON dosyasını her güncellediğinde (yeni besin ekleme, değer
düzeltme vb.) bu script'i TEKRAR çalıştırman gerekir - yoksa
anlamsal_arama_testi.py hâlâ eski veriyle çalışır.

anlamsal_arama_testi.py ile UYUMLULUK (degistirme!):
- persist_directory = "./chroma_db"
- collection_name   = "karbonhidrat_koleksiyonu"
- embedding modeli  = "text-embedding-3-small"

KURULUM
-------
pip install langchain langchain-openai langchain-chroma python-dotenv --break-system-packages

CALISTIRMA
----------
python3 vektor_veritabani_olustur.py
"""

import json
import os
import shutil

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY .env dosyasında bulunamadı!"

JSON_YOLU = "karbonhidrat_veritabani_birlesik.json"
CHROMA_YOLU = "./chroma_db"
KOLEKSIYON_ADI = "karbonhidrat_koleksiyonu"


def besinleri_yukle(json_yolu: str) -> list:
    """Ana JSON veritabanini okur, besinler listesini dondurur."""
    with open(json_yolu, encoding="utf-8") as f:
        data = json.load(f)
    return data["besinler"]


def besin_to_document(besin: dict) -> Document:
    """Tek bir besin kaydini, anlamsal aramanin iyi calismasi icin
    dogal dilde bir metne (page_content) ve metadata'ya cevirir."""
    agirlik = besin.get("agirlik_g")
    agirlik_metni = f"{agirlik} gram" if agirlik not in (None, "") else "belirtilmemiş ağırlık"

    icerik = (
        f"Besin: {besin['besin_adi']}. "
        f"Kategori: {besin.get('kategori', 'belirtilmemiş')}. "
        f"Porsiyon ölçüsü: {besin.get('degisim_olcusu', 'belirtilmemiş')} ({agirlik_metni}). "
        f"Karbonhidrat miktarı: {besin['karbonhidrat_g']} gram. "
        f"Kaynak: {besin.get('kaynak', 'belirtilmemiş')}."
    )

    metadata = {
        "besin_adi": besin["besin_adi"],
        "kategori": besin.get("kategori", ""),
        "karbonhidrat_g": besin["karbonhidrat_g"],
        "agirlik_g": agirlik if agirlik is not None else "",
        "kaynak": besin.get("kaynak", ""),
    }

    return Document(page_content=icerik, metadata=metadata)


def vektor_veritabani_olustur(belgeler: list, chroma_yolu: str, koleksiyon_adi: str):
    """Belgeleri embed edip Chroma'ya kalici olarak kaydeder.
    Eski veritabani varsa temizler (cakisan/eski kayit kalmamasi icin)."""
    if os.path.exists(chroma_yolu):
        print(f"Eski veritabani bulundu, siliniyor: {chroma_yolu}")
        shutil.rmtree(chroma_yolu)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"{len(belgeler)} besin embed ediliyor, bu birkaç dakika sürebilir...")
    vectordb = Chroma.from_documents(
        documents=belgeler,
        embedding=embeddings,
        persist_directory=chroma_yolu,
        collection_name=koleksiyon_adi,
    )
    print(f"Tamamlandı. Vektör veritabanı kaydedildi: {chroma_yolu}")
    return vectordb


def main():
    besinler = besinleri_yukle(JSON_YOLU)
    print(f"{JSON_YOLU} okundu, {len(besinler)} besin bulundu.")

    belgeler = [besin_to_document(b) for b in besinler]

    vektor_veritabani_olustur(belgeler, CHROMA_YOLU, KOLEKSIYON_ADI)

    print()
    print("Şimdi 'python3 anlamsal_arama_testi.py' ile test edebilirsin.")


if __name__ == "__main__":
    main()
