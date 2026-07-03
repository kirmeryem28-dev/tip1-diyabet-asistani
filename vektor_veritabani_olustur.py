"""
Tip 1 Diyabet Karbonhidrat Sayım Asistanı
Görev: Vektör Veritabanı Oluşturma (Çarşamba - Eylül)

Bu script iki tür veriyi tek bir ChromaDB veritabanında birleştirir:
1. PDF'lerden gelen ve parçalanmış (chunk) metinler (Salı günü ürettiğin)
2. Yapılandırılmış besin JSON verisi (karbonhidrat_veritabani_birlesik.json)
"""

import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1) .env dosyasından OpenAI API key'i yükle (Meryem'in kurduğu güvenlik altyapısı)
load_dotenv()
assert os.getenv("OPENAI_API_KEY"), "OPENAI_API_KEY .env dosyasında bulunamadı!"

def pdf_chunklarini_hazirla(pdf_klasoru="data/"):
    """
    data/ klasöründeki PDF'leri okuyup chunk listesine çevirir.
    Ekip PDF'leri elle inceleyip zaten yapılandırılmış JSON'a çevirdiği için
    bu klasör boş olabilir -- bu durumda sorunsuzca boş liste döner.
    """
    if not os.path.isdir(pdf_klasoru):
        print(f"  '{pdf_klasoru}' klasörü bulunamadı, PDF adımı atlanıyor.\n")
        return []

    pdf_dosyalari = [d for d in os.listdir(pdf_klasoru) if d.endswith(".pdf")]
    if not pdf_dosyalari:
        print(f"  '{pdf_klasoru}' içinde PDF bulunamadı, PDF adımı atlanıyor "
              f"(veriniz zaten JSON'a çevrilmiş).\n")
        return []

    tum_sayfalar = []
    for dosya in pdf_dosyalari:
        loader = PyPDFLoader(os.path.join(pdf_klasoru, dosya))
        sayfalar = loader.load()
        tum_sayfalar.extend(sayfalar)
        print(f"  {dosya}: {len(sayfalar)} sayfa yüklendi")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    )
    chunklar = splitter.split_documents(tum_sayfalar)
    print(f"Toplam {len(chunklar)} chunk oluşturuldu (PDF kaynaklı)\n")
    return chunklar


# 3) Yapılandırılmış JSON besin verisini Document nesnelerine çevir
def json_besinlerini_documentlere_cevir(json_yolu="karbonhidrat_veritabani_birlesik.json"):
    """
    Her besin kaydını okunabilir bir cümleye çevirip Document nesnesi yapar.
    Bu şekilde embedding modeli anlamlı bir metin görür, ham JSON değil.
    """
    with open(json_yolu, "r", encoding="utf-8") as f:
        veri = json.load(f)

    documents = []
    for besin in veri["besinler"]:
        # JSON'u insan diline çeviriyoruz -- embedding kalitesi için önemli
        agirlik = besin.get("agirlik_g")
        if agirlik is not None:
            agirlik_ifadesi = f"yaklaşık {agirlik} gram olup "
        else:
            agirlik_ifadesi = ""  # agirlik_g boşsa cümleye hiç eklenmiyor

        icerik = (
            f"{besin['besin_adi']} ({besin['kategori']}): "
            f"{besin.get('degisim_olcusu', 'porsiyon belirtilmemiş')} "
            f"{agirlik_ifadesi}"
            f"{besin['karbonhidrat_g']} gram karbonhidrat içerir."
        )
        doc = Document(
            page_content=icerik,
            metadata={
                "besin_adi": besin["besin_adi"],
                "kategori": besin["kategori"],
                "karbonhidrat_g": besin["karbonhidrat_g"],
                "kaynak": besin.get("kaynak", "bilinmiyor"),
                "tip": "yapilandirilmis_besin_verisi",
            },
        )
        documents.append(doc)

    print(f"Toplam {len(documents)} besin kaydı Document'e çevrildi (JSON kaynaklı)\n")
    return documents


# 4) Hepsini ChromaDB'ye kaydet
def vektor_veritabani_olustur(pdf_chunklari, json_documentleri, kayit_yolu="./chroma_db"):
    """
    PDF chunk'ları + JSON besin verisini birleştirip Chroma'ya embedding'ler.
    persist_directory sayesinde bilgisayarı kapatsan bile veritabanı diskte kalır.
    """
    tum_documentler = pdf_chunklari + json_documentleri

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")  # ucuz ve hızlı model

    vectordb = Chroma.from_documents(
        documents=tum_documentler,
        embedding=embeddings,
        persist_directory=kayit_yolu,
        collection_name="karbonhidrat_koleksiyonu",
    )

    print(f"✅ Vektör veritabanı oluşturuldu: {kayit_yolu}")
    print(f"✅ Toplam {len(tum_documentler)} parça (chunk + besin kaydı) veritabanına eklendi")
    return vectordb


if __name__ == "__main__":
    print("1) PDF'ler okunuyor ve parçalanıyor...")
    pdf_chunklari = pdf_chunklarini_hazirla(pdf_klasoru="data/")

    print("2) JSON besin verisi hazırlanıyor...")
    json_documentleri = json_besinlerini_documentlere_cevir(
        json_yolu="karbonhidrat_veritabani_birlesik.json"
    )

    print("3) Vektör veritabanı oluşturuluyor (bu adım OpenAI'ye embedding isteği gönderir)...")
    vectordb = vektor_veritabani_olustur(pdf_chunklari, json_documentleri)

    print("\n🎉 Bitti! Veritabanın './chroma_db' klasöründe hazır.")
