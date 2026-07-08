# 🩺 Tip 1 Diyabet Karbonhidrat Sayım Asistanı

> Tip 1 Diyabetli bireyler ve aileleri için yapay zeka destekli karbonhidrat sayım asistanı.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://accessguidancetask-hdxvh4a5xve6ndwmqv2bok.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-412991?logo=openai)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-Cloud-FF4B4B?logo=streamlit)

---

## 🎯 Proje Hakkında

Bu proje, Tip 1 Diyabetli bireyler ve ailelerinin günlük beslenme süreçlerinde karbonhidrat sayımını kolaylaştırmak için geliştirilmiş bir **RAG (Retrieval-Augmented Generation)** tabanlı yapay zeka asistanıdır.

Kullanıcılar herhangi bir yiyecek veya içeceği sorarak doğru ve kaynaklara dayalı karbonhidrat bilgisine anında ulaşabilir. Asistan, tahmin yapmak yerine yalnızca kendi veritabanındaki bilgiye dayanarak yanıt verir.

---
## 👩‍💻 Ekip

| İsim | Rol |
|------|-----|
| **Meryem** | Proje Lideri & AI Engineer — RAG zinciri, sistem prompt, OpenAI entegrasyonu |
| **Eylül** | Data Engineer — Vektör veritabanı, chunking, JSON veri hazırlama |
| **Elif** | Frontend & Ürün — Streamlit arayüzü, deployment |

---
## ✨ Özellikler

- 🔍 **Hibrit Arama:** BM25 (anahtar kelime) + Semantic (anlamsal) retrieval — yazım hatalarını ve dolaylı ifadeleri de anlıyor
- 🧠 **RAG Mimarisi:** ChromaDB vektör veritabanı + LangChain zinciri + GPT-4o-mini
- 🍎 **Geniş Besin Veritabanı:** Türk mutfağı, fast-food zincirleri (McDonald's, Burger King), marketten paketli ürünler ve daha fazlası
- 🛡️ **Halüsinasyon Koruması:** Bağlamda olmayan bilgi için asla uydurmuyor
- 💬 **Konuşma Dili Desteği:** "ekmk", "whopperr" gibi yazım hatalarını anlıyor
- ⚠️ **Tıbbi Güvenlik:** Her yanıtta tıbbi sorumluluk reddi uyarısı

---

## 🏗️ Mimari

```
Kullanıcı Sorusu
      │
      ▼
┌─────────────────┐
│  Streamlit UI   │  ← app.py
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────┐
│         RAG Zinciri             │  ← rag_zinciri.py
│                                 │
│  ┌──────────┐  ┌─────────────┐  │
│  │  BM25    │  │  Semantic   │  │
│  │Retriever │  │  Retriever  │  │
│  └────┬─────┘  └──────┬──────┘  │
│       └────────┬───────┘        │
│          EnsembleRetriever      │
│          (0.3 BM25 + 0.7 Sem.)  │
└──────────────┬──────────────────┘
               │
               ▼
    ┌──────────────────┐
    │    ChromaDB      │  ← chroma_db/
    │  (vektör DB)     │
    └──────────────────┘
               │
               ▼
    ┌──────────────────┐
    │  GPT-4o-mini     │
    │  + System Prompt │  ← system_prompt.txt
    └──────────────────┘
               │
               ▼
         Karbonhidrat Cevabı
```

---

## 📁 Dosya Yapısı

```
tip1-diyabet-asistani/
│
├── app.py                              # Streamlit arayüzü
├── rag_zinciri.py                      # LangChain RAG zinciri (hibrit retriever)
├── vektor_veritabani_olustur.py        # ChromaDB vektör veritabanı kurulum scripti
├── system_prompt.txt                   # KarbonAsistan sistem promptu
├── karbonhidrat_veritabani_birlesik.json  # Besin karbonhidrat veritabanı (JSON)
│
├── anlamsal_arama_testi.py             # Vektör DB arama test scripti
├── test_seti_calistir.py               # 80 soruluk kapsamlı test seti
├── test_sonuclari.md                   # Test çıktıları
│
├── openai_test.py                      # OpenAI bağlantı testi
├── test_api.py                         # API test scripti
├── test_cevap_al.py                    # cevap_al() fonksiyon testi
│
├── requirements.txt                    # Python bağımlılıkları
└── .env.example                        # API key şablonu (gerçek key paylaşılmaz)
```

---

## 🚀 Yerel Kurulum

### 1. Repoyu klonla
```bash
git clone https://github.com/kirmeryem28-dev/tip1-diyabet-asistani.git
cd tip1-diyabet-asistani
```

### 2. Sanal ortam oluştur ve bağımlılıkları kur
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. API key'i tanımla
`.env` dosyası oluştur:
```
OPENAI_API_KEY=sk-...
```

### 4. Vektör veritabanını oluştur
```bash
python vektor_veritabani_olustur.py
```

### 5. Uygulamayı başlat
```bash
streamlit run app.py
```

---

## 🧪 Test

80 soruluk kapsamlı test setini çalıştırmak için:
```bash
python test_seti_calistir.py
```

Test kategorileri: Temel Doğruluk, Eş Anlamlı İfadeler, Belirsiz Sorular, Halüsinasyon Koruması, Prompt Injection, Tıbbi Sınır, Birim Dönüşümü, Yazım Hatası.

---


## ⚠️ Tıbbi Sorumluluk Reddi

Bu uygulama yalnızca **bilgilendirme amaçlıdır** ve tıbbi tavsiye niteliği taşımaz. Karbonhidrat sayımı ve insülin dozajı konusunda her zaman doktorunuza veya diyetisyeninize danışınız.

Bu proje bir **öğrenci/staj projesidir** ve tıbbi bir cihaz veya uygulama değildir.

---

## 🛠️ Kullanılan Teknolojiler

- [LangChain](https://langchain.com/) — RAG zinciri ve retriever
- [OpenAI GPT-4o-mini](https://openai.com/) — Dil modeli
- [ChromaDB](https://www.trychroma.com/) — Vektör veritabanı
- [Streamlit](https://streamlit.io/) — Web arayüzü ve deployment
- [rank-bm25](https://github.com/dorianbrown/rank_bm25) — BM25 anahtar kelime araması
