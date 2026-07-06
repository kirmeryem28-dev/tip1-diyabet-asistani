#karb-chat-bot

import streamlit as st

# --- YAN MENÜ (SIDEBAR) ---

with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 10px;">
            <svg width="70" height="90" viewBox="0 0 60 100">
                <line x1="30" y1="20" x2="30" y2="95" stroke="#E8A33D" stroke-width="3"/>
                <ellipse cx="20" cy="30" rx="8" ry="14" fill="#E8A33D" transform="rotate(-30 20 30)"/>
                <ellipse cx="40" cy="30" rx="8" ry="14" fill="#E8A33D" transform="rotate(30 40 30)"/>
                <ellipse cx="18" cy="48" rx="8" ry="14" fill="#E8A33D" transform="rotate(-30 18 48)"/>
                <ellipse cx="42" cy="48" rx="8" ry="14" fill="#E8A33D" transform="rotate(30 42 48)"/>
                <ellipse cx="20" cy="66" rx="8" ry="14" fill="#E8A33D" transform="rotate(-30 20 66)"/>
                <ellipse cx="40" cy="66" rx="8" ry="14" fill="#E8A33D" transform="rotate(30 40 66)"/>
                <ellipse cx="30" cy="10" rx="7" ry="12" fill="#E8A33D"/>
            </svg>
        </div>
    """, unsafe_allow_html=True)
    st.header("👤 Kullanıcı Bilgileri")
    kullanici_adi = st.text_input("Adınız", placeholder="Adınızı girin...")
    
    if kullanici_adi:
        st.success(f"Hoş geldin, {kullanici_adi}! 👋")

# --- ANA EKRAN ---
st.title("Karbonhidrat Sayım Asistanı")

# --- HAFIZA: konuşma geçmişini ilk kez oluştur ---
if "mesajlar" not in st.session_state:
    st.session_state.mesajlar = [
        {"rol": "assistant", "icerik": "Merhaba! Ben senin Karbonhidrat Sayım ve Beslenme Asistanınım 🥗 Bugün sana nasıl yardımcı olabilirim?"}
    ]

# --- GEÇMİŞTEKİ TÜM MESAJLARI EKRANA BAS ---
for mesaj in st.session_state.mesajlar:
    with st.chat_message(mesaj["rol"]):
        st.write(mesaj["icerik"])

# --- YENİ MESAJ GELİRSE ---
user_input = st.chat_input("Mesajınızı yazın...")

if user_input:
    # kullanıcı mesajını hafızaya ekle
    st.session_state.mesajlar.append({"rol": "user", "icerik": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    from rag_zinciri import cevap_al
    sabit_cevap = cevap_al(user_input)

    # Alt kısımdaki girintiler zaten doğru, yukarıyı içeri alınca burası da bloğa bağlanacak
    st.session_state.mesajlar.append({"rol": "assistant", "icerik": sabit_cevap})
    with st.chat_message("assistant"):
        st.write(sabit_cevap)