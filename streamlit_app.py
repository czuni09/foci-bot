import streamlit as st
import ultimate_football_bot as bot

st.set_page_config(page_title="Protipp Debug", page_icon="🛠️")
st.title("🏆 Duplázó Bot - Hibakezelő Üzemmód")

if st.button("Kérem az elemzést"):
    with st.spinner('Adatok lekérése és ellenőrzése...'):
        siker, uzenet = bot.ultimate_football_bot()
        
        if siker:
            st.success(f"✅ {uzenet}")
        else:
            st.error(f"❌ {uzenet}")
            st.info("Tipp: Ellenőrizd a Streamlit Secrets beállításokat!")
