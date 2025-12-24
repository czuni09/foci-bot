import streamlit as st
import ultimate_football_bot as bot

st.set_page_config(page_title="Pro Odds Analyzer", page_icon="🎯", layout="wide")

# Vizuális elem a professzionális megjelenéshez
st.image("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?q=80&w=1000", use_container_width=True)

st.title("🏆 Intelligens Fogadási Elemző")
st.markdown("Ez a rendszer valós piaci adatokat (Odds-API) használ a duplázó esélyek kereséséhez.")

# Beállítások a felületen
col1, col2 = st.columns(2)
with col1:
    target = st.slider("Cél szorzó (Odds)", 1.5, 5.0, 2.0, 0.1)
with col2:
    email_kuldés = st.checkbox("Email riport küldése is", value=True)

if st.button("🚀 Elemzés Indítása"):
    with st.spinner('Adatok lekérése a Londoni és Madridi központokból...'):
        # Futtatás: email küldéssel vagy csak kijelzéssel
        success, message = bot.run(send_email=email_kuldés, target_odds=target)
        
        if success:
            st.success("Az elemzés sikeresen lefutott!")
            if not email_kuldés:
                st.text_area("Mai tippek:", value=message, height=400)
            else:
                st.info(message)
                st.balloons()
        else:
            st.error(message)
            st.warning("Tipp: Ellenőrizd a Streamlit Secrets beállításait (API kulcsok, Email jelszó)!")
