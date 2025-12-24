import streamlit as st
import ultimate_football_bot as bot

st.set_page_config(page_title="2.00 Odds Bot", page_icon="💰")
st.title("🏆 czunidaniel9 Duplázó Rendszer")

if st.button("Kérem a mai 2.00-ás szelvényt"):
    with st.spinner('Adatok és pletykák elemzése...'):
        siker = bot.ultimate_football_bot()
        if siker:
            st.success("✅ A szelvény (1000 -> 2000 Ft) elküldve az e-mailedre!")
        else:
            st.error("❌ Hiba! Ellenőrizd a Secrets beállításokat!")
