import streamlit as st
import ultimate_football_bot as bot

st.set_page_config(page_title="Pro Foci Bot", page_icon="⚽")
st.title("🏆 czunidaniel9 Profi Tippadó")

st.info("A bot elemzi a bírót, az időjárást és a csapatok formáját.")

if st.button("Kérem a mai biztos tippeket"):
    with st.spinner('Elemzés futtatása...'):
        siker = bot.ultimate_football_bot()
        if siker:
            st.success("✅ A pontos tippek (lapok, szögletek, nyertes) elküldve az e-mailedre!")
        else:
            st.error("❌ Hiba történt. Ellenőrizd az API kulcsokat!")
