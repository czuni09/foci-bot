import streamlit as st
import ultimate_football_bot as bot

st.set_page_config(page_title="Data Football Pro", page_icon="📈")

# KÉP MEGJELENÍTÉSE
st.image("https://images.unsplash.com/photo-1574629810360-7efbbe195018?q=80&w=1000", 
         caption="Adatvezérelt Labdarúgó Analitika", use_container_width=True)

st.title("⚽ czunidaniel9 Pro Elemző")

st.info("Ez a rendszer OOP alapú hibakezelést és liga-súlyozott pontozást használ.")

if st.button("🚀 Stratégiai Elemzés Futtatása"):
    with st.spinner('Adatok lekérése az API-ból és pontozás...'):
        siker, uzenet = bot.run_analysis_and_send()
        if siker:
            st.success(uzenet)
            st.balloons()
        else:
            st.error(f"Hiba: {uzenet}")
