import streamlit as st
import ultimate_football_bot as bot

st.set_page_config(page_title="Odds-Master Pro", page_icon="📈")

st.image("https://images.unsplash.com/photo-1518152006812-edab29b069ac?q=80&w=1000", 
         caption="Élő Odds Elemzés és Valószínűség-számítás", use_container_width=True)

st.title("🏆 czunidaniel9 Smart Bet")
st.write("Ez a bot már valós piaci oddsokat elemez az Odds-API segítségével.")

if st.button("🔥 MAI ODDS-VADÁSZAT INDÍTÁSA"):
    with st.spinner('Keresem a legjobb 2.00-ás szorzókat a bukiknál...'):
        siker, uzenet = bot.run()
        if siker:
            st.success(uzenet)
            st.balloons()
        else:
            st.error(uzenet)
