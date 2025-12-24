import streamlit as st
import ultimate_football_bot as bot

st.title("⚽ czunidaniel9 Foci Bot")
st.write("A rendszer minden nap 10:00-kor küld e-mailt.")

if st.button("Teszt elemzés indítása most"):
    st.write("Elemzés és e-mail küldés folyamatban...")
    h = bot.TeamStats("Arsenal")
    v = bot.TeamStats("Crystal Palace")
    
    # Itt ellenőrizzük, hogy a küldés sikeres volt-e
    siker = bot.ultimate_football_bot(h, v, "London", "PL", 1.5, 80)
    
    if siker:
        st.success("✅ Kész! Az e-mail sikeresen elment!")
    else:
        st.error("❌ Hiba! Az e-mail küldése nem sikerült. Ellenőrizd a jelszót (Secret)!")
