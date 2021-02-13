import streamlit as st 
from multiapp import Multiapp  
from apps import home, Clf, Reg, Eda

app = Multiapp()

st.markdown("The Lazy Data Science App")

# Adding all the application here
app.add_app("Home", home.app)
app.add_app("Eda", Eda.app)
app.add_app("Clf", Clf.app)
app.add_app("Reg", Reg.app)
# The main app
app.run()