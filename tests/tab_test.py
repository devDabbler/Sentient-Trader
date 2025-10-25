import streamlit as st

tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
with tab1:
    st.write("This is Tab 1")
with tab2:
    st.write("This is Tab 2")