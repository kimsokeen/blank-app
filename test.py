import streamlit as st
import pandas as pd 
import numpy as np 

# Initialize page state if not set
if "page" not in st.session_state:
    st.session_state.page = 1  # Start from page 1

# Function to navigate to the specified page with single click
def navigate(page):
    st.session_state.page = page

def back_button(destination_page):
    st.markdown(
        """
        <style>
            .stButtonSmall button {{
                width: 60px;
                height: 35px;
                font-size: 14px;
                background-color: #fff;
                color: #333;
                border: 1px solid #333;
                border-radius: 5px;
                font-weight: bold;
            }}
            .stButtonSmall button:hover {{
                background-color: #e6e6e6;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )
    if st.button("← Back", key=f"back_{destination_page}"):
        navigate(destination_page)

def page1():
    st.title("Ulcers Detector")
    if st.button("Get Started"):
        navigate(2)

def page2():
    st.title("page2")
    if st.button("Get Started"):
        navigate(3)

def page3():
    st.title("page3")
    if st.button("Get Started"):
        navigate(4)

def page4():
    st.title("page4")
    if st.button("Get Started"):
        navigate(5)

def page5():
    st.title("page5")
    if st.button("Get Started"):
        navigate(6)

def main():
    if st.session_state.page == 1:
        page1()
    elif st.session_state.page == 2:
        page2()
    elif st.session_state.page == 3:
        page3()
    elif st.session_state.page == 4:
        page4()
    elif st.session_state.page == 5:
        page5()

if __name__ == "__main__":
    main()
