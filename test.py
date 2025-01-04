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
