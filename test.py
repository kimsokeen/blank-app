import streamlit as st
import pandas as pd 
import numpy as np 

# Initialize page state if not set
if "page" not in st.session_state:
    st.session_state.page = 1  # Start from page 1

# Function to navigate to the specified page with single click
def navigate(page):
    st.session_state.page = page
