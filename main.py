import streamlit as st
from apps.complete_phase1 import app as phase1_app
from apps.complete_phase2 import app as phase2_app
from apps.complete_phase3 import app as phase3_app

# Set the main page configuration
st.set_page_config(page_title="Modular Stock Analysis App", layout="wide")

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Select Phase", [
    "Phase 1 - Company Dashboard",
    "Phase 2 - Technical Analysis",
    "Phase 3 - Stock Prediction"
])

# Render the selected module's app
if selection == "Phase 1 - Company Dashboard":
    phase1_app()
elif selection == "Phase 2 - Technical Analysis":
    phase2_app()
elif selection == "Phase 3 - Stock Prediction":
    phase3_app()
