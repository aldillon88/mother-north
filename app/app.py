import streamlit as st
#import pandas as pd
#import plotly.graph_objects as go
#import sys
#import os

# Add the project root to the system path
#project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

#if project_root not in sys.path:
#	sys.path.append(project_root)

st.set_page_config(page_title="Mother North", layout="wide")

st.title('Mother North 2024 Analysis')

analysis_page = st.Page("page_1.py", title="Analysis")
project_page = st.Page("page_2.py", title="Project Description")

pg = st.navigation([analysis_page, project_page])

pg.run()