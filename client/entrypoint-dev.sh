#!/bin/sh
chmod +x entrypoint.sh
pip install -r requirements.txt
streamlit run main.py