#!/bin/bash
echo "Installing requirements..."
pip install -r requirements.txt
echo "Starting NeuroPulse AI..."
streamlit run app.py