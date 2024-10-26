#!/bin/sh

pip uninstall jupyterlab -y
pip install -r requirements.txt
pip install pandas==1.5.3

uvicorn src.main:app --host 0.0.0.0 --port 8000