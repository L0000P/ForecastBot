#!/bin/sh
RUN python /server/src/transformer/Arima/test.py
RUN python /server/src/transformer/Sarimax/test.py
RUN python /server/src/transformer/PatchTST/test.py
uvicorn src.main:app --host 0.0.0.0 --port 8000