#!/bin/sh

if [ ! -d "/server/src/transformer/Arima/results" ]; then
    python /server/src/transformer/Arima/test.py
fi

if [ ! -d "/server/src/transformer/Sarimax/results" ]; then
    python /server/src/transformer/Sarimax/test.py
fi

if [ ! -d "/server/src/transformer/PatchTST/results" ]; then
    python /server/src/transformer/PatchTST/test.py
fi

uvicorn src.main:app --host 0.0.0.0 --port 8000