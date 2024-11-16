# ForecastBot
![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)

**ForecastBot** is a web application designed to provide users with an intelligent chatbot interface that helps 
them select and interact with forecasting models, powered by a Large Language Model (LLM). The chatbot guides users 
through the process of selecting a model based on their dataset and specific parameters, using **Streamlit** for the 
user interface and **FastAPI** for the backend.

The app integrates both advanced transformer-based forecasting models (such as Informer, PatchTST, Pyraformer) 
and traditional models (ARIMA, SARIMAX), offering a seamless experience for model selection, dataset configuration, 
and result visualization.

## Architecture

Below is the architectural pattern of **ForecastBot**, showcasing the flow between its main components: the client, the server, and the forecasting models. 

![ForecastBot Architecture](/img/Architecture.png)



## Features
- **Chatbot Interface**: An interactive chatbot built with LLM capabilities, allowing users to interact naturally 
  and get model recommendations based on their input.
- **Model Support**: Includes the following forecasting models:
  - Informer
  - PatchTST
  - Pyraformer
  - ARIMA
  - SARIMAX
- **Visualization**: Interactive graphs and charts to display model performance.

## Default Dataset
By default, the models use the **ETTh1** dataset, which has been tested by training and evaluating models using both **GPU** and **CPU** environments. The **GPU** testing was conducted using a **container** built with **RapidsAI**, leveraging the power of **CUDA** for accelerated computation. Similarly, the models were also tested on a **CPU**-only environment for comparison.

You can download the dataset from the following link:
[ETTh1.csv](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv).

Ensure the dataset is placed in the following directory structure:
```bash
ForecastBot/
└── server/
  └── data/
    └── ETTh1.csv
```

### Local environment

How to start (both on Linux and Windows (PoweShell/WSL)):
```console
$: ./clip dev up --build
```
or any parameter of `docker compose` you prefer after `dev`.

Pretrain models using dev (inside server container)
```console
$: python /server/src/transformer/Arima/test.py     # Pretrain Arima Model 
$: python /server/src/transformer/Sarimax/test.py   # Pretrain Sarimax Model  
$: python /server/src/transformer/PatchTST/test.py  # Pretrain PatchTST Model
```


Prod configuration
```console
$: ./clip prod up -d --build