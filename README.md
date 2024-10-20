# ForecastBot
**ForecastBot** is a web application designed to provide users with an intelligent chatbot interface that helps 
them select and interact with forecasting models, powered by a Large Language Model (LLM). The chatbot guides users 
through the process of selecting a model based on their dataset and specific parameters, using **Streamlit** for the 
user interface and **FastAPI** for the backend.

The app integrates both advanced transformer-based forecasting models (such as Informer, PatchTST, Pyraformer) 
and traditional models (ARIMA, SARIMAX), offering a seamless experience for model selection, dataset configuration, 
and result visualization.

## **Features**
- **Chatbot Interface**: An interactive chatbot built with LLM capabilities, allowing users to interact naturally 
  and get model recommendations based on their input.
- **Model Support**: Includes the following forecasting models:
  - Informer
  - PatchTST
  - Pyraformer
  - ARIMA
  - SARIMAX
- **Visualization**: Interactive graphs and charts to display model performance.

## **Dataset Tested**
By default, the models use the **ETTh1** dataset, which has been tested by training and evaluating models using both **GPU** and **CPU** environments. The **GPU** testing was conducted using a **container** built with **RapidsAI**, leveraging the power of **CUDA** for accelerated computation. Similarly, the models were also tested on a **CPU**-only environment for comparison.

You can download the dataset from the following link:
[ETTh1.csv](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv).

Ensure the dataset is placed in the following directory structure:
```bash
ForecastHub/
└── data/
    └── ETTh1.csv
```

## **Getting Started**

### **Prerequisites**
Ensure that you have Docker and Docker Compose installed on your system. If not, you can install them using the following links:
- [Install Docker](https://docs.docker.com/get-docker/)
  
### **Installation and Setup**
1. **Clone the Repository**
   Clone this repository to your local machine.
   ```bash
    git clone https://github.com/L0000P/ForecastBot.git
    cd ForecastBot
    ```
2. **Download Dataset**
    Download the ETTh1 dataset from [here](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv).
    and place it in the data folder as shown:
    ```bash
    mkdir data
    mv /path/to/downloaded/ETTh1.csv data/
    ```
3. **Build Image and Run Docker Container**
    ```bash
    docker compose build
    docker compose up 
    ```
4. **Access Web Interface**  After the containers are up and running, 
   you can access the following:
- Streamlit (UI): Open a browser and navigate to http://localhost:8501.
- FastAPI (API): Open http://localhost:8000/docs for the FastAPI documentation.

## **How to Enter the Container**
To test or make modifications inside the container, you can enter the running container with the following command:

- Transformers Container 
```bash
docker exec -it transformers /bin/bash
```

- Web-App Container 
```bash
docker exec -it web_app /bin/bash
```

## **Stopping the Containers**
To stop the stack of running containers, use the following command:
```bash
docker compose down
``` 