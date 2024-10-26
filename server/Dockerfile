FROM rapidsai/rapidsai:cuda11.8-runtime-ubuntu22.04-py3.10

WORKDIR /server

COPY . .

RUN chmod +x entrypoint-prod.sh

# Install dependencies
RUN pip uninstall jupyterlab -y 
RUN pip install -r requirements.txt
RUN pip install pandas==1.5.3

EXPOSE 8000

ENTRYPOINT ["sh", "entrypoint-prod.sh"]