FROM python:3.10-slim

WORKDIR /client

COPY . .

RUN chmod +x entrypoint-prod.sh
RUN pip install -r requirements.txt

CMD ["sh", "entrypoint-prod.sh"]
