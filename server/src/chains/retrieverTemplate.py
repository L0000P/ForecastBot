# retriever_template.py
from langchain.chains import RetrievalChain
from utils.csv_retriever import SimpleCSVRetriever

def getRetrieverChain():
    retriever = SimpleCSVRetriever(file_path='/transformers/data/ETTh1.csv')
    chain = RetrievalChain(retriever=retriever)
    return chain