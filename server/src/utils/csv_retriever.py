import pandas as pd

class SimpleCSVRetriever:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    def query(self, query_str):
        return self.data
