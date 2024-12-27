import pandas as pd
from langchain_core.tools import tool
from transformer import Arima, Sarimax, PatchTST

class Tools:
    def __init__(self):
        pass

    @staticmethod
    @tool
    def ArimaTool(data: list) -> dict:
        """Run the ARIMA model on the given data."""
        try:
            df = pd.DataFrame(data, columns=['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            model = Arima() 
            predictions = model.predict_model(df)  
            return {"result": predictions}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    @tool
    def SarimaxTool(data: list) -> dict:
        """Run the SARIMAX model on the given data."""
        try:
            # Create DataFrame
            df = pd.DataFrame(data, columns=['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'])
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            # Instantiate and predict with SARIMAX
            model = Sarimax()
            target_column = 'HUFL' 
            input_series = model.df[target_column]
            
            # Perform predictions
            forecast_values, forecast_index = model.predict_model(input_series)
            result = {"Date": forecast_index, "Predictions": forecast_values}
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    @staticmethod
    @tool
    def PatchTSTTool(data: list) -> dict:
        """Run the PatchTSTTool model on the given data."""
        try:
            df = pd.DataFrame(data, columns=['date', 'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'])
            df = df.reset_index(drop=True)
            model = PatchTST()
            model.load_data()
            model.configure_model()
            predictions = model.predict(df)
            return {"result": predictions}
        except Exception as e:
            return {"error": str(e)}

    def get_all_tools(self):
        """Return all available tools."""
        return [self.ArimaTool, self.SarimaxTool, self.PatchTSTTool]
