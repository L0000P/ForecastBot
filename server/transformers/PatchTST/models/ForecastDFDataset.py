import pandas as pd
import copy
from typing import List
from models.BaseDFDataset import BaseDFDataset, np_to_torch

# ForecastDFDataset class
class ForecastDFDataset(BaseDFDataset):
    def __init__(
        self, # reference to the class instance
        data_df          : pd.DataFrame, # DataFrame containing the data 
        timestamp_column : str, # name of the column containing the timestamps
        context_length   : int        = 1, # number of time steps to consider as context
        prediction_length: int        = 1, # number of time steps to predict
        target_columns   : List[str]  = [] # list of column names for the target features            
    ):
        self.target_columns = list(target_columns) # list of column names for the target features
        x_cols = self.target_columns # list of column names for the input features
        y_cols = copy.copy(x_cols) # list of column names for the target features

        super().__init__( # initialize the base class
            data_df           = data_df, # DataFrame containing the data
            timestamp_column  = timestamp_column, # name of the column containing the timestamps
            x_cols            = x_cols, # list of column names for the input features
            y_cols            = y_cols, # list of column names for the target features  
            context_length    = context_length, # number of time steps to consider as context
            prediction_length = prediction_length # number of time steps to predict
        )

    def __getitem__(self, time_id): # get an item from the dataset
        s_pos = time_id # start position
        e_pos = time_id + self.context_length # end position
        seq_x = self.data_df.loc[s_pos : e_pos-1, self.x_cols].values # seq_x: batch_size x context_len x num_x_cols
            
        seq_y = self.data_df.loc[ # seq_y: batch_size x pred_len x num_y_cols
            e_pos : e_pos + self.prediction_length -1, # start:end
            self.y_cols # columns
        ].values

        ret = {
            "past_values"  : np_to_torch(seq_x), # past_values: batch_size x context_len x num_x_cols
            "future_values": np_to_torch(seq_y), # future_values: batch_size x pred_len x num_y_cols
        }

        return ret # return the item

    # get the length of the dataset
    def __len__(self):
        return len(self.data_df) - self.context_length - self.prediction_length + 1 # return the length of the dataset