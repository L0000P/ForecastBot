import torch
import pandas as pd
import numpy as np

# Transform a DataFrame into a time series dataset by padding it with zeros
def ts_padding(df: pd.DataFrame, timestamp_column: str = None, context_length: int = 1) -> pd.DataFrame:
    l = len(df) if df is not None else 0 # length of the dataframe
    if l >= context_length: return df # no padding needed
    fill_length = context_length - l # number of rows to fill

    pad_df = pd.DataFrame(np.zeros([fill_length, df.shape[1]]), columns=df.columns) # create a new DataFrame with zeros
    for c in df.columns: # copy the data types
        if c == timestamp_column: continue # skip the timestamp column
        pad_df[c] = pad_df[c].astype(df.dtypes[c], copy=False) # copy the data type

    if (df[timestamp_column].dtype.type == np.datetime64) or (df[timestamp_column].dtype == int): # if the timestamp is a datetime or an integer
        last_timestamp = df.iloc[0][timestamp_column] # get the first timestamp
        period = df.iloc[1][timestamp_column] - df.iloc[0][timestamp_column] # get the period
        prepended_timestamps = [last_timestamp + offset * period for offset in range(-fill_length, 0)] # create a list of prepended timestamps
        pad_df[timestamp_column] = prepended_timestamps # prepend the timestamps
    else: # if the timestamp is not a datetime or an integer
        pad_df[timestamp_column] = None # set the timestamp to None  
    pad_df[timestamp_column] = pad_df[timestamp_column].astype(df[timestamp_column].dtype) # copy the data type of the timestamp column
    
    new_df = pd.concat([pad_df, df]) # concatenate the new DataFrame with the original DataFrame
    return new_df # return the new DataFrame

def np_to_torch(data: np.array, float_type=np.float32): # convert a numpy array to a PyTorch tensor
    if data.dtype == "float": # if the data type is float
        return torch.from_numpy(data.astype(float_type)) # convert the data to a PyTorch tensor
    elif data.dtype == "int": # if the data type is int
        return torch.from_numpy(data) # convert the data to a PyTorch tensor
    return torch.from_numpy(data) # convert the data to a PyTorch tensor

class BaseDFDataset(torch.utils.data.Dataset): # base class for time series datasets
    def __init__( # initialize the class
        self, # reference to the class instance
        data_df           : pd.DataFrame, # DataFrame containing the data
        timestamp_column  : str, # name of the column containing the timestamps
        x_cols            : list = [], # list of column names for the input features
        y_cols            : list = [], # list of column names for the target features
        context_length    : int  = 1, # number of time steps to consider as context
        prediction_length : int  = 0, # number of time steps to predict
        zero_padding      : bool = True # whether to pad the data with zeros 
    ):
        super().__init__() # initialize the base class
        if not isinstance(x_cols, list): x_cols = [x_cols] # if x_cols is not a list, convert it to a list
        if not isinstance(y_cols, list): y_cols = [y_cols] # if y_cols is not a list, convert it to a list

        self.data_df           = data_df # DataFrame containing the data
        self.datetime_col      = timestamp_column # name of the column containing the timestamps
        self.x_cols            = x_cols # list of column names for the input features
        self.y_cols            = y_cols # list of column names for the target features
        self.context_length    = context_length # number of time steps to consider as context
        self.prediction_length = prediction_length # number of time steps to predict
        self.zero_padding      = zero_padding # whether to pad the data with zeros
        
        data_df[timestamp_column] = pd.to_datetime(data_df[timestamp_column]) # convert the timestamp column to datetime
        self.data_df = data_df.sort_values(timestamp_column, ignore_index=True) # sort the DataFrame by the timestamp column

        if zero_padding: # if zero padding is enabled
            self.data_df = self.pad_zero(data_df) # pad the data with zeros

        self.timestamps  = self.data_df[timestamp_column].values # get the timestamps

    # pad the data with zeros
    def pad_zero(self, data_df):
        return ts_padding( # pad the data with zeros
            data_df, # DataFrame containing the data
            timestamp_column= self.datetime_col, # name of the column containing the timestamps
            context_length  = self.context_length + self.prediction_length # number of time steps to consider as context 
        )

    # get the length of the dataset
    def __len__(self):
        return len(self.data_df) - self.context_length - self.prediction_length + 1 # return the length of the dataset

    # get an item from the dataset
    def __getitem__(self, index: int):
        raise NotImplementedError # raise an error if the method is not implemented