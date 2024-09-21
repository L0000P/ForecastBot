import pandas as pd
import glob
import warnings
from pathlib import Path
from tsfm_public.toolkit.dataset import ForecastDFDataset
from models.ForecastDFDataset import ForecastDFDataset
from transformers import (
    EarlyStoppingCallback,
    PatchTSTConfig,
    PatchTSTForPrediction,
    set_seed,
    Trainer,
    TrainingArguments
)

warnings.filterwarnings("ignore", module="torch") # Ignore Torch Warnings.
set_seed(2023) # Set Seed for Reproducibility.

log_path   = "Data/Log/" # Path to Log.
model_path = "Data/Model/" # Path to Model.

Path(log_path).mkdir(parents=True, exist_ok=True) # Create Log Path.
Path(model_path).mkdir(parents=True, exist_ok=True) # Create Model Path.

context_length   = 1024 # Context Length.
forecast_horizon = 192 # Forecast Horizon.
patch_length     = 64 # Patch Length.
num_workers      = 8 # Number of Workers.  
batch_size       = 4 # Batch Size.

timestamp_column = input("Insert timestamp column name: ")
csv_files = glob.glob(input("Insert Path of csv files: "))
delimiter = input("Insert delimiter of dataset: ")
index_start = input("Insert start index dataset: ")
column_search_start = input("Insert start string column search: ")
column_search_end = input("Insert end string column search: ")
resume_from_checkpoint = input("Resume from checkpoint?(<y>,n): ")

if not timestamp_column:
    timestamp_column = "time_stamp" 
if not csv_files:
    csv_files = glob.glob("../data/SCADA_Turbine/WindFarmA/datasets/0.csv") 
if not delimiter:
    delimiter = ";"
if not index_start:
    index_start = 5
if not column_search_start:
    column_search_start = "sensor"
if not column_search_end:
    column_search_end = "avg"
if resume_from_checkpoint == "" or resume_from_checkpoint == "y" or resume_from_checkpoint == "yes":
    resume_from_checkpoint = True
if not(resume_from_checkpoint == "" or resume_from_checkpoint == "y" or resume_from_checkpoint == "yes"):
    resume_from_checkpoint = False

dataframes = [pd.read_csv(f, parse_dates=[timestamp_column], delimiter=delimiter) for f in csv_files] # Read CSV Files.
dataset = pd.concat(dataframes, ignore_index=True) # Concatenate DataFrames.
past_columns = [col for col in dataset.columns[index_start:] if col.startswith(column_search_start) and col.endswith(column_search_end)] # Forecast Columns.
forecast_columns = [col for col in dataset.columns[index_start:] if col.startswith(column_search_start)and col.endswith(column_search_end)] # Forecast Columns.

# Splitting
num_train = int(len(dataset) * 0.7) # 70% Train.
num_test = int(len(dataset) * 0.2) # 20% Train.
num_valid = len(dataset) - num_train - num_test # 10% Train.

# TEST AND VALIDATION are shifted by context_length so the first prediction immediately follows the training and validation.
f1, e1     = 0, num_train # Train 
f2, e2     = e1-context_length, e1+num_valid # Valid
f3, e3     = e2-context_length, len(dataset) # Test

train_dataset = dataset.iloc[f1:e1, :].reset_index(drop=True) # Train Dataset.
valid_dataset = dataset.iloc[f2:e2, :].reset_index(drop=True) # Valid Dataset.
test_dataset = dataset.iloc[f3:e3, :].reset_index(drop=True) # Test Dataset.

# Train Dataset
train_dataset = ForecastDFDataset(
    data_df=train_dataset,
    timestamp_column=timestamp_column,
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon
)

# Valid Dataset
valid_dataset = ForecastDFDataset(
    data_df=valid_dataset,
    timestamp_column=timestamp_column,
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon
)

# Test Dataset
test_dataset = ForecastDFDataset(
    data_df=test_dataset,
    timestamp_column=timestamp_column,
    target_columns=forecast_columns,
    context_length=context_length,
    prediction_length=forecast_horizon
)

# Config PatchTST
config = PatchTSTConfig(
        num_input_channels=len(forecast_columns),
        context_length=context_length,
        patch_length=patch_length,
        patch_stride=patch_length,
        prediction_length=forecast_horizon,
        random_mask_ratio=0.4,
        d_model=128,
        num_attention_heads=16,
        num_hidden_layers=3,
        ffn_dim=256,
        dropout=0.2,
        head_dropout=0.2,
        pooling_type=None,
        channel_attention=False,
        scaling="std",
        loss="mse",
        pre_norm=True,
        norm_type="batchnorm",
    )
# Model
model = PatchTSTForPrediction(config)

# Training Arguments
training_args  = TrainingArguments(
    output_dir                  = model_path,
    overwrite_output_dir        = True,
    num_train_epochs            = 100,
    do_eval                     = True,
    evaluation_strategy         = "epoch",
    per_device_train_batch_size = batch_size,
    per_device_eval_batch_size  = batch_size,
    dataloader_num_workers      = num_workers,
    save_strategy               = "epoch",
    logging_strategy            = "epoch",
    save_total_limit            = 3,
    logging_dir                 = log_path,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    label_names                 = ["future_values"],
)

# Create the early stopping callback
early_stopping_callback      = EarlyStoppingCallback(
    early_stopping_patience  = 10,      # Number of epochs with no improvement after which to stop
    early_stopping_threshold = 0.0001,  # Minimum improvement required to consider as improvement
)

# Define trainer
trainer = Trainer(
    model         = model,
    args          = training_args,
    train_dataset = train_dataset,
    eval_dataset  = valid_dataset,
    callbacks     = [early_stopping_callback]
)

# Pretrain
trainer.train()


