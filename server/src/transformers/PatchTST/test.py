import pandas as pd
import numpy as np
from PatchTST import PatchTST
from pathlib import Path

# Define paths
log_path = "Data/Log"
model_path = "Data/Model"
csv_file = "/server/data/ETTh1.csv"  # Path to the original CSV file

# Initialize the PatchTST model
patch_tst = PatchTST(
    log_path=log_path,
    model_path=model_path,
    context_length=512,
    forecast_horizon=96,
    patch_length=32,
    num_workers=8,
    batch_size=8
)

# Load and prepare data
patch_tst.load_data([csv_file])

# Configure the model
patch_tst.configure_model()

# Train the model
patch_tst.train(epochs=20, learning_rate=5e-6)

# Load the model (skip if already trained)
patch_tst.load_model()

# Load the original data and select a portion for prediction
original_data = pd.read_csv(csv_file)
new_data = original_data.tail(100).reset_index(drop=True)  # Using the last 100 rows for prediction

# Perform prediction on the extracted data portion
predictions = patch_tst.predict(new_data)