# test.py
import sys
import os
import pandas as pd
from pathlib import Path

# Add the src/transformer path to sys.path so PatchTST can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "/server/src")))

# Now you can import PatchTST
from transformer.PatchTST import PatchTST

# Define paths
csv_file = "/server/data/ETTh1.csv"  # Path to the original CSV file

# Initialize the PatchTST model
patch_tst = PatchTST()

# Load and prepare data
patch_tst.load_data([csv_file])

# Configure the model
patch_tst.configure_model()

# Train the model
#patch_tst.train(epochs=20, learning_rate=5e-6)

# Load the model (skip if already trained)
patch_tst.load_model()

# Load the original data and select a portion for prediction
original_data = pd.read_csv(csv_file)
new_data = original_data.tail(100).reset_index(drop=True)  # Using the last 100 rows for prediction

# Perform prediction on the extracted data portion
predictions = patch_tst.predict(new_data)

# Print the predictions to check output
print(predictions)