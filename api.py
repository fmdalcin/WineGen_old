from fastapi import FastAPI, Query
import pandas as pd
from typing import Any, List
from user_input import process_user_input

from main import X_scaled
from main import neigh
from main import df

app = FastAPI()

@app.get('/predict', response_model=None)  # Define response model as Any vvvv  = Query(..., title="List of indices for prediction", description="Provide a list of indices for prediction.")
def predict():
    all_index: List[any]

    global X_scaled
    global neigh
    global df


    # if not all(X_scaled is not None, neigh is not None, df is not None):
    #     return {"error": "Missing required parameters: X_scaled, neigh, df"}

    # Call function from the package passing all_index as input
    processed_input = process_user_input(all_index, X_scaled, neigh, df)

    # Convert DataFrame to dictionary
    processed_input_dict = processed_input.to_dict(orient='records')

    return processed_input_dict  # Returning processed input data as dictionary


@app.get("/")
def root():
    return {"greeting": "Hello"}
