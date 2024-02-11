from fastapi import FastAPI, HTTPException
import pandas as pd
from typing import Any, List
from user_input import process_user_input
from main import X_scaled, neigh, df
app = FastAPI()

#@app.get('/predict', response_model=None)
@app.post('/predict', response_model=None)
def predict(all_index: List[Any] = None):
    try:
        if not all([X_scaled is not None , neigh is not None, df is not None]):
            raise HTTPException(status_code=500, detail="Missing required parameters: X_scaled, neigh, df")

        if all_index is None:
            raise HTTPException(status_code=400, detail="List of indices for prediction is missing")

        processed_input = process_user_input(all_index, X_scaled, neigh, df)
        processed_input_dict = processed_input.to_dict(orient='records')
        return processed_input_dict

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def root():
    return {"greeting": "Hello"}
