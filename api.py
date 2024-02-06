from fastapi import FastAPI,Query
import pandas as pd
import requests
from typing import Dict,Any
from typing import List

import logging

app = FastAPI()

# app.state.model = load_model()

@app.get('/predict')

def predict(
    all_index: List[Any] = Query(..., title="List of indices for prediction", description="Provide a list of indices for prediction.")
):


    X_pred = pd.DataFrame({'all_index':all_index})



    return {'selected_country': X_pred.to_dict(orient='records')}


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END
