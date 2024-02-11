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
    ): # every parameter reveived will be of string type  ****Convet to the required data type*****



    #X_pred = pd.DataFrame({'all_index': all_index}, index=[0])

    #model = app.state.model
    #assert model is not None


    #y_pred = model.predict(X_pred)

    #return {'selected_country': all_index} # PD Dataframe



    return {'selected_country': all_index}


@app.get("/")
def root():
    # $CHA_BEGIN
    return dict(greeting="Hello")
    # $CHA_END
