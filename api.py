from fastapi import FastAPI, HTTPException
import pandas as pd
import json
from typing import Any, List
from user_input import process_user_input
from main import X_scaled, neigh, df, vectorized_descriptions, top_token, lda_wine_cluster_prob
from token_processing import generate_tokens_list
from final_score_ranking import process_selected_tokens

app = FastAPI()

@app.post('/predict', response_model=None)
def predict(all_index: List[Any] = None):
    type_selected = df.iloc[all_index[0]]['type']
    print(all_index)
    try:
        if not all([X_scaled is not None , neigh[type_selected] is not None, df is not None]):
            raise HTTPException(status_code=500, detail="Missing required parameters: X_scaled, neigh, df")

        if all_index is None:
            raise HTTPException(status_code=400, detail="List of indices for prediction is missing")

        pred_df = process_user_input(all_index, X_scaled, neigh, df, type_selected)
        tokens_list= generate_tokens_list(vectorized_descriptions, type_selected,pred_df)
        # processed_input_dict = processed_input.to_dict(orient='records')
        response = {'predictions': pred_df.to_dict(), 'tokens':tokens_list}
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/predict2', response_model=None)
def predict2(request2: dict = None):
    # type_selected = df.iloc[all_index[0]]['type']
    type_selected = request2['type_selected']
    selected_tokens = request2['tokens']
    print(selected_tokens, type_selected)
    pred_df = request2['predictions']
    pred_df = pd.DataFrame(pred_df)
    print(pred_df)
    try:
        if not all([X_scaled is not None , neigh[type_selected] is not None, df is not None]):
            raise HTTPException(status_code=500, detail="Missing required parameters: X_scaled, neigh, df")

        if pred_df is None:
            raise HTTPException(status_code=400, detail="List of indices for prediction is missing")

        lda_selected = process_selected_tokens(type_selected,selected_tokens, top_token, lda_wine_cluster_prob,pred_df)
        print(lda_selected)
        return {'recommendations':lda_selected}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"greeting": "Hello"}
