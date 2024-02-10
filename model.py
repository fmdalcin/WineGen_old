import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

wines = ['red', 'white', 'rose', 'spark']

def train(X_scaled:pd.DataFrame)-> NearestNeighbors:
    '''Receives the X scaled Dataframe and trains the classification model'''
    neigh={}
    for type in wines:
        neigh[type] = NearestNeighbors(n_neighbors=10000, algorithm='ball_tree', n_jobs=-1)
        neigh[type].fit(X_scaled[X_scaled[f'type_{type}']>0])
        X_scaled[X_scaled[f'type_{type}']>0].to_csv(f"./raw_data/X_scaled_{type}.csv", index=True)

    return neigh

def predict(neigh:NearestNeighbors, type_selected:str, X_pred:list|np.ndarray, X_scaled:pd.DataFrame) -> (tuple[pd.DataFrame, np.ndarray]):
    '''Read the pre-trained model and finds the neighbors and distances to the
    sample'''

    distance, indices = neigh[type_selected].kneighbors(X_pred)
    # pred_df=df.iloc[indices[0]] #Commented so it returns a np.array instead of pd.DataFrame
    indices_original=[]
    for i in indices[0]:
        indices_original.append(X_scaled[X_scaled[f"type_{type_selected}"]>0].iloc[i].name)

    return distance, np.array(indices_original)
