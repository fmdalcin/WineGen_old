from sklearn.neighbors import NearestNeighbors

def train(X_scaled):
    '''Receives the X scaled Dataframe and trains the classification model'''

    neigh = NearestNeighbors(n_neighbors=200, algorithm='ball_tree', n_jobs=-1)
    neigh.fit(X_scaled)

    return neigh

def predict(neigh, X_pred, df):
    '''Read the pre-trained model and finds the neighbors and distances to the
    sample'''

    distance, indices = neigh.kneighbors(X_pred)
    pred_df=df.iloc[indices[0]]

    return pred_df, distance
