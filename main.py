import os

from preprocessing import  clean_rawdata, scaling
from model import train, predict

df = clean_rawdata()
X_scaled = scaling(df)
neigh=train(X_scaled)
pred_df, distance =predict(neigh,[X_scaled.iloc[23238]], df)
print(pred_df)
