import os

from preprocessing import  clean_rawdata, resolve_synonyms, scaling
from model import train, predict

df = clean_rawdata()
df = resolve_synonyms(df)
X_scaled = scaling(df)
neigh=train(X_scaled)
pred_df, distance =predict(neigh,[X_scaled.iloc[23238]], df)
print(pred_df)
pred_df.to_csv("./raw_data/predictions.csv", index=False)
