import os

from preprocessing import  clean_rawdata, resolve_synonyms, scaling
from model import train, predict
from user_input import process_user_input

df = clean_rawdata()
df = resolve_synonyms(df)
X_scaled = scaling(df)
neigh=train(X_scaled)
# distance, pred_df = predict(neigh,[X_scaled.iloc[23238]], df)
input_list=[482, 33, 1338, 106274, 8090] # Jaime's list
pred_df=process_user_input(input_list,X_scaled, neigh, df)
print(pred_df)
pred_df.to_csv("./raw_data/predictions.csv", index=False)
