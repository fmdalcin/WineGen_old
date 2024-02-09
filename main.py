import os

from preprocessing import  clean_rawdata, resolve_synonyms, scaling
from model import train, predict
from user_input import process_user_input
from token_processing import split_token_datasets, vectoriser, lda_modeler

df = clean_rawdata()
df = resolve_synonyms(df)
df_red, df_white, df_rose, df_spark = split_token_datasets(df)
print(df_red)
vectorized_descr_red, vectorized_descr_white, vectorized_descr_rose, vectorized_descr_spark = vectoriser(df_red, df_white, df_rose, df_spark)
lda_modeler(vectorized_descr_red, vectorized_descr_white, vectorized_descr_rose, vectorized_descr_spark)
X_scaled = scaling(df)
neigh=train(X_scaled)
# distance, pred_df = predict(neigh,[X_scaled.iloc[23238]], df)
#API Call #1 Input
input_list=[482, 33, 1338, 106274, 8090] # Jaime's list
pred_df=process_user_input(input_list,X_scaled, neigh, df)
print(pred_df)

#run_tokens(pred_df:pd.Dataframe)->list
#API Call #1 Output
#User Selects relevant tokens

#API Call #2 Input (User selection as a list)
#Process ranking and generate recomendation

#API Call #2 Output

pred_df.to_csv("./raw_data/predictions.csv", index=False) #point to GCP Bucket and
                                                            #next API entry point runs a load data function
