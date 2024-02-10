import os

from preprocessing import  clean_rawdata, resolve_synonyms, scaling
from model import train, predict
from user_input import process_user_input
from token_processing import split_token_datasets, vectoriser, lda_modeler, generate_tokens_list
from final_score_ranking import process_selected_tokens

df = clean_rawdata()
df = resolve_synonyms(df)
df_red, df_white, df_rose, df_spark = split_token_datasets(df)
vectorized_descriptions = vectoriser(df_red, df_white, df_rose, df_spark)
lda_models, lda_wine_cluster_prob, lda_token_weights, top_token = lda_modeler(vectorized_descriptions)
X_scaled = scaling(df)
neigh=train(X_scaled)
# distance, pred_df = predict(neigh,[X_scaled.iloc[23238]], df)
#API Call #1 Input
input_list=[482, 33, 1338, 106274, 8090] # Jaime's list
type_selected = df.iloc[input_list[0]]['type']

pred_df=process_user_input(input_list,X_scaled, neigh, df, type_selected)
print(pred_df)
#API Call #1 Output
tokens_list=generate_tokens_list(vectorized_descriptions, type_selected,pred_df)
#User Selects relevant tokens
#API Call #2 Input (User selection as a list)
selected_tokens_list = ['cherry', 'fruit', 'tannin']
#Process ranking and generate recomendation
#API Call #2 Output
final_list = process_selected_tokens(type_selected, selected_tokens_list, top_token, lda_wine_cluster_prob,pred_df)
print(final_list)

pred_df.to_csv("./raw_data/predictions.csv", index=False) #point to GCP Bucket and
                                                            #next API entry point runs a load data function
