import os, pickle
from pathlib import Path
import pandas as pd

from preprocessing import  clean_rawdata, resolve_synonyms, scaling
from model import train, predict
from user_input import process_user_input
from token_processing import split_token_datasets, vectoriser, lda_modeler, generate_tokens_list
from final_score_ranking import process_selected_tokens


# Checks for saved pre-processed data, if present loads it, otherwise triggers preprocessing.
if Path("./raw_data/preprocessed_data.csv").is_file():
    print("Using saved preprocessed data.")
    df = pd.read_csv("./raw_data/preprocessed_data.csv", index_col=0)
else:
    print('Running data preprocessing.')
    df = clean_rawdata()
    df = resolve_synonyms(df)

# Checks for token's database. If present, loads it, otherwise process it (lenghty process)
if Path('./models/nlp_tokens.pkl').is_file():
    print("Using saved NLP Models.")
    with open('./models/nlp_tokens.pkl', 'rb') as file:
        df_red, df_white, df_rose, df_spark, vectorized_descriptions,lda_models, lda_wine_cluster_prob, lda_token_weights, top_token = pickle.load(file)
else:
    print("Recalculating NLP Models.")
    df_red, df_white, df_rose, df_spark = split_token_datasets(df)
    vectorized_descriptions = vectoriser(df_red, df_white, df_rose, df_spark)
    lda_models, lda_wine_cluster_prob, lda_token_weights, top_token = lda_modeler(vectorized_descriptions)
    nlp_tokens=(df_red, df_white, df_rose, df_spark, vectorized_descriptions,
                lda_models, lda_wine_cluster_prob, lda_token_weights, top_token)
    with open('./models/nlp_tokens.pkl', 'wb') as file:
        pickle.dump(nlp_tokens, file)

if Path('./models/scaled_data.pkl').is_file():
    print('Using saved scaled data')
    with open('./models/scaled_data.pkl', 'rb') as file:
        X_scaled = pickle.load(file)
else:
    print('Scaling data.')
    X_scaled = scaling(df)
neigh=train(X_scaled)

#API Call #1 Input
# input_list=[482, 33, 1338, 106274, 8090] # Jaime's list



#API Call #1 Output
#User Selects relevant tokens
#API Call #2 Input (User selection as a list)
#Process ranking and generate recomendation
#API Call #2 Output
# final_list = process_selected_tokens(type_selected, selected_tokens_list, top_token, lda_wine_cluster_prob,pred_df)
# print(final_list)

# pred_df.to_csv("./raw_data/predictions.csv", index=False) #point to GCP Bucket and
                                                            #next API entry point runs a load data function


# input_list=[11090, 19078, 51516, 81076]
# input_list = [35926, 85548, 108957] #reds are generating whites
# type_selected = df.iloc[input_list[0]]['type']
# pred_df=process_user_input(input_list,X_scaled, neigh, df, type_selected)
# print(pred_df)
# tokens_list=generate_tokens_list(vectorized_descriptions, type_selected,pred_df)
# selected_tokens_list = ['rich']
# final_list = process_selected_tokens(type_selected, selected_tokens_list, top_token, lda_wine_cluster_prob,pred_df)
# print(final_list)
