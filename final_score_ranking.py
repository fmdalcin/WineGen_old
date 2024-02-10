import pandas as pd
import numpy as np


def process_selected_tokens(type_selected:str, selected_tokens_list:str, top_token:dict, lda_wine_cluster_prob:dict, pred_df:pd.DataFrame)->list:
    top_tokens=top_token[type_selected]
    cluster_ranking={}
    for i in range(0,10):
        cluster_ranking[f'wine_cluster_{i+1}']=0
        for token in selected_tokens_list:
            if token in top_tokens.iloc[:,i*2].to_list():
                cluster_ranking[f'wine_cluster_{i+1}']+=top_tokens[f'weights_{i+1}'][top_tokens[f'wine_cluster_{i+1}']==token].sum()
    top_clusters=pd.DataFrame(list(cluster_ranking.items())).sort_values(1, ascending=False)[0][0:2].to_list()
    top_clusters.append('ID')

    lda_wine_cluster = lda_wine_cluster_prob[type_selected]

    lda_selected=pd.merge(pred_df, lda_wine_cluster[top_clusters], on="ID", how="inner")
    lda_selected['total_cluster']=lda_selected.iloc[:,3]+lda_selected.iloc[:,4]
    lda_selected['distance_rank']=lda_selected['total_distance'].rank(ascending=False)
    lda_selected['cluster_rank']=lda_selected['total_cluster'].rank(ascending=True)
    #For now this is assuming a 1-to-1 weight ratio for distance and token match
    lda_selected['final_rank']=lda_selected['distance_rank']+lda_selected['cluster_rank']
    lda_selected.sort_values(by='final_rank', ascending=False)

    return lda_selected.iloc[0:10]['match_wine'].tolist()
