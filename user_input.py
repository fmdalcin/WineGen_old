import pandas as pd
from itertools import combinations
import numpy as np
from model import predict


def process_user_input (input_list:list, X_scaled:pd.DataFrame, neigh, df:pd.DataFrame) -> pd.DataFrame:
    input_df=X_scaled.iloc[input_list]

    #Create a subset of predicted wines for each wine inputted by the user
    subsets={}
    distances={}
    list_subsets=[]
    for i, wine in enumerate(input_list):
        distance, subset=predict(neigh,[X_scaled.iloc[wine]],df)
        subsets[f'subset{i}']=subset
        distances[f'distance{i}']=distance
        list_subsets.append(subset)

    # Creates callable DataFrames with the predictions
    subsets_df={}
    for key in subsets:
        subsets_df[key]=subsets[key].flatten()
    subsets_df=pd.DataFrame(subsets_df)

    distances_df={}
    for key in distances:
        distances_df[key]=distances[key].flatten()
    distances_df=pd.DataFrame(distances_df)

    # Generate all possible combinations for the number of wines inputted
    combinations_list = []
    array_indeces = []
    for r in range (2, len(list_subsets)+1):
        combinations_list.extend(combinations(list_subsets, r))
        array_indeces.extend(combinations(range(0,len(list_subsets)),r))

    # Calculates the number of matches that are intercepted (in common)
    # Places the information in a DataFrame for future reference
    i=0
    j=0
    intercepts=[]
    intercept_wines=[]
    for perm in combinations_list:
        for i, arr in enumerate(perm):
            if i ==0:
                arr_ref=arr
            else:
                arr_ref=np.intersect1d(arr_ref, arr)
        intercepts.append(arr_ref.shape[0])
        intercept_wines.append(arr_ref)
        j+=1
    wine_combinations={'wines':array_indeces,'wines_used':[len(combo) for combo in array_indeces],'intercept':intercepts,'intercept_wines':intercept_wines}
    wine_combinations=pd.DataFrame(wine_combinations)


    # Searches for the best match meeting the criteria of at least 1000 wines, and the most number
    # of inputted wines used in the prediction.
    found_best_match=False
    no_matches=False
    best_match_row=None
    best_single_wine_match=None
    for i in range(len(input_list),1,-1):
        if not found_best_match:
            if wine_combinations[wine_combinations['wines_used']==i]['intercept'].max() > 1000:
                best_match_row=wine_combinations.loc[
                (wine_combinations['intercept']==wine_combinations[wine_combinations['wines_used']==i]['intercept'].max())
                &(wine_combinations['wines_used']==i)]
                found_best_match=True
            elif i==2:
                if wine_combinations['intercept'].max()==0:
                    no_matches=True
                else:
                    best_match_row=wine_combinations.loc[wine_combinations['intercept'].max()]
    if no_matches: # If there are not combine matches, it returns the recommendations for a single with the shortest distances.
        total_distance=pd.DataFrame()
        for i, wine in enumerate(distances):
            total_distance=total_distance._append({'distance':distances[wine].sum()}, ignore_index=True)
        best_single_wine_match=total_distance.loc[total_distance['distance']==total_distance['distance'].min()]


    # Generates a DataFrame with the best predicted wines sorted by distance to the prediction point
    # (for 1 wine inputted), or sum of distances (for more than 1 wine)
    unfiltered_wine_list=pd.DataFrame()
    match_distances=[]
    if best_match_row is not None: # if multiple wines, sums the distances
        unfiltered_wine_list['match_wine']=best_match_row['intercept_wines'].values[0]
        unfiltered_wine_list['total_distance']=0
        for wine in best_match_row['wines'].values[0]:
            match_distances=[]
            for match_wine in unfiltered_wine_list['match_wine']:
                match_distances.append(distances_df.iloc[subsets_df.loc[subsets_df[f'subset{wine}']==match_wine].index[0]].iloc[wine])
            unfiltered_wine_list['total_distance']=unfiltered_wine_list['total_distance'] + match_distances
    else: #if 1 wine only, uses the distances directly provided by the model
        unfiltered_wine_list['match_wine']=subsets_df[f'subset{best_single_wine_match.index[0]}']
        unfiltered_wine_list['total_distance']=distances_df[f'distance{best_single_wine_match.index[0]}']
    unfiltered_wine_list.sort_values(by=['total_distance'], inplace=True)
    unfiltered_wine_list.reset_index(inplace=True,drop=True)

    return unfiltered_wine_list
