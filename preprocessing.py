import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

def clean_rawdata():
    '''Once called, reads the raw data and convert it in a Pandas Dataframe.
        Carries out the cleaning tasks and returns a clean Dataframe, also saves
        a csv file locally'''


    #Clean duplicates and deals with unmatched grape varieties
    df = pd.read_csv("./raw_data/wine_data_csv.csv", index_col=0)
    df.drop_duplicates(inplace=True)
    columns_to_drop=['designation','region_2','taster_name','taster_twitter_handle','region_1']
    df.drop(columns=columns_to_drop,inplace=True)
    df.dropna(inplace=True)
    df.drop(df[df['type']=='delete'].index,inplace=True)
    # df.to_csv("./raw_data/preprocessed_data.csv", index=False) #backup file moved to after resolve_synonyms()
    return df

def resolve_synonyms(df:pd.DataFrame)->pd.DataFrame:
    '''Reads the original data and implement standardised names for grape
    varieties. The user does not know this happens in the background,
    recommendations are still provided in the original variety.'''
    # Loading wine synonyms input data from file created with unique grape for one or several wines
    syns_raw = pd.read_csv("./raw_data/wine_synonyms_csv.csv", index_col=0)
    # Extracting the single column from the synonyms files that has all synonyns for each row
    all_grape_names = syns_raw.NAME_ALL.str.split(', ')
    # The synonyms file has multiple rows (synonyms) that hasn't been unified (synonymised) properly.
    # Here a table of synonyms is created to consolidate all synonyms.
    syns = {}     # for each item: key will be main synonym, values will include all synonyms including the main one (used for key)
    for row in all_grape_names:
        flat_dict = [num for elem in list(syns.values()) for num in elem]
        # checking if synonyms in each row are already present is the dictionary being created
        # if not, it creates a key and values
        if any(grape in flat_dict for grape in row) == False:
            syns[row[0]] = row
        # if yes, adds the new synonyns that don't exist yet in the list
        else:
            res = next((sub for sub in syns if any(grape in syns[sub] for grape in row) == True), None)
            syns[res].extend([item for item in row if item not in syns[res]])
    del flat_dict
    # Using the unified synonym table, it populates the column for grape vaiety in the main data table
    # with the main synonyms for each grape variety
    df['variety_adj'] = df['variety'].apply(lambda grape: next((key for key, value in syns.items() if grape in value), None))
    df.drop(columns=['variety'], inplace=True)
    df.to_csv("./raw_data/preprocessed_data.csv", index=True)

    return df


def scaling(df):
    '''Scales the columns of the dataset in an X and reference.'''

    #Description and tiles won't be used in the model, but are required in the
    # output for the user


    X=df.drop(columns=['description','title'])
    X.reset_index(inplace=True, drop=True)

    #Encoding text fields into unique distinct numbers
    label_encoder_country = LabelEncoder()
    label_encoder_province = LabelEncoder()
    label_encoder_winery = LabelEncoder()
    label_encoder_variety = LabelEncoder()
    label_encoder_region = LabelEncoder()

    X['country'] = label_encoder_country.fit_transform(X['country'])
    X['province'] = label_encoder_province.fit_transform(X['province'])
    X['winery'] = label_encoder_winery.fit_transform(X['winery'])
    X['variety_adj'] = label_encoder_variety.fit_transform(X['variety_adj'])
    X['region'] = label_encoder_region.fit_transform(X['region'])

    #Type is a strong required feature. One Hot Encoded instead of labeled to
    # ensure it is factored in and not scaled.
    one_hot_encoder_type = OneHotEncoder(sparse_output=False)

    type_encoded = one_hot_encoder_type.fit_transform(X[['type']])
    type_categories = one_hot_encoder_type.get_feature_names_out(['type'])
    type_encoded=pd.DataFrame(type_encoded)

    X.drop(columns=['type'], inplace=True)



    #Scaling features so the magnitude of the encoded number does not affect
    #the outcome. Consider not scaling 'points' and 'price' in future versions.
    scaler = RobustScaler()
    X_scaled=scaler.fit_transform(X)
    X_scaled=pd.DataFrame(X_scaled, columns=X.columns)

    #The OneHotEncoded Features are added to the scaled DataFrame
    X_scaled[type_categories] = type_encoded
    # X_scaled.drop(columns=['ID'], inplace=True)
    features_weights=[1,1,1,1,2,3,1,3,3,3,3]
    X_scaled=X_scaled*features_weights
    with open('./models/scaled_data.pkl', 'wb') as file:
        pickle.dump(X_scaled, file)

    return X_scaled
