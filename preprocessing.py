import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

def clean_rawdata():
    '''Once called, reads the raw data and convert it in a Pandas Dataframe.
        Carries out the cleaning tasks and returns a clean Dataframe, also saves
        a csv file locally'''


    #Clean duplicates and deals with unmatched grape varieties
    df = pd.read_csv("./raw_data/winemag-data-130k-v2_18Jan2024.csv", index_col=0)
    df.drop_duplicates(inplace=True)
    columns_to_drop=['designation','region_2','taster_name','taster_twitter_handle','variety','region_1']
    df.drop(columns=columns_to_drop,inplace=True)
    df.dropna(inplace=True)
    df.drop(df[df['type']=='delete'].index,inplace=True)
    df.to_csv("./raw_data/preprocessed_data.csv", index=False)
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
    print(df.columns)
    features_weights=[1,1,1,1,2,3,1,3,3,3,3]
    X_scaled=X_scaled*features_weights

    return X_scaled
