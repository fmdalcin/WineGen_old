import pandas as pd
import numpy as np
from pathlib import Path


# For NLP pre-processing text
from nltk.corpus import stopwords
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize

# For Vectorizing tokens and LDA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

token_clean_path=Path('./raw_data/token_cleaned_df.csv')

#Global Variables
wines = ['red', 'white', 'rose', 'spark']

def clean_text (text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, ' ') # Remove Punctuation
        text = text.strip() ## remove whitespaces
        lowercased = text.lower() # Lower Case
        tokenized = word_tokenize(lowercased) # Tokenize
        words_only = [word for word in tokenized if word.isalpha()] # Remove numbers
        stop_words = set(stopwords.words('english')) # Make stopword list
        without_stopwords = [word for word in words_only if not word in stop_words] # Remove Stop Words
        lemma=WordNetLemmatizer() # Initiate Lemmatizer
        lemmatized = [lemma.lemmatize(word) for word in without_stopwords] # Lemmatize
        cleaned = ' '.join(lemmatized) # Join back to a string
    return cleaned


def split_token_datasets(df:pd.DataFrame):

    if token_clean_path.is_file():
        df = pd.read_csv(token_clean_path, index_col=0)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'ID'}, inplace=True)

    else:
        # Apply to all texts
        df['description_clean'] = df.description.apply(clean_text)
        df.reset_index(inplace=True)
        df.to_csv('./raw_data/token_cleaned_df.csv', index=False)


    # Spliting dataset by wine type (4 datasets now)
    df_red = df[df['type']=='red'].reset_index()
    df_white = df[df['type']=='white'].reset_index()
    df_rose = df[df['type']=='rose'].reset_index()
    df_spark = df[df['type']=='sparkling'].reset_index()

    return df_red, df_white, df_rose, df_spark

def vectoriser(df_red:pd.DataFrame, df_white:pd.DataFrame, df_rose:pd.DataFrame, df_spark:pd.DataFrame):
    '''Generating Tokens with TfidfVectorizer ("tfidf_tokens")'''

    datasets = {'red':df_red, 'white':df_white, 'rose':df_rose, 'spark':df_spark}
    vectorized_descriptions= {}

    for type in wines:
        vectorizer_tfidf = TfidfVectorizer(min_df=0.05, max_df=0.9, max_features=80)
        vectorized_descr = vectorizer_tfidf.fit_transform(datasets[type]["description_clean"])
        vectorized_descriptions[f'vectorized_descr_{type}'] = pd.DataFrame(vectorized_descr.toarray(), columns = vectorizer_tfidf.get_feature_names_out())


    # Generated dataframes:   vectorized_descr_red,   vectorized_descr_white,  vectorized_descr_rose,  vectorized_descr_spark

    print(f"{'Wine Data': >8}\
        No.Dirty Tokens    ---   still need cleaning, several are meaningless (cleaning done in next cells)\n-----------------------------\
                                            \n{'Red :': >11}\t{vectorized_descriptions['vectorized_descr_red'].shape[1]}\
                                            \n{'White :': >11}\t{vectorized_descriptions['vectorized_descr_white'].shape[1]}\
                                            \n{'Rose :': >11}\t{vectorized_descriptions['vectorized_descr_rose'].shape[1]}\
                                            \n{'Sparkling :': >11}\t{vectorized_descriptions['vectorized_descr_spark'].shape[1]}")

    not_useful_tokens_red = ['aroma', 'bit', 'black', 'blend', 'bodied', 'cabernet', 'come',
                             'concentrated', 'dark', 'dense', 'dried', 'drink', 'feel', 'fine',
                             'finish', 'firm', 'flavor', 'full', 'give', 'good', 'hint', 'juicy',
                             'like', 'made', 'merlot', 'nose', 'note', 'offer', 'open', 'palate',
                             'pinot', 'red', 'sauvignon', 'show', 'structure', 'structured', 'syrah',
                             'texture', 'touch', 'vineyard', 'well', 'wine', 'year']

    not_useful_tokens_white = ['acidity', 'aroma', 'balance', 'balanced', 'blanc', 'blend',
                               'bodied', 'character', 'chardonnay', 'drink', 'dry', 'feel',
                               'finish', 'flavor',       'full', 'give', 'good', 'hint', 'like',
                               'long', 'medium', 'nose', 'note', 'offer', 'palate', 'riesling',
                               'sauvignon', 'show', 'style', 'texture', 'touch', 'well', 'white',
                               'wine', 'yellow']

    not_useful_tokens_rose = ['aftertaste', 'aroma', 'attractive', 'blend', 'bodied', 'character',
                              'color', 'delicious', 'drink', 'end', 'finish', 'flavor', 'food', 'full',
                              'give', 'good', 'grenache', 'hint', 'like', 'lively', 'made', 'medium',
                              'nose', 'note', 'offer', 'palate', 'pale', 'pink', 'pinot', 'ready', 'red',
                              'show', 'style', 'syrah', 'texture', 'touch', 'well', 'white', 'wine']

    not_useful_tokens_spark = ['age', 'aroma', 'blend', 'bottle', 'brut', 'bubble', 'bubbly',
                               'champagne', 'character', 'chardonnay', 'color', 'drink', 'elegant',
                               'feel', 'fine', 'finish', 'flavor', 'full', 'give', 'good', 'hint',
                               'like', 'lively', 'made', 'mouth', 'noir', 'nose', 'note', 'offer',
                               'palate', 'pinot', 'prosecco', 'ready', 'red', 'show', 'sparkler',
                               'sparkling', 'style', 'texture', 'tight', 'touch', 'well', 'white',
                               'wine', 'year', 'yellow']

    # vectorized_descriptions['vectorized_descr_red'].drop(columns=not_useful_tokens_red, inplace=True)

    for type in wines:
        for bad_token in not_useful_tokens_red:
            if bad_token in vectorized_descriptions[f'vectorized_descr_{type}'].columns:
                vectorized_descriptions[f'vectorized_descr_{type}'].drop(columns=bad_token, inplace=True)
        vectorized_descriptions[f'vectorized_descr_{type}']['ID'] = datasets[type]['ID']


    return vectorized_descriptions['vectorized_descr_red'], vectorized_descriptions['vectorized_descr_white'], vectorized_descriptions['vectorized_descr_rose'], vectorized_descriptions['vectorized_descr_spark']

def top_tokens(token_weights, top_words, columns_wine_clusters):

    # Dimensions
    n_components = token_weights.shape[0]   # number of clusters (topics)
    weight_headers = [f'weights_{n}' for n in range(1, n_components+1)]
    cluster_headers = columns_wine_clusters  # from previous cells
    indices = [f'token{n}' for n in range(1, top_words+1)]  #  output table row-indices

    ## Finding top tokens for each cluster (topic)
    top = pd.DataFrame()
    for cluster in range(n_components):
        cluster_df = token_weights.iloc[cluster].sort_values(ascending = False).head(top_words)
        top[cluster_headers[cluster]] = cluster_df.index.tolist()
        top[weight_headers[cluster]] = cluster_df.values.tolist()

    top = top.set_axis(indices)
    return top

def lda_modeler(vectorized_descr_red:pd.DataFrame, vectorized_descr_white:pd.DataFrame, vectorized_descr_rose:pd.DataFrame, vectorized_descr_spark:pd.DataFrame):
    n_components = 10   # number of clusters (topics) we want
    columns_wine_clusters = [f'wine_cluster_{n}' for n in range(1, n_components+1)] # creating column names

    vectorized_descr={'red':vectorized_descr_red, 'white':vectorized_descr_white,
                      'rose':vectorized_descr_rose, 'spark':vectorized_descr_spark}
    lda_models={}
    lda_wine_cluster_prob={}

    for type in wines:
        lda_model = LatentDirichletAllocation(n_components=n_components)
        lda_model.fit(vectorized_descr[type].drop(columns=['ID']))
        lda_models[type]=lda_model
        lda_wine_cluster_prob[type] = pd.DataFrame(lda_model.transform(vectorized_descr[type].drop(columns=['ID'])), columns = columns_wine_clusters)
        lda_wine_cluster_prob[type]['ID'] = vectorized_descr[type]['ID']
        csv_path = f'./raw_data/lda_wine_cluster_prob_{type}.csv'
        lda_wine_cluster_prob[type].to_csv(csv_path, index=True)


    lda_token_weights={}
    top_token={}
    for type in wines:
        columns_tokens = vectorized_descr[type].drop(columns="ID").columns
        lda_token_weights[type] = pd.DataFrame(lda_models[type].components_, columns = columns_tokens)
        lda_token_weights[type].loc["total"] = lda_token_weights[type].sum()
        lda_token_weights[type].sort_values(by = "total", axis = 1, ascending = False, inplace=True)
        lda_token_weights[type].drop("total", inplace=True)  # row "total" deleted because was needed only to sort tokens by overall importance
        lda_token_weights[type] = lda_token_weights[type].set_axis(columns_wine_clusters)
        top_token[type]=top_tokens(lda_token_weights[type],30, columns_wine_clusters)
        top_token[type].to_csv(f'./raw_data/top_token_{type}.csv', index=False)

    return lda_models, lda_wine_cluster_prob, lda_token_weights, top_token
