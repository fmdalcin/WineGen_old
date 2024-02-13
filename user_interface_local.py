import streamlit as st
import pandas as pd
import model
import requests
from main import X_scaled, neigh, df, vectorized_descriptions, top_token, lda_wine_cluster_prob
from token_processing import generate_tokens_list
from final_score_ranking import process_selected_tokens
from user_input import process_user_input
import json

def generate_predictions(all_index:list, df:pd.DataFrame, type_selected:str)->dict:

    pred_df = process_user_input(all_index, X_scaled, neigh, df, type_selected)
    options = generate_tokens_list(vectorized_descriptions, type_selected,pred_df)
    st.session_state['API_response1_flag']=True
    return options, pred_df

def main():
    st.title("WineGen - Smart Wine Suggestions \n Powered by Machine Learning")
    # df = pd.read_csv("./raw_data/preprocessed_data.csv")
    df_main = df.reset_index()

    wine_type = sorted(df_main["type"].unique())
    selected_wine_type = st.selectbox(f"Select wine type" , wine_type, key="wine_type")
    filtered_df_wine_type = df_main[df_main["type"] == selected_wine_type]


    col1, col2, col3, col4, col5, col6 = st.columns(6)
    clear_button = st.button("Clear Selection")
    if clear_button:
        st.session_state['recommendations'] = None
        st.session_state['API_response1_flag']=False


    with col1:

        country_placeholder = st.empty()
        # Get unique values from the "country" column
        countries = sorted(filtered_df_wine_type["country"].unique())

        # Create a dropdown list using selectbox for the country

        selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox1")
        if clear_button:
            # Remove the dropdown box by emptying the placeholder
            country_placeholder.empty()
            # Reset the selected_country to the default value
            selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox2")

        #  Filter DataFrame based on the selected country
        filtered_df_country = filtered_df_wine_type[filtered_df_wine_type["country"] == selected_country]

        # Get unique values from the "variety" column for the selected country
        varieties_for_country = sorted(filtered_df_country["variety_adj"].unique())

        # Create a second dropdown list for the variety
        selected_variety = st.selectbox(f"Select a variety for {selected_country}", varieties_for_country, key="selected_variety1")

        # Filter DataFrame based on the selected variety
        filtered_df_variety = filtered_df_country[filtered_df_country["variety_adj"] == selected_variety]

        # Get unique values from the "winery" column for the selected variety
        wineries_for_variety = sorted(filtered_df_variety["winery"].unique())

        # Create a third dropdown list for the winery
        selected_winery = st.selectbox(f"Select a winery for {selected_variety} in {selected_country}", wineries_for_variety, key="winery for variety1")

        # Filter DataFrame based on the selected winery
        filtered_df_winery = filtered_df_variety[filtered_df_variety["winery"] == selected_winery]

        # Get unique values from the "title" column for the selected winery
        titles_for_winery = sorted(filtered_df_winery["title"].unique())

        # Create a fourth dropdown list for the title
        selected_title1 = st.selectbox(f"Select a wine for {selected_winery} in {selected_country} ({selected_variety})", titles_for_winery, key="titles_for_winery1")

    with col2:

        country_placeholder = st.empty()
        # Get unique values from the "country" column
        countries = sorted(filtered_df_wine_type["country"].unique())

        # Create a dropdown list using selectbox for the country

        selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox3")
        if clear_button:
            # Remove the dropdown box by emptying the placeholder
            country_placeholder.empty()
            # Reset the selected_country to the default value
            selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox4")

        #  Filter DataFrame based on the selected country
        filtered_df_country = filtered_df_wine_type[filtered_df_wine_type["country"] == selected_country]

        # Get unique values from the "variety" column for the selected country
        varieties_for_country = sorted(filtered_df_country["variety_adj"].unique())

        # Create a second dropdown list for the variety
        selected_variety = st.selectbox(f"Select a variety for {selected_country}", varieties_for_country, key="selected_variety2")

        # Filter DataFrame based on the selected variety
        filtered_df_variety = filtered_df_country[filtered_df_country["variety_adj"] == selected_variety]

        # Get unique values from the "winery" column for the selected variety
        wineries_for_variety = sorted(filtered_df_variety["winery"].unique())

        # Create a third dropdown list for the winery
        selected_winery = st.selectbox(f"Select a winery for {selected_variety} in {selected_country}", wineries_for_variety, key="winery for variety2")

        # Filter DataFrame based on the selected winery
        filtered_df_winery = filtered_df_variety[filtered_df_variety["winery"] == selected_winery]

        # Get unique values from the "title" column for the selected winery
        titles_for_winery = sorted(filtered_df_winery["title"].unique())

        # Create a fourth dropdown list for the title
        selected_title2 = st.selectbox(f"Select a wine for {selected_winery} in {selected_country} ({selected_variety})", titles_for_winery, key="titles_for_winery2")

    with col3:

        country_placeholder = st.empty()
        # Get unique values from the "country" column
        countries = sorted(filtered_df_wine_type["country"].unique())

        # Create a dropdown list using selectbox for the country

        selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox5")
        if clear_button:
            # Remove the dropdown box by emptying the placeholder
            country_placeholder.empty()
            # Reset the selected_country to the default value
            selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox6")

        #  Filter DataFrame based on the selected country
        filtered_df_country = filtered_df_wine_type[filtered_df_wine_type["country"] == selected_country]

        # Get unique values from the "variety" column for the selected country
        varieties_for_country = sorted(filtered_df_country["variety_adj"].unique())

        # Create a second dropdown list for the variety
        selected_variety = st.selectbox(f"Select a variety for {selected_country}", varieties_for_country, key="selected_variety3")

        # Filter DataFrame based on the selected variety
        filtered_df_variety = filtered_df_country[filtered_df_country["variety_adj"] == selected_variety]

        # Get unique values from the "winery" column for the selected variety
        wineries_for_variety = sorted(filtered_df_variety["winery"].unique())

        # Create a third dropdown list for the winery
        selected_winery = st.selectbox(f"Select a winery for {selected_variety} in {selected_country}", wineries_for_variety, key="winery for variety3")

        # Filter DataFrame based on the selected winery
        filtered_df_winery = filtered_df_variety[filtered_df_variety["winery"] == selected_winery]

        # Get unique values from the "title" column for the selected winery
        titles_for_winery = sorted(filtered_df_winery["title"].unique())

        # Create a fourth dropdown list for the title
        selected_title3 = st.selectbox(f"Select a wine for {selected_winery} in {selected_country} ({selected_variety})", titles_for_winery, key="titles_for_winery3")

    with col4:

        country_placeholder = st.empty()
        # Get unique values from the "country" column
        countries = sorted(filtered_df_wine_type["country"].unique())

        # Create a dropdown list using selectbox for the country

        selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox7")
        if clear_button:
            # Remove the dropdown box by emptying the placeholder
            country_placeholder.empty()
            # Reset the selected_country to the default value
            selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox8")

        #  Filter DataFrame based on the selected country
        filtered_df_country = filtered_df_wine_type[filtered_df_wine_type["country"] == selected_country]

        # Get unique values from the "variety" column for the selected country
        varieties_for_country = sorted(filtered_df_country["variety_adj"].unique())

        # Create a second dropdown list for the variety
        selected_variety = st.selectbox(f"Select a variety for {selected_country}", varieties_for_country, key="selected_variety4")

        # Filter DataFrame based on the selected variety
        filtered_df_variety = filtered_df_country[filtered_df_country["variety_adj"] == selected_variety]

        # Get unique values from the "winery" column for the selected variety
        wineries_for_variety = sorted(filtered_df_variety["winery"].unique())

        # Create a third dropdown list for the winery
        selected_winery = st.selectbox(f"Select a winery for {selected_variety} in {selected_country}", wineries_for_variety, key="winery for variety4")

        # Filter DataFrame based on the selected winery
        filtered_df_winery = filtered_df_variety[filtered_df_variety["winery"] == selected_winery]

        # Get unique values from the "title" column for the selected winery
        titles_for_winery = sorted(filtered_df_winery["title"].unique())

        # Create a fourth dropdown list for the title
        selected_title4 = st.selectbox(f"Select a wine for {selected_winery} in {selected_country} ({selected_variety})", titles_for_winery, key="titles_for_winery4")

    with col5:

        country_placeholder = st.empty()
        # Get unique values from the "country" column
        countries = sorted(filtered_df_wine_type["country"].unique())

        # Create a dropdown list using selectbox for the country

        selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox9")
        if clear_button:
            # Remove the dropdown box by emptying the placeholder
            country_placeholder.empty()
            # Reset the selected_country to the default value
            selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox10")

        #  Filter DataFrame based on the selected country
        filtered_df_country = filtered_df_wine_type[filtered_df_wine_type["country"] == selected_country]

        # Get unique values from the "variety" column for the selected country
        varieties_for_country = sorted(filtered_df_country["variety_adj"].unique())

        # Create a second dropdown list for the variety
        selected_variety = st.selectbox(f"Select a variety for {selected_country}", varieties_for_country, key="selected_variety5")

        # Filter DataFrame based on the selected variety
        filtered_df_variety = filtered_df_country[filtered_df_country["variety_adj"] == selected_variety]

        # Get unique values from the "winery" column for the selected variety
        wineries_for_variety = sorted(filtered_df_variety["winery"].unique())

        # Create a third dropdown list for the winery
        selected_winery = st.selectbox(f"Select a winery for {selected_variety} in {selected_country}", wineries_for_variety, key="winery for variety5")

        # Filter DataFrame based on the selected winery
        filtered_df_winery = filtered_df_variety[filtered_df_variety["winery"] == selected_winery]

        # Get unique values from the "title" column for the selected winery
        titles_for_winery = sorted(filtered_df_winery["title"].unique())

        # Create a fourth dropdown list for the title
        selected_title5 = st.selectbox(f"Select a wine for {selected_winery} in {selected_country} ({selected_variety})", titles_for_winery, key="titles_for_winery5")



    #print(all_index)
    if 'API_response1_flag' not in st.session_state:
        st.session_state['API_response1_flag']=False
    all_index = []  # Initialize all_index outside the if block
    selected_df = df_main[(df_main["title"] == selected_title1) | (df_main["title"] == selected_title2) | (df_main["title"] == selected_title3) | (df_main["title"] == selected_title4) | (df_main["title"] == selected_title5)]
    if selected_title1:
        st.write('Your wine list:')
        all_index = selected_df['ID'].to_list() # Assign value to all_index
        st.dataframe(selected_df.drop(columns=['description',"ID", 'type']), hide_index=True)
    if st.button("Send Wine List"):
        print(all_index)
        # Make API call to FastAPI endpoint #1
        if all_index:
            st.session_state['options'], st.session_state['predictions'] = generate_predictions(all_index, df_main, selected_wine_type)
            # st.write(pred_df)


    #Token list
    selected_tokens = []
    if st.session_state['API_response1_flag']:
        st.write("Custom Tasting Notes")
        question = "We have selected the most relevant features of your first wine list. To refine your selection, select the features that you consider of high importance"
        st.write(question)
        # wine_taste_dictionary = {'token': 'Full-bodied, rich, bold, tannic, blue ,black, grey, yellow'}
        # options = [option.strip() for value in wine_taste_dictionary.values() for option in value.split(',')]

        options=st.session_state['options']
        num_columns = 4  # Number of columns in the table
        num_options = len(options)
        num_rows = -(-num_options // num_columns)  # Ceiling division to calculate number of rows

        # Create columns for horizontal display
        columns = st.columns(num_columns)


        selected_tokens = []
        # Display checkboxes horizontally
        for i in range(num_rows):
            row_options = options[i*num_columns : (i+1)*num_columns]
            for j, option in enumerate(row_options):
                selected = columns[j].checkbox(option)
                if selected:
                    selected_tokens.append(option)
        # selected_tokens = ['plum', 'spice', 'fruit']
    # Display selected options
        # st.write('You have selected:', selected_tokens)

    if st.session_state['API_response1_flag']:
        if st.button("Get Recommendations"):
            st.session_state['recommendations'] = None
            selected_tokens = selected_tokens
            print(selected_tokens)
            # Make API call to FastAPI endpoint #1

            lda_selected = process_selected_tokens(selected_wine_type,selected_tokens, top_token, lda_wine_cluster_prob,st.session_state.predictions)
            st.session_state['recommendations'] = lda_selected

    if 'recommendations' in st.session_state:
        if st.session_state.recommendations:
                final_list_df=df_main[df_main['ID'].isin(st.session_state.recommendations)]
                st.dataframe(final_list_df.drop(columns=['description',"ID", 'type']), hide_index=True)








if __name__ == "__main__":
    main()
