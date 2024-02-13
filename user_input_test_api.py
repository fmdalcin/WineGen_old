import streamlit as st
import pandas as pd
from api import predict
import model
import requests
from main import X_scaled
from main import neigh
import main
import json

def api_call_1(all_index:list)->dict:
    api_url = " http://127.0.0.1:8000/predict"

    response = requests.post(api_url, json.dumps(all_index))
    if response.status_code == 200:
        try:
            options = response.json()['tokens']
            pred_df = response.json()['predictions']
            st.session_state['API_response1_flag']=True
        except ValueError as e:
            st.error("Error parsing JSON response: {}".format(e))
    else:
        st.error("UI Error: Request failed with status code {}".format(response.status_code))
    return options, pred_df, st.session_state['API_response1_flag']

def main():
    st.title("WineGen - Smart Wine Suggestions \n Powered by Machine Learning")
    df = pd.read_csv("./raw_data/preprocessed_data.csv")


    wine_type = sorted(df["type"].unique())
    selected_wine_type = st.selectbox(f"Select wine type" , wine_type, key="wine_type")
    filtered_df_wine_type = df[df["type"] == selected_wine_type]


    col1, col2, col3, col4, col5, col6 = st.columns(6)
    clear_button = st.button("Clear Selection")


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
    selected_df = df[(df["title"] == selected_title1) | (df["title"] == selected_title2) | (df["title"] == selected_title3) | (df["title"] == selected_title4) | (df["title"] == selected_title5)]
    if selected_title1:
        st.write('Your wine list:')
        all_index = selected_df['ID'].to_list() # Assign value to all_index
        st.dataframe(selected_df.drop(columns=['description',"ID", 'type']), hide_index=True)
    if st.button("Send Wine List"):
        print(all_index)
        # Make API call to FastAPI endpoint #1
        if all_index:
            st.session_state['options'], st.session_state['predictions'], st.session_state['API_response1_flag'] = api_call_1(all_index)
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
            selected_tokens = selected_tokens
            print(selected_tokens)
            # Make API call to FastAPI endpoint #1
            if selected_tokens:
                # st.write(st.session_state.predictions)
                api_url = " http://127.0.0.1:8000/predict2"
                response = requests.post(api_url, json={'predictions':st.session_state.predictions, 'tokens':selected_tokens, 'type_selected':selected_wine_type})
                if response.status_code == 200:
                    try:
                        # processed_input_dict = response.json()
                        # st.write("Processed Input:", processed_input_dict)
                        st.session_state['recommendations'] = response.json()['recommendations']
                        # API_response2_flag=True
                    except ValueError as e:
                        st.error("Error parsing JSON response: {}".format(e))
                else:
                    st.error("UI Error: Request failed with status code {}".format(response.status_code))

    if 'recommendations' in st.session_state:
        if st.session_state.recommendations:
                final_list_df=df[df['ID'].isin(st.session_state.recommendations)]
                st.dataframe(final_list_df.drop(columns=['description',"ID", 'type']), hide_index=True)








if __name__ == "__main__":
    main()
