import streamlit as st
import pandas as pd
from api import predict
import model
import requests
from main import X_scaled
from main import neigh
import main

def main():
    st.title("Wine Selection App")
    df = pd.read_csv("./raw_data/preprocessed_data.csv")


    wine_type = sorted(df["type"].unique())
    selected_wine_type = st.selectbox(f"Select wine type" , wine_type, key="wine_type")
    filtered_df_wine_type = df[df["type"] == selected_wine_type]


    col1, col2, col3, col4, col5, col6 = st.columns(6)
    clear_button = st.button("Clear Country")


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
        selected_title1 = st.selectbox(f"Select a title for {selected_winery} in {selected_country} ({selected_variety})", titles_for_winery, key="titles_for_winery1")

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
        selected_title2 = st.selectbox(f"Select a title for {selected_winery} in {selected_country} ({selected_variety})", titles_for_winery, key="titles_for_winery2")

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
        selected_title3 = st.selectbox(f"Select a title for {selected_winery} in {selected_country} ({selected_variety})", titles_for_winery, key="titles_for_winery3")

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
        selected_title4 = st.selectbox(f"Select a title for {selected_winery} in {selected_country} ({selected_variety})", titles_for_winery, key="titles_for_winery4")

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
        selected_title5 = st.selectbox(f"Select a title for {selected_winery} in {selected_country} ({selected_variety})", titles_for_winery, key="titles_for_winery5")


    #st.table(selected_df)
    #print(all_index)

    #Token list
    preselected_tokens = ['berry', 'coffee', 'oak', 'ripe', 'spicy']
    default_tokens = ['oak', 'ripe']
    question = "We have selected the most relevant features of your first wine list. To refine your selection, uncheck the features that you consider of low importance"
    user_tokens = st.multiselect(question, preselected_tokens, default_tokens)
    st.write('You have selected:', user_tokens)



    # if st.button("Get Prediction"):
    #     #Make a request to the FastAPI backend
    #     api_url = "http://localhost:8000/predict"

    #     processed_input_dict = predict()
    #     params = (processed_input_dict)

    #     response = requests.get(api_url, params=params)


    #     # Display the API response
    #     # if response.status_code == 200:
    #     st.success(f"API Response: {response.json()}")
    # else:
    #     st.success(f"Select country")



    all_index = []  # Initialize all_index outside the if block
    if st.button("Get Prediction"):
        selected_df = df[(df["title"] == selected_title1) | (df["title"] == selected_title2) | (df["title"] == selected_title3) | (df["title"] == selected_title4) | (df["title"] == selected_title5)]
        all_index = selected_df.index.to_list() # Assign value to all_index
        #st.table(selected_df)
        #st.write(all_index)
        print(all_index)
        # Make API call to FastAPI endpoint
        if all_index:
            #api_url = "http://localhost:8000/predict"
            api_url = " http://127.0.0.1:8000/predict"
            #response = requests.post(api_url, params={"all_index": all_index})
            #response = requests.post(api_url, json = {"all_index": all_index})
            response = requests.post(api_url, json = all_index)
            if response.status_code == 200:
                try:
                    processed_input_dict = response.json()
                    st.write("Processed Input:", processed_input_dict)
                except ValueError as e:
                    st.error("Error parsing JSON response: {}".format(e))
            else:
                st.error("Error: Request failed with status code {}".format(response.status_code))




if __name__ == "__main__":
    main()
