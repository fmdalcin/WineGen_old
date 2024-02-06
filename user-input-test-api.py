import streamlit as st
import pandas as pd
import json
import requests

def main():
    st.title("Wine Selection App")
    df = pd.read_csv("./raw_data/preprocessed_data.csv")
    col1, col2, col3, col4, col5 = st.columns(5)
    clear_button = st.button("Clear Country")


    with col1:

        country_placeholder = st.empty()
        # Get unique values from the "country" column
        countries = sorted(df["country"].unique())

        # Create a dropdown list using selectbox for the country

        selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox")
        if clear_button:
            # Remove the dropdown box by emptying the placeholder
            country_placeholder.empty()
            # Reset the selected_country to the default value
            selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox2")

        #  Filter DataFrame based on the selected country
        filtered_df_country = df[df["country"] == selected_country]

        # Get unique values from the "variety" column for the selected country
        varieties_for_country = sorted(filtered_df_country["variety_adj"].unique())

        # Create a second dropdown list for the variety
        selected_variety = st.selectbox(f"Select a variety for {selected_country}", varieties_for_country)

        # Filter DataFrame based on the selected variety
        filtered_df_variety = filtered_df_country[filtered_df_country["variety_adj"] == selected_variety]

        # Get unique values from the "winery" column for the selected variety
        wineries_for_variety = sorted(filtered_df_variety["winery"].unique())

        # Create a third dropdown list for the winery
        selected_winery = st.selectbox(f"Select a winery for {selected_variety} in {selected_country}", wineries_for_variety)

        # Filter DataFrame based on the selected winery
        filtered_df_winery = filtered_df_variety[filtered_df_variety["winery"] == selected_winery]

        # Get unique values from the "title" column for the selected winery
        titles_for_winery = sorted(filtered_df_winery["title"].unique())

        # Create a fourth dropdown list for the title
        selected_title = st.selectbox(f"Select a title for {selected_winery} in {selected_country} ({selected_variety})", titles_for_winery)

    with col2:

        country_placeholder = st.empty()
        # Get unique values from the "country" column
        countries = sorted(df["country"].unique())

        # Create a dropdown list using selectbox for the country

        selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox3")
        if clear_button:
            # Remove the dropdown box by emptying the placeholder
            country_placeholder.empty()
            # Reset the selected_country to the default value
            selected_country = country_placeholder.selectbox("Select a country", [""] +countries, key="country_selectbox4")

        #  Filter DataFrame based on the selected country
        filtered_df_country = df[df["country"] == selected_country]

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

    # Create a dataframe of selections and extract index
    selected_df = df[(df["title"] == selected_title) | (df["title"] == selected_title2)]
    all_index = selected_df.index.to_list() # Convert the Index to a list

    st.table(selected_df)
    print(all_index)

    if st.button("Get Prediction"):
        # Make a request to the FastAPI backend
        api_url = "http://localhost:8000/predict"
        params = {"all_index": all_index}

        #response = requests.post(api_url, json={"all_index": all_index})
        response = requests.get(api_url, params=params)
        print("API Response Content:", response.content)

        # Decode the JSON response
        try:
            json_response = response.json()
            st.success(f"API Response: {json_response}")
        except json.decoder.JSONDecodeError as e:
            st.error(f"Error decoding JSON response: {e}")

        # Display the API response
        # if response.status_code == 200:
        st.success(f"API Response: {response.json()}")
        # else:
        #     st.success(f"Select country")






if __name__ == "__main__":
    main()
