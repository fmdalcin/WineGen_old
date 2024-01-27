import streamlit as st

import numpy as np
import pandas as pd

df = pd.read_csv("./raw_data/preprocessed_data.csv")
col1, col2, col3= st.columns(3)
print(df.columns)
with col1:

# Get unique values from the "country" column
    countries = sorted(df["country"].unique())

# Create a dropdown list using selectbox for the country
    selected_country  = st.selectbox("Select a country", countries)


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
    st.write("This is inside column 2")

with col3:
    st.write("This is inside column 3")

#with col4:
    #st.write("This is inside column 4")

# Display the selected country, selected variety, selected winery, selected title, and filtered DataFrame
#st.write(f"You selected country: {selected_country}")
#st.write(f"You selected variety: {selected_variety}")
#st.write(f"You selected winery: {selected_winery}")
#st.write(f"You selected title: {selected_title}")
#st.write(filtered_df_winery[filtered_df_winery["title"] == selected_title])
