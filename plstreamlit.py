#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import os
import pandas as pd

matchweeks = ["Matchweek1", "Matchweek2", "Matchweek3", "Matchweek4"]

def list_matches(matchweek):
    matchweek_path = os.path.join(matchweek)
    match_folders = os.listdir(matchweek_path)
    matches = [match for match in match_folders if os.path.isdir(os.path.join(matchweek_path, match))]
    return matches


def load_data(matchweek, selected_match):
    match_path = os.path.join(matchweek, selected_match)
    players_df = pd.read_csv(os.path.join(match_path, "players_df.csv"))
    events_data = pd.read_csv(os.path.join(match_path, "eventsdata.csv"))
    return players_df, events_data

st.title("Premier League Player Analysis")


selected_matchweek = st.sidebar.selectbox("Select Matchweek", matchweeks)


st.write(f"You selected: {selected_matchweek}")


matches_in_selected_matchweek = list_matches(selected_matchweek)


selected_match = st.sidebar.selectbox("Select Match", matches_in_selected_matchweek)


st.write(f"You selected: {selected_match}")

players_df, events_data = load_data(selected_matchweek, selected_match)

# Display player data and event data
st.write("Player Data:")
st.write(players_df)

st.write("Events Data:")
st.write(events_data)

