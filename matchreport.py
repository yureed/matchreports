import streamlit as st
import pandas as pd
from st_supabase_connection import SupabaseConnection

# Initialize connection.
conn = st.connection("supabase", type=SupabaseConnection)

# Function to query data from a table
def query_table(table_name):
    query_result = conn.query("*", table=table_name).execute()
    return pd.DataFrame(query_result.data)



# Query and load data from the database
consolidated_defined_actions = query_table('consolidated_defined_actions')
consolidated_players = query_table('consolidated_players')
consolidated_teams = query_table('consolidated_teams')

# Load CSV file for eng_premier_league_2324
eng_premier_league_2324 = pd.read_csv('ENG-Premier League_2324.csv')

# Assuming both DataFrames have a 'game_id' column
common_game_ids = consolidated_defined_actions['game_id'].unique()

# Filter rows in eng_premier_league_2324 where game_id is in common_game_ids
filtered_df_games = eng_premier_league_2324[eng_premier_league_2324['game_id'].isin(common_game_ids)]


# Create a dropdown for selecting matches
selected_match = st.selectbox('Select a match:', filtered_df_games.apply(lambda row: f"{row['home_team']} vs {row['away_team']}", axis=1).tolist())

# Extract home and away teams
home_team, away_team = selected_match.split(' vs ')

# Find the corresponding game_id in filtered_df_games
desired_game_id = filtered_df_games.loc[(filtered_df_games['home_team'] == home_team) & (filtered_df_games['away_team'] == away_team), 'game_id'].values[0]

# Now you can use desired_game_id as needed
st.write(f"Selected Match: {selected_match}")
st.write(f"Desired Game ID: {desired_game_id}")
