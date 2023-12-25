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

# Show matches in Streamlit
st.title('Filtered Matches')
for index, row in filtered_df_games.iterrows():
    match_name = f"{row['home_team']} vs {row['away_team']}"
    st.write(match_name)
