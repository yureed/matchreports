import streamlit as st
import pandas as pd
from supabase import create_client

# Replace 'your-url' and 'your-key' with your Supabase project URL and API key
supabase_url = st.secrets["sb_url"]
supabase_key = st.secrets["sb_api"]

# Connect to Supabase
supabase = create_client(supabase_url, supabase_key)

# Query and load data from Supabase for consolidated_defined_actions
query_defined_actions = 'SELECT * FROM consolidated_defined_actions'
consolidated_defined_actions = supabase.sql(query_defined_actions).get('data')

# Query and load data from Supabase for consolidated_players
query_players = 'SELECT * FROM consolidated_players'
consolidated_players = supabase.sql(query_players).get('data')

# Query and load data from Supabase for consolidated_teams
query_teams = 'SELECT * FROM consolidated_teams'
consolidated_teams = supabase.sql(query_teams).get('data')

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

# Additional Streamlit features can be added as needed
# For example, you can create interactive widgets for filtering, sorting, etc.
# Refer to the Streamlit documentation for more options: https://docs.streamlit.io/
