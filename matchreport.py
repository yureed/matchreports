import streamlit as st
import pandas as pd
from st_supabase_connection import SupabaseConnection

# Initialize the Supabase connection
conn = st.connection("supabase", type=SupabaseConnection)

# Replace 'your-url' and 'your-key' with your Supabase project URL and API key
supabase_url = st.secrets["sb_url"]
supabase_key = st.secrets["sb_api"]

# Connect to Supabase
supabase: Client = create_client(supabase_url, supabase_key)

# Function to query data from a table
def query_table(table_name):
    data = supabase.table(table_name).select("*").execute()
    if data.error:
        st.error(f"Error fetching data from {table_name}: {data.error.message}")
        return pd.DataFrame()
    return pd.DataFrame(data.data)

# Query and load data from Supabase
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

# Additional Streamlit features can be added as needed
