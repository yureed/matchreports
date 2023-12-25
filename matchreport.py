import streamlit as st
import pandas as pd
from st_supabase_connection import SupabaseConnection
import math
import numpy as np

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
home_team_name = filtered_df_games.loc[filtered_df_games['game_id'] == desired_game_id, 'home_team'].values[0]
# Find the team_id in consolidated_teams for the home team name
home_team_id = consolidated_teams.loc[consolidated_teams['team_name'] == home_team_name, 'team_id'].values[0]
away_team_name = filtered_df_games.loc[filtered_df_games['game_id'] == desired_game_id, 'away_team'].values[0]
# Find the team_id in consolidated_teams for the home team name
away_team_id = consolidated_teams.loc[consolidated_teams['team_name'] == away_team_name, 'team_id'].values[0]
# Filter rows where 'game_id' is equal to the desired value
arsenalwolves = consolidated_defined_actions[consolidated_defined_actions['game_id'] == desired_game_id]
# Assuming 'arsenalwolves' is your DataFrame
arsenalwolves['receiver'] = arsenalwolves['player_id'].shift(-1)

# For the last row, use the same player_id value
arsenalwolves.loc[arsenalwolves.index[-1], 'receiver'] = arsenalwolves['player_id'].iloc[-1]

# Convert seconds to minutes
arsenalwolves['minute'] = (arsenalwolves['time_seconds_overall'] // 60).astype(int)
mask = (arsenalwolves['start_x'] == arsenalwolves['end_x']) & (arsenalwolves['start_y'] == arsenalwolves['end_y'])

# If the type_name is dribble, replace it with 'take_on'
arsenalwolves.loc[mask & (arsenalwolves['type_name'] == 'dribble'), 'type_name'] = 'take_on'
arsenalwolves['beginning'] = np.sqrt(np.square(100-arsenalwolves['start_x']) + np.square(40 - arsenalwolves['start_y']))
arsenalwolves['end'] = np.sqrt(np.square(100 - arsenalwolves['end_x']) + np.square(40 - arsenalwolves['end_y']))
arsenalwolves['progressive'] = False

for index, row in arsenalwolves.iterrows():
    try:
        if (row['end']) / (row['beginning']) < 0.75 :
            arsenalwolves.loc[index,'progressive'] = True
    except:
        continue

goal_rows = arsenalwolves[arsenalwolves['goal_from_shot']]

# Counting the number of rows where team_id is equal to home_team_id
home_team_goal_count = goal_rows[goal_rows['team_id'] == home_team_id]

# Counting the remaining rows
away_team_goal_count = goal_rows[goal_rows['team_id'] != home_team_id]


players_df = consolidated_players[consolidated_players['game_id'] == desired_game_id]


# Filter rows where 'game_id' is equal to the desired value
passes_df = arsenalwolves[arsenalwolves['type_name'] == 'pass']
# Assuming df is your DataFrame
unique_columns = passes_df.columns.unique()
desired_types = ['pass']
xt_df = arsenalwolves[arsenalwolves['type_name'].isin(desired_types)]


def get_passes_between_df(team_id, passes_df, players_df):
    # filter for only team
    passes_df = passes_df[passes_df["team_id"] == team_id]

    # add column with first eleven players only
    passes_df = passes_df.merge(players_df[["player_id", "is_starter"]], on='player_id', how='left')
    # filter on first eleven column
    passes_df = passes_df[passes_df['is_starter'] == True]

    # calculate mean positions for players
    average_locs_and_count_df = (passes_df.groupby('player_id')
                                 .agg({'start_x': ['mean'], 'start_y': ['mean', 'count']}))
    average_locs_and_count_df.columns = ['start_x', 'start_y', 'count']
    average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['player_id', 'player_name', 'jersey_number',
                                                                            'starting_position']],
                                                                on='player_id', how='left')
    average_locs_and_count_df = average_locs_and_count_df.set_index('player_id')

    # calculate the number of passes between each position (using min/ max so we get passes both ways)
    passes_player_ids_df = passes_df.loc[:, ['original_event_id', 'player_id', 'receiver', 'team_id']]
    passes_player_ids_df['pos_max'] = (passes_player_ids_df[['player_id', 'receiver']].max(axis='columns'))
    passes_player_ids_df['pos_min'] = (passes_player_ids_df[['player_id', 'receiver']].min(axis='columns'))

    # get passes between each player
    passes_between_df = passes_player_ids_df.groupby(['pos_min', 'pos_max']).original_event_id.count().reset_index()
    passes_between_df.rename({'original_event_id': 'pass_count'}, axis='columns', inplace=True)

    # add on the location of each player so we have the start and end positions of the lines
    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_min', right_index=True)
    passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_max', right_index=True,
                                                suffixes=['', '_end'])
    return passes_between_df, average_locs_and_count_df

# Get the home team name for the specified game_id
home_team_name = filtered_df_games.loc[filtered_df_games['game_id'] == desired_game_id, 'home_team'].values[0]
# Find the team_id in consolidated_teams for the home team name
home_team_id = consolidated_teams.loc[consolidated_teams['team_name'] == home_team_name, 'team_id'].values[0]
home_passes_between_df, home_average_locs_and_count_df = get_passes_between_df(home_team_id, passes_df, players_df)
# Get the home team name for the specified game_id
away_team_name = filtered_df_games.loc[filtered_df_games['game_id'] == desired_game_id, 'away_team'].values[0]
# Find the team_id in consolidated_teams for the home team name
away_team_id = consolidated_teams.loc[consolidated_teams['team_name'] == away_team_name, 'team_id'].values[0]
away_passes_between_df, away_average_locs_and_count_df = get_passes_between_df(away_team_id, passes_df, players_df)
import matplotlib.pyplot as plt
from mplsoccer import Pitch


successful_passes = passes_df[passes_df['result_name'] == 'success']
# Filter passes for home and away teams
passes_home = successful_passes[successful_passes['team_id'] == home_team_id]
passes_away = successful_passes[successful_passes['team_id'] != home_team_id]

# Define criteria for passes entering final third and penalty area
threshold_final_third = 66.67
# Define the coordinates for the penalty area
penalty_area_x_min = 83
penalty_area_x_max = 100
penalty_area_y_min = 21
penalty_area_y_max = 79
# Filter passes entering final third and penalty area for home team
passes_home_final_third = passes_home[
    (passes_home['end_x'] >= threshold_final_third) &
    (passes_home['start_x'] <= threshold_final_third)
]

passes_home_penalty_area = passes_home[
    (penalty_area_x_min <= passes_home['end_x']) & (passes_home['end_x'] <= penalty_area_x_max) &
    (penalty_area_y_min <= passes_home['end_y']) & (passes_home['end_y'] <= penalty_area_y_max) &
    ~((penalty_area_x_min <= passes_home['start_x']) & (passes_home['start_x'] <= penalty_area_x_max) &
      (penalty_area_y_min <= passes_home['start_y']) & (passes_home['start_y'] <= penalty_area_y_max))
]

# Filter passes entering final third and penalty area for away team
passes_away_final_third = passes_away[
    (passes_away['end_x'] >= threshold_final_third) &
    (passes_away['start_x'] <= threshold_final_third)
]

passes_away_penalty_area = passes_away[
    (penalty_area_x_min <= passes_away['end_x']) & (passes_away['end_x'] <= penalty_area_x_max) &
    (penalty_area_y_min <= passes_away['end_y']) & (passes_away['end_y'] <= penalty_area_y_max) &
    ~((penalty_area_x_min <= passes_away['start_x']) & (passes_away['start_x'] <= penalty_area_x_max) &
      (penalty_area_y_min <= passes_away['start_y']) & (passes_away['start_y'] <= penalty_area_y_max))
]

# Create a pitch
pitch = Pitch(pitch_type='opta', pitch_color='black', line_color='white')

# Plot passes entering final third and penalty area for home team
fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].set_facecolor('black')
pitch.draw(ax=axes[0])
pitch.arrows(passes_away_final_third.start_x, passes_away_final_third.start_y,
             passes_away_final_third.end_x, passes_away_final_third.end_y,
             color='red', ax=axes[0])

axes[1].set_facecolor('black')
pitch.draw(ax=axes[1])
pitch.arrows(passes_home_penalty_area.start_x, passes_home_penalty_area.start_y,
             passes_home_penalty_area.end_x, passes_home_penalty_area.end_y,
             color='red', ax=axes[1])
filtered_rows = passes_home_penalty_area[passes_home_penalty_area['result_name'] == 'success']

home_xt_df = xt_df[xt_df['team_id'] == home_team_id]

# Filter data for away team
away_xt_df = xt_df[xt_df['team_id'] == away_team_id]
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import numpy as np
import json
import re
import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import colors
from mplsoccer import Pitch, FontManager, Sbopen, VerticalPitch
path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()]

# see the custom colormaps example for more ideas on setting colormaps
pearl_earring_cmap = LinearSegmentedColormap.from_list("Pearl Earring - 10 colors", ['#15242e', '#4393c4'], N=10)


# Define design parameters
BACKGROUND_COLOR = 'white'
TEXT_COLOR = 'black'
BOLD_FONT_SIZE = 35
fc = colors.to_rgba('white')
ec = colors.to_rgba('white')
NORMAL_FONT_SIZE = 28
TITLE_FONT_SIZE = 40
MAX_LINE_WIDTH = 5
MAX_MARKER_SIZE = 1500

def pass_network_visualization(ax, passes_between_df, average_locs_and_count_df, flipped=False):
    MAX_LINE_WIDTH = 5
    MAX_MARKER_SIZE = 1500  # Reduce the maximum marker size
    passes_between_df['width'] = (passes_between_df.pass_count / passes_between_df.pass_count.max() *
                                  MAX_LINE_WIDTH)
    average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count']
                                                / average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE)

    MIN_TRANSPARENCY = 0.3
    color = np.array(to_rgba('black'))
    color = np.tile(color, (len(passes_between_df), 1))
    c_transparency = passes_between_df.pass_count / passes_between_df.pass_count.max()
    c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
    color[:, 3] = c_transparency

    pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black')
    pitch.draw(ax=ax)

    if flipped:
        passes_between_df['start_x'] = pitch.dim.right - passes_between_df['start_x']
        passes_between_df['start_y'] = pitch.dim.right - passes_between_df['start_y']
        passes_between_df['start_x_end'] = pitch.dim.right - passes_between_df['start_x_end']
        passes_between_df['start_y_end'] = pitch.dim.right - passes_between_df['start_y_end']
        average_locs_and_count_df['start_x'] = pitch.dim.right - average_locs_and_count_df['start_x']
        average_locs_and_count_df['start_y'] = pitch.dim.right - average_locs_and_count_df['start_y']

    pass_lines = pitch.lines(passes_between_df.start_x, passes_between_df.start_y,
                             passes_between_df.start_x_end, passes_between_df.start_y_end, lw=passes_between_df.width,
                             color=color, zorder=1, ax=ax)
    
    for index, row in average_locs_and_count_df.iterrows():
        if flipped:
            pitch.scatter(row.start_x, row.start_y, s=row.marker_size, color='violet', edgecolors='red', linewidth=1, alpha=0.8, ax=ax)
            pitch.annotate(str(row['jersey_number']), xy=(row.start_x, row.start_y), color='black', va='center', ha='center', size=10, weight='bold', bbox=dict(boxstyle='round,pad=22.5', facecolor='none', edgecolor='none'), ax=ax)

        else:
            pitch.scatter(row.start_x, row.start_y, s=row.marker_size, color='red', edgecolors='black', linewidth=1, alpha=0.8, ax=ax)
            pitch.annotate(str(row['jersey_number']), xy=(row.start_x, row.start_y), color='black', va='center', ha='center', size=10,weight='bold', ax=ax)


    return pitch

def make_xt_zone(xtdataframe,ax,Ccolor):
    
    
        # Setup pitch
    pitch = Pitch(pitch_type='opta', line_zorder=2,
                  pitch_color='#22312b', line_color='white')
    pitch.draw(ax=ax)
    ax.set_facecolor('white')
    # Calculate the bin_statistic for xT
    xT_bin_statistic = pitch.bin_statistic_positional(xtdataframe.start_x, xtdataframe.start_y, 
                                                      values=xtdataframe.xT, statistic='sum', 
                                                      positional='full', normalize=True)

    # Overlay the xT heatmap, using a colormap that indicates intensity
    pitch.heatmap_positional(xT_bin_statistic, ax=ax, cmap='coolwarm', edgecolors='#22312b')

    # Scatter points for passes
    pitch.scatter(xtdataframe.start_x, xtdataframe.start_y, c='none', s=2, ax=ax)

    # Label the xT heatmap with the amount of xT in each zone
    xT_labels = pitch.label_heatmap(xT_bin_statistic, color='white', fontsize=12, 
                                    ax=ax, ha='center', va='center', 
                                    str_format='{:.2f}', path_effects=path_eff,weight='bold')

# Create the figure and grid layout
fig = plt.figure(figsize=(20, 20))
grid = plt.GridSpec(12, 15, figure=fig)
fig.set_facecolor(BACKGROUND_COLOR)

# Passing network visualizations
ax_home_passing = fig.add_subplot(grid[0:5, 0:4])
pass_network_visualization(ax_home_passing, home_passes_between_df, home_average_locs_and_count_df)
ax_home_passing.set_title('Home Passing Network', fontsize=15, color='red',weight='bold')

ax_away_passing = fig.add_subplot(grid[0:5, 10:14])
pass_network_visualization(ax_away_passing, away_passes_between_df, away_average_locs_and_count_df, flipped=True)
ax_away_passing.set_title('Away Passing Network', fontsize=15, color='purple',weight='bold')


import pandas as pd

# Assuming the provided code snippet is what the user has in their environment, I will replicate those steps first.

# User code starts here
# Creating dummy data to simulate the user's environment since the actual data wasn't provided
import numpy as np

# Group by 'timestamp' and count the passes in each minute
passes_per_minute = passes_away_final_third.groupby('minute').size().reset_index(name='pass_count')

# Create a DataFrame with a range of minutes from 1 to 98
all_minutes = pd.DataFrame({'minute': range(1, 99)})

# Merge the 'all_minutes' DataFrame with 'passes_per_minute' to fill missing minutes
passes_per_minute_away = all_minutes.merge(passes_per_minute, on='minute', how='left').fillna(0)

# Group by 'timestamp' and count the passes in each minute
passes_per_minute = passes_home_final_third.groupby('minute').size().reset_index(name='pass_count')

# Create a DataFrame with a range of minutes from 1 to 98
all_minutes = pd.DataFrame({'minute': range(1, 99)})

# Merge the 'all_minutes' DataFrame with 'passes_per_minute' to fill missing minutes
passes_per_minute_home = all_minutes.merge(passes_per_minute, on='minute', how='left').fillna(0)
passes_per_minute_home['rolling_passes'] = passes_per_minute_home['pass_count'].rolling(5).mean()
passes_per_minute_away['rolling_passes'] = passes_per_minute_away['pass_count'].rolling(5).mean()
passes_per_minute_home.fillna(0, inplace=True)
passes_per_minute_away.fillna(0, inplace=True)

# Combining the data
passes_total = passes_per_minute_home.merge(passes_per_minute_away, on='minute', how='outer', suffixes=['_local', '_visit'])
passes_total['dif_pass'] = passes_total['rolling_passes_local'] - passes_total['rolling_passes_visit']
# User code ends here

# Momentum plot in the middle
momentumax = fig.add_subplot(grid[2:4, 4:10])

# Plotting the momentum based on the difference in rolling passes
momentumax.plot(passes_total['minute'], passes_total['dif_pass'], color='black')
momentumax.set_title('Momentum Chart', fontsize='15', color=TEXT_COLOR,weight='bold',pad=-18)

# Filling the area under the plot based on the sign of 'dif_pass'
momentumax.fill_between(passes_total['minute'], passes_total['dif_pass'], where=passes_total['dif_pass'] > 0, color='red', alpha=0.3, interpolate=True)
momentumax.fill_between(passes_total['minute'], passes_total['dif_pass'], where=passes_total['dif_pass'] <= 0, color='violet', alpha=0.3, interpolate=True)
# Filtering the rows where goals were scored
goals = arsenalwolves[arsenalwolves['goal_from_shot']]

# Enhancing the plot
momentumax.axhline(0, color='black', linewidth=1)
# Plot goal marks for home and away teams
for _, goal in goals.iterrows():
    if goal['team_id'] == home_team_id:
        image_path = 'football_red.png'
    else:
        image_path = 'footall_white.png'

    img = plt.imread(image_path)
    zoom_factor = 0.025 if image_path == 'football_red.png' else 0.05

    imagebox = OffsetImage(img, zoom=zoom_factor)
    ab = AnnotationBbox(imagebox, (goal['minute'], 0), frameon=False)
    momentumax.add_artist(ab)

# Momentum plot in the middle
xtax = fig.add_subplot(grid[5:7, 3:11])
make_xt_zone(home_xt_df,xtax,'Reds')
xtax.set_title('Home xT Passing Area', fontsize='15', color='red',weight='bold',pad=10)
xtax = fig.add_subplot(grid[7:9, 3:11])
make_xt_zone(away_xt_df,xtax,'Purples')
xtax.set_title('Away xT Passing Area', fontsize='15', color='purple',weight='bold')

# Remove all labels
momentumax.set_xticks([])
momentumax.set_yticks([])
momentumax.set_xticklabels([])
momentumax.set_yticklabels([])

# Remove spines
momentumax.spines['top'].set_visible(False)
momentumax.spines['right'].set_visible(False)
momentumax.spines['bottom'].set_visible(False)
momentumax.spines['left'].set_visible(False)

# Set facecolor to white
momentumax.set_facecolor('white')







# Create a pitch
pitch = Pitch(pitch_type='opta', pitch_color='white', line_color='black')
home_penalty_ax =fig.add_subplot(grid[5:7, 0:4])
home_penalty_ax.set_facecolor('black')
pitch.draw(ax=home_penalty_ax)
pitch.lines(passes_home_penalty_area.start_x, passes_home_penalty_area.start_y,
                  passes_home_penalty_area.end_x, passes_home_penalty_area.end_y,
                  lw=2, transparent=True, comet=True,
                  color='red', ax=home_penalty_ax)
home_penalty_ax.set_title('Home Passes Into Penalty Area', fontsize='15', color='red',weight='bold',pad=10)

pitch = Pitch(pitch_type='opta', pitch_color='white', line_color='black')
away_penalty_ax =fig.add_subplot(grid[5:7, 10:14])
away_penalty_ax.set_facecolor('black')
pitch.draw(ax=away_penalty_ax)
pitch.lines(passes_away_penalty_area.start_x, passes_away_penalty_area.start_y,
                  passes_away_penalty_area.end_x, passes_away_penalty_area.end_y,
                  lw=2, transparent=True, comet=True,
                  color='violet', ax=away_penalty_ax)
away_penalty_ax.set_title('Away Passes Into Penalty Area', fontsize='15', color='purple',weight='bold',pad=10)
# Create a pitch
pitch = Pitch(pitch_type='opta', pitch_color='white', line_color='black')
home_ft_ax =fig.add_subplot(grid[7:9, 0:4])
home_ft_ax.set_facecolor('black')
pitch.draw(ax=home_ft_ax)
pitch.lines(passes_home_final_third.start_x, passes_home_final_third.start_y,
                  passes_home_final_third.end_x, passes_home_final_third.end_y,
                  lw=2, transparent=True, comet=True,
                  color='red', ax=home_ft_ax)
home_ft_ax.set_title('Home Passes Into Final Third', fontsize='15', color='red',weight='bold')
# Manually adjust the position of the home subplot


pitch = Pitch(pitch_type='opta', pitch_color='white', line_color='black')
away_ft_ax =fig.add_subplot(grid[7:9, 10:14])
away_ft_ax.set_facecolor('black')
pitch.draw(ax=away_ft_ax)
pitch.lines(passes_away_final_third.start_x, passes_away_final_third.start_y,
                  passes_away_final_third.end_x, passes_away_final_third.end_y,
                  lw=2, transparent=True, comet=True,
                  color='violet', ax=away_ft_ax)
away_ft_ax.set_title('Away Passes Into Final Third', fontsize='15', color='purple',weight='bold')
# Add the scoreline above the momentum chart
home_goal = len(home_team_goal_count)
away_goal = len(away_team_goal_count)
scoreline_ax = fig.add_subplot(grid[0:2, 4:10])

scoreline_text = f'{home_team_name} {home_goal} - {away_goal} {away_team_name}'

scoreline_ax.text(0.5, 0.5, scoreline_text, color=TEXT_COLOR,
                  va='center', ha='center', fontdict={'weight': 'bold', 'size': BOLD_FONT_SIZE, 'family': 'Arial'})



# Remove all labels from the scoreline_ax
scoreline_ax.set_xticks([])
scoreline_ax.set_yticks([])
scoreline_ax.set_xticklabels([])
scoreline_ax.set_yticklabels([])

# Remove spines from the scoreline_ax
scoreline_ax.spines['top'].set_visible(False)
scoreline_ax.spines['right'].set_visible(False)
scoreline_ax.spines['bottom'].set_visible(False)
scoreline_ax.spines['left'].set_visible(False)

fig.text(0.49, 0.85, 'Made by @yureedelahi', color=TEXT_COLOR, ha='center', fontdict={'weight': 'bold', 'size': 12, 'family': 'Arial'})

st.pyplot(fig)
