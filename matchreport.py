import streamlit as st
import pandas as pd
import math
import matplotlib.patheffects as path_effects
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import json
from matplotlib.colors import LinearSegmentedColormap
import re
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib import colors
from mplsoccer import Pitch, FontManager, Sbopen, VerticalPitch
path_eff = [path_effects.Stroke(linewidth=1.5, foreground='black'), path_effects.Normal()]
import numpy as np

from streamlit_gsheets import GSheetsConnection

# Create a connection object.
conn = st.connection("gsheets", type=GSheetsConnection)
# Add a title to the Streamlit app
st.title("Premier League Match Reports")


@st.cache_data(ttl=3600) 
def read_data(worksheet):
    consolidated_data = conn.read(worksheet=worksheet, ttl="60m")
    return consolidated_data

consolidated_defined_actions = read_data("events")
consolidated_teams = read_data("teams")
consolidated_players = read_data("players")


eng_premier_league_2324 = pd.read_csv('ENG-Premier League_2324.csv')
common_game_ids = consolidated_defined_actions['game_id'].unique()

# Filter rows in eng_premier_league_2324 where game_id is in common_game_ids
filtered_df_games = eng_premier_league_2324[eng_premier_league_2324['game_id'].isin(common_game_ids)]

# Create a sidebar with match selection
selected_match = st.sidebar.selectbox('Select a match:', filtered_df_games['home_team'] + ' vs ' + filtered_df_games['away_team'])

# Get the selected home_team and away_team
selected_home_team, selected_away_team = selected_match.split(' vs ')

# Find the corresponding game_id
selected_game_id = filtered_df_games.loc[
    (filtered_df_games['home_team'] == selected_home_team) & 
    (filtered_df_games['away_team'] == selected_away_team), 'game_id'].values[0]

# Display the selected match and its game_id
st.write(f'Selected Match: {selected_match}')

report_type_options = ["Team Report", "Player Report"]
report_type = st.sidebar.selectbox("Select report type:", report_type_options)

if report_type == "Team Report":
    # Team report options
    team_report_options = [
        "General Report",
        "Passes",
        "Shot",
        "Dribbles/Carries",
        "Progressive Actions",
        "Free Kick",
        "Passes Into Final Third",
        "Passes Into Penalty Area",
        "Defensive Actions",
        "Throw In",
        "Corner",
        "Cross"
    ]
    selected_team_report = st.sidebar.selectbox("Select team report type:", team_report_options)
    st.write(f'Selected Team Report Type: {selected_team_report}')

elif report_type == "Player Report":
    # Assuming you have a DataFrame called consolidated_players
    player_names = consolidated_players.loc[consolidated_players['game_id'] == selected_game_id, 'player_name'].unique()

    # Player selection
    selected_player_name = st.sidebar.selectbox("Select player:", player_names)
    selected_player_id = consolidated_players.loc[consolidated_players['player_name'] == selected_player_name, 'player_id'].iloc[0]
    st.write(f'Selected Player: {selected_player_name} (ID: {selected_player_id})')


    # Player report options
    player_report_options = [
        "Passes",
        "Shot",
        "Dribbles/Carries",
        "Progressive Actions",
        "Free Kick",
        "Passes Into Final Third",
        "Passes Into Penalty Area",
        "Defensive Actions",
        "Throw In",
        "Corner",
        "Cross"
    ]
    selected_player_report = st.sidebar.selectbox("Select player report type:", player_report_options)
    st.write(f'Selected Player Report Type: {selected_player_report}')


desired_game_id = selected_game_id
match_info = eng_premier_league_2324.loc[eng_premier_league_2324['game_id'] == selected_game_id]
home_team_name = match_info['home_team'].values[0]
away_team_name = match_info['away_team'].values[0]


# Use consolidated_teams to get team_id for home and away teams
home_team_id = consolidated_teams.loc[consolidated_teams['team_name'] == home_team_name, 'team_id'].values[0]
away_team_id = consolidated_teams.loc[consolidated_teams['team_name'] == away_team_name, 'team_id'].values[0]

# Filter rows where 'game_id' is equal to the desired value
filteredRows = consolidated_defined_actions[consolidated_defined_actions['game_id']== desired_game_id]
matchdataframe = filteredRows
# Assuming 'matchdataframe' is your DataFrame
matchdataframe['receiver'] = matchdataframe['player_id'].shift(-1)

# For the last row, use the same player_id value
matchdataframe.loc[matchdataframe.index[-1], 'receiver'] = matchdataframe['player_id'].iloc[-1]

# Convert seconds to minutes
matchdataframe['minute'] = (matchdataframe['time_seconds_overall'] // 60).astype(int)
mask = (matchdataframe['start_x'] == matchdataframe['end_x']) & (matchdataframe['start_y'] == matchdataframe['end_y'])

# If the type_name is dribble, replace it with 'take_on'
matchdataframe.loc[mask & (matchdataframe['type_name'] == 'dribble'), 'type_name'] = 'take_on'
own_half = 50
opponents_half = 50

matchdataframe['progressive'] = False

for index, row in matchdataframe.iterrows():
    try:
        start_in_own_half = row['start_x'] <= own_half
        end_in_own_half = row['end_x'] <= own_half
        start_in_opponents_half = row['start_x'] > opponents_half
        end_in_opponents_half = row['end_x'] > opponents_half

        condition1 = start_in_own_half and end_in_own_half and (row['end_x'] - row['start_x']) >= 30
        condition2 = start_in_own_half and end_in_opponents_half and (row['end_x'] - row['start_x']) >= 15
        condition3 = start_in_opponents_half and end_in_opponents_half and (row['end_x'] - row['start_x']) >= 10

        if condition1 or condition2 or condition3:
            matchdataframe.loc[index, 'progressive'] = True
    except:
        continue

goal_rows = matchdataframe[matchdataframe['goal_from_shot']]
# Check if there are rows with 'owngoal' in result_name
owngoal_rows_matchdataframe = matchdataframe[matchdataframe['result_name'] == 'owngoal']

# Initialize goal_rows as an empty DataFrame if it hasn't been defined yet or if it's not a DataFrame
if not isinstance(goal_rows, pd.DataFrame):
    goal_rows = pd.DataFrame()

# If there are 'owngoal' rows, add them to goal_rows with the appropriate team_id
if not owngoal_rows_matchdataframe.empty:
    for index, row in owngoal_rows_matchdataframe.iterrows():
        if row['team_id'] == home_team_id:
            row['team_id'] = away_team_id
        else:
            row['team_id'] = home_team_id
        goal_rows = pd.concat([goal_rows, pd.DataFrame([row])], ignore_index=True)

# Counting the number of rows where team_id is equal to home_team_id
home_team_goal_count = goal_rows[goal_rows['team_id'] == home_team_id]

# Counting the remaining rows
away_team_goal_count = goal_rows[goal_rows['team_id'] != home_team_id]


players_df = consolidated_players[consolidated_players['game_id'] == desired_game_id]


# Filter rows where 'game_id' is equal to the desired value
passes_df = matchdataframe[matchdataframe['type_name'] == 'pass']
# Assuming df is your DataFrame
unique_columns = passes_df.columns.unique()
desired_types = ['pass']
xt_df = matchdataframe[matchdataframe['type_name'].isin(desired_types)]


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

home_passes_between_df, home_average_locs_and_count_df = get_passes_between_df(home_team_id, passes_df, players_df)

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

def general_report(home_passes_between_df,home_average_locs_and_count_df,away_passes_between_df,away_average_locs_and_count_df,
                  passes_home_final_third,passes_away_final_third,passes_away_penalty_area,passes_home_penalty_area,goal_rows,
                  home_team_goal_count,away_team_goal_count,home_team_name,away_team_name):
                      
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
        import numpy as np
        color = np.array(to_rgba('black'))
        color = np.tile(color, (len(passes_between_df), 1))
        c_transparency = passes_between_df.pass_count / passes_between_df.pass_count.max()
        c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
        color[:, 3] = c_transparency
    
        pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black')
        pitch.draw(ax=ax)
    
        if flipped:
            passes_between_df['start_y'] = pitch.dim.right - passes_between_df['start_y']
            passes_between_df['start_y_end'] = pitch.dim.right - passes_between_df['start_y_end']
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
    goals = goal_rows
    
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
    shortened_names = {
        'Manchester United': 'Man Utd',
        'Manchester City': 'Man City',
        'Sheffield United': 'Sheff Utd',
        'Newcastle': 'New Utd',
        'Bournemouth': "B'mouth",
        'Nottingham Forest': 'Nott Forest',
        'Aston Villa':"A' Villa",
        'Crystal Palace': "Palace",
        'Tottenham':'Spurs'
        # Add more mappings as needed
    }
    
    if home_team_name in shortened_names:
        home_team_shortened = shortened_names[home_team_name]
    else:
        home_team_shortened = home_team_name
    
    if away_team_name in shortened_names:
        away_team_shortened = shortened_names[away_team_name]
    else:
        away_team_shortened = away_team_name
    scoreline_text = f'{home_team_shortened} {home_goal} - {away_goal} {away_team_shortened}'
    
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


def team_reports(matchdataframe,selected_team_report,passes_home_penalty_area,passes_away_penalty_area,
                passes_away_final_third,passes_home_final_third):
    
    fig = plt.figure(figsize=(20, 20))
    grid = plt.GridSpec(7, 5, figure=fig)
    fig.set_facecolor('white')

    pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black')


    home_plot_ax =fig.add_subplot(grid[1:6, 0:2])
    home_plot_ax.set_facecolor('white')
    away_plot_ax =fig.add_subplot(grid[1:6, 2:4])
    away_plot_ax.set_facecolor('white')
    if selected_team_report == 'Passes':
        home_plot = matchdataframe[matchdataframe['team_id'] == home_team_id]
        away_plot = matchdataframe[matchdataframe['team_id'] == away_team_id]
        home_plot = home_plot[home_plot['type_name'] == 'pass']
        away_plot = away_plot[away_plot['type_name'] == 'pass']
        successful_home_plot = home_plot[home_plot['result_name'] == 'success']
        successful_away_plot = away_plot[away_plot['result_name'] == 'success']
        fail_home_plot = home_plot[home_plot['result_name'] == 'fail']
        fail_away_plot = away_plot[away_plot['result_name'] == 'fail']
        # Create the figure and grid layout
        pitch.draw(ax=home_plot_ax)
        pitch.arrows(successful_home_plot['start_x'], successful_home_plot['start_y'],
                          successful_home_plot['end_x'], successful_home_plot['end_y'],
                          width=2, headwidth=3, label='Successful Passes',
                          color='green', ax=home_plot_ax, alpha=.99)
        pitch.arrows(fail_home_plot['start_x'], fail_home_plot['start_y'],
                          fail_home_plot['end_x'], fail_home_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful Passes',
                          color='red', ax=home_plot_ax, alpha=.99)


        l = home_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")

        pitch.draw(ax=away_plot_ax)
        pitch.arrows(successful_away_plot['start_x'], successful_away_plot['start_y'],
                          successful_away_plot['end_x'], successful_away_plot['end_y'],
                          width=2, headwidth=3, label='Successful Passes',
                          color='green', ax=away_plot_ax, alpha=.99)
        pitch.arrows(fail_away_plot['start_x'], fail_away_plot['start_y'],
                          fail_away_plot['end_x'], fail_away_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful Passes',
                          color='red', ax=away_plot_ax, alpha=.99)


        l = away_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{home_team_name} vs {away_team_name} - {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)




    elif selected_team_report == 'Shot':
        home_plot = matchdataframe[matchdataframe['team_id'] == home_team_id]
        away_plot = matchdataframe[matchdataframe['team_id'] == away_team_id]
        home_plot = home_plot[home_plot['type_name'].isin(['shot', 'shot_penalty'])]
        away_plot = away_plot[away_plot['type_name'].isin(['shot', 'shot_penalty'])]
        successful_home_plot = home_plot[home_plot['result_name'] == 'success']
        successful_away_plot = away_plot[away_plot['result_name'] == 'success']
        fail_home_plot = home_plot[home_plot['result_name'] == 'fail']
        fail_away_plot = away_plot[away_plot['result_name'] == 'fail']
        # Create the figure and grid layout
        pitch.draw(ax=home_plot_ax)
        pitch.arrows(successful_home_plot['start_x'], successful_home_plot['start_y'],
                          successful_home_plot['end_x'], successful_home_plot['end_y'],
                          width=2, headwidth=3, label='Goal',
                          color='green', ax=home_plot_ax, alpha=.99)
        pitch.arrows(fail_home_plot['start_x'], fail_home_plot['start_y'],
                          fail_home_plot['end_x'], fail_home_plot['end_y'],
                          width=2, headwidth=3, label='No Goal',
                          color='red', ax=home_plot_ax, alpha=.99)


        l = home_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")

        pitch.draw(ax=away_plot_ax)
        pitch.arrows(successful_away_plot['start_x'], successful_away_plot['start_y'],
                          successful_away_plot['end_x'], successful_away_plot['end_y'],
                          width=2, headwidth=3, label='Goal',
                          color='green', ax=away_plot_ax, alpha=.99)
        pitch.arrows(fail_away_plot['start_x'], fail_away_plot['start_y'],
                          fail_away_plot['end_x'], fail_away_plot['end_y'],
                          width=2, headwidth=3, label='No Goal',
                          color='red', ax=away_plot_ax, alpha=.99)


        l = away_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{home_team_name} vs {away_team_name} - {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)
    elif selected_team_report == 'Dribbles/Carries':
        home_plot = matchdataframe[matchdataframe['team_id'] == home_team_id]
        away_plot = matchdataframe[matchdataframe['team_id'] == away_team_id]
        home_plot = home_plot[home_plot['type_name'] == 'dribble']
        away_plot = away_plot[away_plot['type_name'] == 'dribble']
        successful_home_plot = home_plot[home_plot['result_name'] == 'success']
        successful_away_plot = away_plot[away_plot['result_name'] == 'success']
        fail_home_plot = home_plot[home_plot['result_name'] == 'fail']
        fail_away_plot = away_plot[away_plot['result_name'] == 'fail']
        # Create the figure and grid layout
        pitch.draw(ax=home_plot_ax)
        pitch.arrows(successful_home_plot['start_x'], successful_home_plot['start_y'],
                          successful_home_plot['end_x'], successful_home_plot['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=home_plot_ax, alpha=.99)
        pitch.arrows(fail_home_plot['start_x'], fail_home_plot['start_y'],
                          fail_home_plot['end_x'], fail_home_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful',
                          color='red', ax=home_plot_ax, alpha=.99)


        l = home_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")

        pitch.draw(ax=away_plot_ax)
        pitch.arrows(successful_away_plot['start_x'], successful_away_plot['start_y'],
                          successful_away_plot['end_x'], successful_away_plot['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=away_plot_ax, alpha=.99)
        pitch.arrows(fail_away_plot['start_x'], fail_away_plot['start_y'],
                          fail_away_plot['end_x'], fail_away_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful',
                          color='red', ax=away_plot_ax, alpha=.99)


        l = away_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{home_team_name} vs {away_team_name} - {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)
    elif selected_team_report == 'Take Ons':
        home_plot = matchdataframe[matchdataframe['team_id'] == home_team_id]
        away_plot = matchdataframe[matchdataframe['team_id'] == away_team_id]
        home_plot = home_plot[home_plot['type_name'] == 'take_on']
        away_plot = away_plot[away_plot['type_name'] == 'take_on']
        successful_home_plot = home_plot[home_plot['result_name'] == 'success']
        successful_away_plot = away_plot[away_plot['result_name'] == 'success']
        fail_home_plot = home_plot[home_plot['result_name'] == 'fail']
        fail_away_plot = away_plot[away_plot['result_name'] == 'fail']
        # Create the figure and grid layout
        pitch.draw(ax=home_plot_ax)
        pitch.arrows(successful_home_plot['start_x'], successful_home_plot['start_y'],
                          successful_home_plot['end_x'], successful_home_plot['end_y'],
                          width=20, headwidth=3, label='Successful',
                          color='green', ax=home_plot_ax, alpha=.99)
        pitch.arrows(fail_home_plot['start_x'], fail_home_plot['start_y'],
                          fail_home_plot['end_x'], fail_home_plot['end_y'],
                          width=20, headwidth=3, label='Unsuccessful',
                          color='red', ax=home_plot_ax, alpha=.99)


        
        pitch.draw(ax=away_plot_ax)
        pitch.arrows(successful_away_plot['start_x'], successful_away_plot['start_y'],
                          successful_away_plot['end_x'], successful_away_plot['end_y'],
                          width=20, headwidth=3, label='Successful',
                          color='green', ax=away_plot_ax, alpha=.99)
        pitch.arrows(fail_away_plot['start_x'], fail_away_plot['start_y'],
                          fail_away_plot['end_x'], fail_away_plot['end_y'],
                          width=20, headwidth=3, label='Unsuccessful',
                          color='red', ax=away_plot_ax, alpha=.99)


        
        plt.suptitle(f'{home_team_name} vs {away_team_name} - {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)
    elif selected_team_report == 'Progressive Actions':
        home_plot = matchdataframe[matchdataframe['team_id'] == home_team_id]
        away_plot = matchdataframe[matchdataframe['team_id'] == away_team_id]
        home_plot = home_plot[(home_plot['type_name'].isin(['pass', 'dribble'])) & (home_plot['progressive'] == True)]
        away_plot = away_plot[(away_plot['type_name'].isin(['pass', 'dribble'])) & (away_plot['progressive'] == True)]
        successful_home_plot = home_plot[home_plot['result_name'] == 'success']
        successful_away_plot = away_plot[away_plot['result_name'] == 'success']
        fail_home_plot = home_plot[home_plot['result_name'] == 'fail']
        fail_away_plot = away_plot[away_plot['result_name'] == 'fail']
        # Create the figure and grid layout
        pitch.draw(ax=home_plot_ax)
        successful_pass_home_plot = successful_home_plot[successful_home_plot['type_name'] == 'pass']
        successful_dribble_home_plot = successful_home_plot[successful_home_plot['type_name'] == 'dribble']

        # Plot successful passes with green color
        pitch.arrows(successful_pass_home_plot['start_x'], successful_pass_home_plot['start_y'],
                     successful_pass_home_plot['end_x'], successful_pass_home_plot['end_y'],
                     width=2, headwidth=3, label='Passes', color='green', ax=home_plot_ax, alpha=0.99)

        # Plot successful dribbles with blue color
        pitch.arrows(successful_dribble_home_plot['start_x'], successful_dribble_home_plot['start_y'],
                     successful_dribble_home_plot['end_x'], successful_dribble_home_plot['end_y'],
                     width=2, headwidth=3, label='Dribbles', color='blue', ax=home_plot_ax, alpha=0.99)



        l = home_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")

        pitch.draw(ax=away_plot_ax)
        # Filter for successful passes and dribbles for away team
        successful_pass_away_plot = successful_away_plot[successful_away_plot['type_name'] == 'pass']
        successful_dribble_away_plot = successful_away_plot[successful_away_plot['type_name'] == 'dribble']

        # Plot successful passes with green color
        pitch.arrows(successful_pass_away_plot['start_x'], successful_pass_away_plot['start_y'],
                     successful_pass_away_plot['end_x'], successful_pass_away_plot['end_y'],
                     width=2, headwidth=3, label='Successful Passes', color='green', ax=away_plot_ax, alpha=0.99)

        # Plot successful dribbles with blue color
        pitch.arrows(successful_dribble_away_plot['start_x'], successful_dribble_away_plot['start_y'],
                     successful_dribble_away_plot['end_x'], successful_dribble_away_plot['end_y'],
                     width=2, headwidth=3, label='Successful Dribbles', color='blue', ax=away_plot_ax, alpha=0.99)



        l = away_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{home_team_name} vs {away_team_name} - Successful {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)
    elif selected_team_report == 'Free Kick':
        
        home_plot = matchdataframe[matchdataframe['team_id'] == home_team_id]
        away_plot = matchdataframe[matchdataframe['team_id'] == away_team_id]
        
        home_plot = home_plot[home_plot['type_name'].isin(['freekick_crossed', 'freekick_short'])]
        away_plot = away_plot[away_plot['type_name'].isin(['freekick_crossed', 'freekick_short'])]
        successful_home_plot = home_plot[home_plot['result_name'] == 'success']
        successful_away_plot = away_plot[away_plot['result_name'] == 'success']
        fail_home_plot = home_plot[home_plot['result_name'] == 'fail']
        fail_away_plot = away_plot[away_plot['result_name'] == 'fail']
        # Create the figure and grid layout
        pitch.draw(ax=home_plot_ax)
        pitch.arrows(successful_home_plot['start_x'], successful_home_plot['start_y'],
                          successful_home_plot['end_x'], successful_home_plot['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=home_plot_ax, alpha=.99)
        pitch.arrows(fail_home_plot['start_x'], fail_home_plot['start_y'],
                          fail_home_plot['end_x'], fail_home_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful',
                          color='red', ax=home_plot_ax, alpha=.99)


        l = home_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")

        pitch.draw(ax=away_plot_ax)
        pitch.arrows(successful_away_plot['start_x'], successful_away_plot['start_y'],
                          successful_away_plot['end_x'], successful_away_plot['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=away_plot_ax, alpha=.99)
        pitch.arrows(fail_away_plot['start_x'], fail_away_plot['start_y'],
                          fail_away_plot['end_x'], fail_away_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful',
                          color='red', ax=away_plot_ax, alpha=.99)


        l = away_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{home_team_name} vs {away_team_name} - {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)
    elif selected_team_report == 'Passes Into Final Third':
        # Create the figure and grid layout
        pitch.draw(ax=home_plot_ax)
        pitch.arrows(passes_home_final_third['start_x'], passes_home_final_third['start_y'],
                          passes_home_final_third['end_x'], passes_home_final_third['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=home_plot_ax, alpha=.99)


        l = home_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")

        pitch.draw(ax=away_plot_ax)
        pitch.arrows(passes_away_final_third['start_x'], passes_away_final_third['start_y'],
                          passes_away_final_third['end_x'], passes_away_final_third['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=away_plot_ax, alpha=.99)


        l = away_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{home_team_name} vs {away_team_name} - {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)

        
    elif selected_team_report == 'Passes Into Penalty Area':
        # Create the figure and grid layout
        pitch.draw(ax=home_plot_ax)
        pitch.arrows(passes_home_penalty_area['start_x'], passes_home_penalty_area['start_y'],
                          passes_home_penalty_area['end_x'], passes_home_penalty_area['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=home_plot_ax, alpha=.99)


        l = home_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")

        pitch.draw(ax=away_plot_ax)
        pitch.arrows(passes_away_penalty_area['start_x'], passes_away_penalty_area['start_y'],
                          passes_away_penalty_area['end_x'], passes_away_penalty_area['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=away_plot_ax, alpha=.99)

        l = away_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{home_team_name} vs {away_team_name} - {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)
                
    elif selected_team_report == 'Defensive Actions':
        home_plot = matchdataframe[matchdataframe['team_id'] == home_team_id]
        away_plot = matchdataframe[matchdataframe['team_id'] == away_team_id]
        home_plot = home_plot[home_plot['type_name'].isin(['tackle', 'interception','clearance','foul'])]
        away_plot = away_plot[away_plot['type_name'].isin(['tackle', 'interception','clearance','foul'])]
        successful_home_plot = home_plot[home_plot['result_name'] == 'success']
        successful_away_plot = away_plot[away_plot['result_name'] == 'success']
        fail_home_plot = home_plot[home_plot['result_name'] == 'fail']
        fail_away_plot = away_plot[away_plot['result_name'] == 'fail']
        pitch.draw(ax=home_plot_ax)
        legend_labels_home = []
        legend_elements_home = []

        # Plot successful and unsuccessful arrows for home team
        for type_name, color, marker in zip(['tackle', 'interception', 'clearance', 'foul'],
                                            ['blue', 'orange', 'purple', 'yellow'],
                                            ['o', '^', 's', 'D']):
            type_success_home_plot = successful_home_plot[successful_home_plot['type_name'] == type_name]
            type_fail_home_plot = fail_home_plot[fail_home_plot['type_name'] == type_name]

            # Plot successful actions with different markers and only show start_x and start_y
            scatter_success = pitch.scatter(type_success_home_plot['start_x'], type_success_home_plot['start_y'],
                  s=100, color='green', marker=marker, label=f'{type_name} Successful', ax=home_plot_ax, alpha=0.99)

            # Plot unsuccessful arrows with different markers and only show start_x and start_y
            scatter_fail = pitch.scatter(type_fail_home_plot['start_x'], type_fail_home_plot['start_y'],
                  s=100, color='red', marker=marker, label=f'{type_name} Unsuccessful', ax=home_plot_ax, alpha=0.99)
            legend_labels_home.append(f'{type_name}')
            legend_elements_home.append((scatter_success, scatter_fail))


        # Add legend for home team
        legend_home = home_plot_ax.legend(legend_elements_home, legend_labels_home, handler_map={tuple: HandlerTuple(ndivide=None)},
                                  loc='lower center', ncol=2, prop={'size': 8},
                                  facecolor='black', edgecolor='#22312b')
        for text in legend_home.get_texts():
            text.set_color("white")

        # Draw pitch and plot arrows for away team
        pitch.draw(ax=away_plot_ax)
        legend_labels_away = []
        legend_elements_away = []

        # Plot successful and unsuccessful arrows for away team
        for type_name, marker in zip(['tackle', 'interception', 'clearance', 'foul'],
                                            ['o', '^', 's', 'D']):
            type_success_away_plot = successful_away_plot[successful_away_plot['type_name'] == type_name]
            type_fail_away_plot = fail_away_plot[fail_away_plot['type_name'] == type_name]

            # Plot successful actions with different markers and only show start_x and start_y
            scatter_success=pitch.scatter(type_success_away_plot['start_x'], type_success_away_plot['start_y'],
                  s=100, color='green', marker=marker, label=f'{type_name} Successful', ax=away_plot_ax, alpha=0.99)

            # Plot unsuccessful arrows with different markers and only show start_x and start_y
            scatter_fail=pitch.scatter(type_fail_away_plot['start_x'], type_fail_away_plot['start_y'],
                  s=100, color='red', marker=marker, label=f'{type_name} Unsuccessful', ax=away_plot_ax, alpha=0.99)
            legend_labels_away.append(f'{type_name}')
            legend_elements_away.append((scatter_success, scatter_fail))


        # Add legend for home team
        legend_away = away_plot_ax.legend(legend_elements_away, legend_labels_away, handler_map={tuple: HandlerTuple(ndivide=None)},
                                  loc='lower center', ncol=2, prop={'size': 8},
                                  facecolor='black', edgecolor='#22312b')
        for text in legend_away.get_texts():
            text.set_color("white")

        plt.suptitle(f'{home_team_name} vs {away_team_name} - {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)
    elif selected_team_report == 'Throw In':
        home_plot = matchdataframe[matchdataframe['team_id'] == home_team_id]
        away_plot = matchdataframe[matchdataframe['team_id'] == away_team_id]
        home_plot = home_plot[home_plot['type_name'] == 'throw_in']
        away_plot = away_plot[away_plot['type_name'] == 'throw_in']
        successful_home_plot = home_plot[home_plot['result_name'] == 'success']
        successful_away_plot = away_plot[away_plot['result_name'] == 'success']
        fail_home_plot = home_plot[home_plot['result_name'] == 'fail']
        fail_away_plot = away_plot[away_plot['result_name'] == 'fail']
        # Create the figure and grid layout
        pitch.draw(ax=home_plot_ax)
        pitch.arrows(successful_home_plot['start_x'], successful_home_plot['start_y'],
                          successful_home_plot['end_x'], successful_home_plot['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=home_plot_ax, alpha=.99)
        pitch.arrows(fail_home_plot['start_x'], fail_home_plot['start_y'],
                          fail_home_plot['end_x'], fail_home_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful',
                          color='red', ax=home_plot_ax, alpha=.99)


        l = home_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")

        pitch.draw(ax=away_plot_ax)
        pitch.arrows(successful_away_plot['start_x'], successful_away_plot['start_y'],
                          successful_away_plot['end_x'], successful_away_plot['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=away_plot_ax, alpha=.99)
        pitch.arrows(fail_away_plot['start_x'], fail_away_plot['start_y'],
                          fail_away_plot['end_x'], fail_away_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful',
                          color='red', ax=away_plot_ax, alpha=.99)


        l = away_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{home_team_name} vs {away_team_name} - {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)
    elif selected_team_report == 'Corner':
        home_plot = matchdataframe[matchdataframe['team_id'] == home_team_id]
        away_plot = matchdataframe[matchdataframe['team_id'] == away_team_id]
        home_plot = home_plot[home_plot['type_name'] == 'corner_crossed']
        away_plot = away_plot[away_plot['type_name'] == 'corner_crossed']
        successful_home_plot = home_plot[home_plot['result_name'] == 'success']
        successful_away_plot = away_plot[away_plot['result_name'] == 'success']
        fail_home_plot = home_plot[home_plot['result_name'] == 'fail']
        fail_away_plot = away_plot[away_plot['result_name'] == 'fail']
        # Create the figure and grid layout
        pitch.draw(ax=home_plot_ax)
        pitch.arrows(successful_home_plot['start_x'], successful_home_plot['start_y'],
                          successful_home_plot['end_x'], successful_home_plot['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=home_plot_ax, alpha=.99)
        pitch.arrows(fail_home_plot['start_x'], fail_home_plot['start_y'],
                          fail_home_plot['end_x'], fail_home_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful',
                          color='red', ax=home_plot_ax, alpha=.99)


        l = home_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")

        pitch.draw(ax=away_plot_ax)
        pitch.arrows(successful_away_plot['start_x'], successful_away_plot['start_y'],
                          successful_away_plot['end_x'], successful_away_plot['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=away_plot_ax, alpha=.99)
        pitch.arrows(fail_away_plot['start_x'], fail_away_plot['start_y'],
                          fail_away_plot['end_x'], fail_away_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful',
                          color='red', ax=away_plot_ax, alpha=.99)


        l = away_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{home_team_name} vs {away_team_name} - {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)
    elif selected_team_report == 'Cross':
        home_plot = matchdataframe[matchdataframe['team_id'] == home_team_id]
        away_plot = matchdataframe[matchdataframe['team_id'] == away_team_id]
        home_plot = home_plot[home_plot['type_name'] == 'cross']
        away_plot = away_plot[away_plot['type_name'] == 'cross']
        successful_home_plot = home_plot[home_plot['result_name'] == 'success']
        successful_away_plot = away_plot[away_plot['result_name'] == 'success']
        fail_home_plot = home_plot[home_plot['result_name'] == 'fail']
        fail_away_plot = away_plot[away_plot['result_name'] == 'fail']
        # Create the figure and grid layout
        pitch.draw(ax=home_plot_ax)
        pitch.arrows(successful_home_plot['start_x'], successful_home_plot['start_y'],
                          successful_home_plot['end_x'], successful_home_plot['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=home_plot_ax, alpha=.99)
        pitch.arrows(fail_home_plot['start_x'], fail_home_plot['start_y'],
                          fail_home_plot['end_x'], fail_home_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful',
                          color='red', ax=home_plot_ax, alpha=.99)


        l = home_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")

        pitch.draw(ax=away_plot_ax)
        pitch.arrows(successful_away_plot['start_x'], successful_away_plot['start_y'],
                          successful_away_plot['end_x'], successful_away_plot['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=away_plot_ax, alpha=.99)
        pitch.arrows(fail_away_plot['start_x'], fail_away_plot['start_y'],
                          fail_away_plot['end_x'], fail_away_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful',
                          color='red', ax=away_plot_ax, alpha=.99)


        l = away_plot_ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 8}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{home_team_name} vs {away_team_name} - {selected_team_report}', fontsize=26, fontweight='bold', color='black',x=0.43)
        plt.subplots_adjust(top=1.35)  # Increase the top margin
        st.pyplot(fig)
def player_reports(matchdataframe,selected_player_report,selected_player_id,selected_player_name,passes_final_third,passes_penalty_area):
    player_plot = matchdataframe[matchdataframe['player_id'] == selected_player_id]
    
    # Assuming you have already defined 'matchdataframe', 'grid', 'VerticalPitch' before this code snippet
    fig, ax = plt.subplots(figsize=(27,16))
    fig.set_facecolor('white')

    pitch = VerticalPitch(pitch_type='opta', pitch_color='white', line_color='black')
    pitch.draw(ax=ax)

    if selected_player_report == 'Passes':
        player_plot = player_plot[player_plot['type_name'] == 'pass']
        successful_player_plot = player_plot[player_plot['result_name'] == 'success']
        fail_player_plot = player_plot[player_plot['result_name'] == 'fail']
        pitch.arrows(successful_player_plot['start_x'], successful_player_plot['start_y'],
                          successful_player_plot['end_x'], successful_player_plot['end_y'],
                          width=2, headwidth=3, label='Successful Passes',
                          color='green', ax=ax, alpha=.99)
        pitch.arrows(fail_player_plot['start_x'], fail_player_plot['start_y'],
                          fail_player_plot['end_x'], fail_player_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful Passes',
                          color='red', ax=ax, alpha=.99)


        l = ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 15}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{selected_player_name} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)




    elif selected_player_report == 'Shot':
        player_plot = player_plot[player_plot['type_name'] == 'shot']
        successful_player_plot = player_plot[player_plot['result_name'] == 'success']
        fail_player_plot = player_plot[player_plot['result_name'] == 'fail']
        pitch.arrows(successful_player_plot['start_x'], successful_player_plot['start_y'],
                          successful_player_plot['end_x'], successful_player_plot['end_y'],
                          width=2, headwidth=3, label='Goal',
                          color='green', ax=ax, alpha=.99)
        pitch.arrows(fail_player_plot['start_x'], fail_player_plot['start_y'],
                          fail_player_plot['end_x'], fail_player_plot['end_y'],
                          width=2, headwidth=3, label='No Goal',
                          color='red', ax=ax, alpha=.99)


        l = ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 15}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{selected_player_id} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)
    elif selected_player_report == 'Dribbles/Carries':
        player_plot = player_plot[player_plot['type_name'] == 'dribble']
        successful_player_plot = player_plot[player_plot['result_name'] == 'success']
        fail_player_plot = player_plot[player_plot['result_name'] == 'fail']
        pitch.arrows(successful_player_plot['start_x'], successful_player_plot['start_y'],
                          successful_player_plot['end_x'], successful_player_plot['end_y'],
                          width=2, headwidth=3, label='Successful',
                          color='green', ax=ax, alpha=.99)

        plt.suptitle(f'{selected_player_name} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)
    elif selected_player_report == 'Take Ons':
        player_plot = player_plot[player_plot['type_name'] == 'take_on']
        successful_player_plot = player_plot[player_plot['result_name'] == 'success']
        fail_player_plot = player_plot[player_plot['result_name'] == 'fail']
        pitch.arrows(successful_player_plot['start_x'], successful_player_plot['start_y'],
                          successful_player_plot['end_x'], successful_player_plot['end_y'],
                          width=25, headwidth=3, label='Successful Take On',
                          color='green', ax=ax, alpha=.99)
        pitch.arrows(fail_player_plot['start_x'], fail_player_plot['start_y'],
                          fail_player_plot['end_x'], fail_player_plot['end_y'],
                          width=25, headwidth=3, label='Unsuccessful Take On',
                          color='red', ax=ax, alpha=.99)


       
        plt.suptitle(f'{selected_player_name} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)
    elif selected_player_report == 'Progressive Actions':
        player_plot = player_plot[(player_plot['type_name'].isin(['pass', 'dribble'])) & (player_plot['progressive'] == True)]
        print(player_plot)
        successful_player_plot = player_plot[player_plot['result_name'] == 'success']
        fail_player_plot = player_plot[player_plot['result_name'] == 'fail']
        successful_pass_player_plot = successful_player_plot[successful_player_plot['type_name'] == 'pass']
        successful_dribble_player_plot = successful_player_plot[successful_player_plot['type_name'] == 'dribble']
        pitch.arrows(successful_pass_player_plot['start_x'], successful_pass_player_plot['start_y'],
                          successful_pass_player_plot['end_x'], successful_pass_player_plot['end_y'],
                          width=2, headwidth=3, label='Pass',
                          color='green', ax=ax, alpha=.99)
        pitch.arrows(successful_dribble_player_plot['start_x'], successful_dribble_player_plot['start_y'],
                          successful_dribble_player_plot['end_x'], successful_dribble_player_plot['end_y'],
                          width=2, headwidth=3, label='Carry',
                          color='blue', ax=ax, alpha=.99)


        l = ax.legend(shadow=True, loc='lower center', ncol=2, prop={'size': 15}, facecolor='black', edgecolor='#22312b')
        for text in l.get_texts():
            text.set_color("white")
        plt.suptitle(f'{selected_player_name} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)
    elif selected_player_report == 'Free Kick':
        player_plot = player_plot[player_plot['type_name'].isin(['freekick_short', 'freekick_crossed'])]
        successful_player_plot = player_plot[player_plot['result_name'] == 'success']
        fail_player_plot = player_plot[player_plot['result_name'] == 'fail']
        pitch.arrows(successful_player_plot['start_x'], successful_player_plot['start_y'],
                          successful_player_plot['end_x'], successful_player_plot['end_y'],
                          width=2, headwidth=3, label='Successful Freekick',
                          color='green', ax=ax, alpha=.99)
        pitch.arrows(fail_player_plot['start_x'], fail_player_plot['start_y'],
                          fail_player_plot['end_x'], fail_player_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful Freekick',
                          color='red', ax=ax, alpha=.99)


       
        plt.suptitle(f'{selected_player_name} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)
    elif selected_player_report == 'Passes Into Final Third':
        player_plot = passes_final_third[passes_final_third['player_id'] == selected_player_id]
        pitch.arrows(player_plot['start_x'], player_plot['start_y'],
                          player_plot['end_x'], player_plot['end_y'],
                          width=2, headwidth=3, label='Successful Passes',
                          color='green', ax=ax, alpha=.99)
       
        plt.suptitle(f'{selected_player_name} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)

        
    elif selected_player_report == 'Passes Into Penalty Area':
        player_plot = passes_penalty_area[passes_penalty_area['player_id'] == selected_player_id]
        pitch.arrows(player_plot['start_x'], player_plot['start_y'],
                          player_plot['end_x'], player_plot['end_y'],
                          width=2, headwidth=3, label='Successful Passes',
                          color='green', ax=ax, alpha=.99)
       
        plt.suptitle(f'{selected_player_name} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)

        
                
    elif selected_player_report == 'Defensive Actions':
        player_plot = player_plot[player_plot['type_name'].isin(['tackle', 'interception','clearance','foul'])]
        successful_player_plot = player_plot[player_plot['result_name'] == 'success']
        fail_player_plot = player_plot[player_plot['result_name'] == 'fail']
        legend_labels = []
        legend_elements = []

        # Plot successful and unsuccessful arrows for home team
        for type_name, color, marker in zip(['tackle', 'interception', 'clearance', 'foul'],
                                            ['blue', 'orange', 'purple', 'yellow'],
                                            ['o', '^', 's', 'D']):
            type_success_home_plot = successful_player_plot[successful_player_plot['type_name'] == type_name]
            type_fail_home_plot = fail_player_plot[fail_player_plot['type_name'] == type_name]

            # Plot successful actions with different markers and only show start_x and start_y
            scatter_success = pitch.scatter(type_success_home_plot['start_x'], type_success_home_plot['start_y'],
                  s=100, color='green', marker=marker, label=f'{type_name} Successful', ax=ax, alpha=0.99)

            # Plot unsuccessful arrows with different markers and only show start_x and start_y
            scatter_fail = pitch.scatter(type_fail_home_plot['start_x'], type_fail_home_plot['start_y'],
                  s=100, color='red', marker=marker, label=f'{type_name} Unsuccessful', ax=ax, alpha=0.99)
            legend_labels.append(f'{type_name}')
            legend_elements.append((scatter_success, scatter_fail))


        # Add legend for home team
        ax = ax.legend(legend_elements, legend_labels, handler_map={tuple: HandlerTuple(ndivide=None)},
                                  loc='lower center', ncol=2, prop={'size': 8},
                                  facecolor='black', edgecolor='#22312b')
        for text in ax.get_texts():
            text.set_color("white")
        plt.suptitle(f'{selected_player_name} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)
    elif selected_player_report == 'Throw In':
        player_plot = player_plot[player_plot['type_name'] == 'throw_in']
        successful_player_plot = player_plot[player_plot['result_name'] == 'success']
        fail_player_plot = player_plot[player_plot['result_name'] == 'fail']
        pitch.arrows(successful_player_plot['start_x'], successful_player_plot['start_y'],
                          successful_player_plot['end_x'], successful_player_plot['end_y'],
                          width=2, headwidth=3, label='Successful Throw In',
                          color='green', ax=ax, alpha=.99)
        pitch.arrows(fail_player_plot['start_x'], fail_player_plot['start_y'],
                          fail_player_plot['end_x'], fail_player_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful Trhow In',
                          color='red', ax=ax, alpha=.99)


       
        plt.suptitle(f'{selected_player_name} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)
    elif selected_player_report == 'Corner':
        player_plot = player_plot[player_plot['type_name'] == 'corner']
        successful_player_plot = player_plot[player_plot['result_name'] == 'success']
        fail_player_plot = player_plot[player_plot['result_name'] == 'fail']
        pitch.arrows(successful_player_plot['start_x'], successful_player_plot['start_y'],
                          successful_player_plot['end_x'], successful_player_plot['end_y'],
                          width=2, headwidth=3, label='Successful Corner',
                          color='green', ax=ax, alpha=.99)
        pitch.arrows(fail_player_plot['start_x'], fail_player_plot['start_y'],
                          fail_player_plot['end_x'], fail_player_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful Corner',
                          color='red', ax=ax, alpha=.99)


       
        plt.suptitle(f'{selected_player_name} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)
    elif selected_player_report == 'Cross':
        player_plot = player_plot[player_plot['type_name'] == 'cross']
        successful_player_plot = player_plot[player_plot['result_name'] == 'success']
        fail_player_plot = player_plot[player_plot['result_name'] == 'fail']
        pitch.arrows(successful_player_plot['start_x'], successful_player_plot['start_y'],
                          successful_player_plot['end_x'], successful_player_plot['end_y'],
                          width=2, headwidth=3, label='Successful Cross',
                          color='green', ax=ax, alpha=.99)
        pitch.arrows(fail_player_plot['start_x'], fail_player_plot['start_y'],
                          fail_player_plot['end_x'], fail_player_plot['end_y'],
                          width=2, headwidth=3, label='Unsuccessful Cross',
                          color='red', ax=ax, alpha=.99)


       
        plt.suptitle(f'{selected_player_name} - {selected_player_report}', fontsize=26, fontweight='bold', color='black',x=0.52)
        plt.subplots_adjust(top=0.96)  # Increase the top margin
        st.pyplot(fig)

if report_type == 'Team Report':
    if selected_team_report == 'General Report':
        general_report(home_passes_between_df,home_average_locs_and_count_df,away_passes_between_df,away_average_locs_and_count_df,
                  passes_home_final_third,passes_away_final_third,passes_away_penalty_area,passes_home_penalty_area,goal_rows,
                  home_team_goal_count,away_team_goal_count,home_team_name,away_team_name)
    else:
        team_reports(matchdataframe,selected_team_report,passes_home_penalty_area,passes_away_penalty_area,
                passes_away_final_third,passes_home_final_third)
elif report_type =='Player Report':
    passes_penalty_area = pd.concat([passes_away_penalty_area, passes_home_penalty_area], ignore_index=True)
    passes_final_third = pd.concat([passes_away_final_third, passes_home_final_third], ignore_index=True)
    player_reports(matchdataframe,selected_player_report,selected_player_id,selected_player_name,passes_final_third,passes_penalty_area)

