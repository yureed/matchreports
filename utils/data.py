"""
Data Loading and Processing Functions
"""

import streamlit as st
import pandas as pd
import numpy as np
from .constants import ARROW_EVENTS, CARRY_EVENTS


@st.cache_data(ttl=600)
def load_table(table_name: str) -> pd.DataFrame:
    supabase = get_supabase()

    data = []
    offset = 0
    limit = 1000

    while True:
        response = (
            supabase
            .table(table_name)
            .select("*")
            .range(offset, offset + limit - 1)
            .execute()
        )

        if not response.data:
            break

        data.extend(response.data)
        offset += limit

    return pd.DataFrame(data)


@st.cache_data(ttl=600)
def load_data():
    actions = load_table("defined_actions")
    players = load_table("players")
    teams = load_table("teams")

    teams = teams.drop_duplicates(subset=["team_id"])

    actions = actions.merge(
        teams[["team_id", "team_name"]],
        on="team_id",
        how="left"
    )

    actions = actions.merge(
        players[["player_id", "player_name"]]
        .drop_duplicates(subset=["player_id"]),
        on="player_id",
        how="left"
    )

    actions["player_name"] = actions["player_name"].fillna("Unknown")

    actions["minute"] = (actions["time_seconds_overall"] / 60).astype(int)
    actions["time_display"] = (
        actions["minute"].astype(str)
        + "'"
        + ((actions["time_seconds_overall"] % 60).astype(int))
        .astype(str).str.zfill(2)
    )

    matches = {}
    for gid in actions["game_id"].unique():
        gdf = actions[actions["game_id"] == gid]
        tms = gdf["team_name"].dropna().unique()
        matches[gid] = (
            f"{tms[0]} vs {tms[1]}" if len(tms) >= 2 else f"Match {gid}"
        )

    return actions, matches

def flip_coords(df, away_team):
    """Flip coordinates for away team to show both teams attacking right"""
    df = df.copy()
    mask = df['team_name'] == away_team
    df.loc[mask, 'start_x'] = 100 - df.loc[mask, 'start_x']
    df.loc[mask, 'end_x'] = 100 - df.loc[mask, 'end_x']
    df.loc[mask, 'start_y'] = 100 - df.loc[mask, 'start_y']
    df.loc[mask, 'end_y'] = 100 - df.loc[mask, 'end_y']
    return df


def is_progressive_advanced(row):
    """
    Advanced progressive action detection.
    A pass/carry is progressive if it moves the ball significantly towards goal.
    """
    if pd.isna(row['end_x']) or pd.isna(row['start_x']):
        return False
    if row['result_name'] != 'success':
        return False

    start_x, end_x = row['start_x'], row['end_x']

    # Progressive thresholds based on pitch zone
    if start_x <= 50 and end_x <= 50 and (end_x - start_x) >= 30:
        return True
    if start_x <= 50 and end_x > 50 and (end_x - start_x) >= 15:
        return True
    if start_x > 50 and end_x > 50 and (end_x - start_x) >= 10:
        return True

    return False


def is_carry(row):
    """Check if action is a carry (dribble/take_on)"""
    return row['type_name'] in CARRY_EVENTS


def estimate_player_minutes(actions, player_id):
    """
    Estimate total minutes played based on action timestamps.
    Uses first and last action times with buffer.
    """
    p_actions = actions[actions['player_id'] == player_id]
    if len(p_actions) == 0:
        return 0

    total_minutes = 0
    for game_id in p_actions['game_id'].unique():
        game_actions = p_actions[p_actions['game_id'] == game_id]
        first_action = game_actions['time_seconds_overall'].min()
        last_action = game_actions['time_seconds_overall'].max()
        # Add buffer before first and after last action
        start_min = max(0, (first_action / 60) - 3)
        end_min = min(95, (last_action / 60) + 3)
        total_minutes += (end_min - start_min)

    return total_minutes


def get_final_third_entries(df, team, home_team):
    """Get passes and carries into final third"""
    team_df = df[df['team_name'] == team]
    is_home = team == home_team

    # For home team attacking right, final third is x > 66.67
    # For away team (coords flipped), also x > 66.67
    passes = team_df[
        (team_df['type_name'].isin(ARROW_EVENTS)) &
        (team_df['start_x'] <= 66.67) &
        (team_df['end_x'] > 66.67) &
        (team_df['result_name'] == 'success')
    ]

    carries = team_df[
        (team_df['type_name'].isin(CARRY_EVENTS)) &
        (team_df['start_x'] <= 66.67) &
        (team_df['end_x'] > 66.67) &
        (team_df['result_name'] == 'success')
    ]

    return passes, carries


def get_box_entries(df, team, home_team):
    """Get passes and carries into penalty box"""
    team_df = df[df['team_name'] == team]

    # Box is x > 83, y between 21-79 (Opta coords)
    passes = team_df[
        (team_df['type_name'].isin(ARROW_EVENTS)) &
        (team_df['start_x'] <= 83) &
        (team_df['end_x'] > 83) &
        (team_df['end_y'] >= 21) & (team_df['end_y'] <= 79) &
        (team_df['result_name'] == 'success')
    ]

    carries = team_df[
        (team_df['type_name'].isin(CARRY_EVENTS)) &
        (team_df['start_x'] <= 83) &
        (team_df['end_x'] > 83) &
        (team_df['end_y'] >= 21) & (team_df['end_y'] <= 79) &
        (team_df['result_name'] == 'success')
    ]

    return passes, carries
