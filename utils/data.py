"""
Optimized Data Loading Functions
Uses SQL views and RPC functions for efficient queries
"""

import streamlit as st
import pandas as pd
import numpy as np
from utils.supabase import get_supabase


# =============================================================================
# LIGHTWEIGHT QUERIES (cached)
# =============================================================================

@st.cache_data(ttl=3600)
def load_matches_list() -> pd.DataFrame:
    """Load lightweight matches list for dropdowns"""
    supabase = get_supabase()
    try:
        # Supabase-py requires params dict even if empty
        response = supabase.rpc('get_matches_list', {}).execute()
        if response.data:
            return pd.DataFrame(response.data)
    except Exception as e:
        st.warning(f"RPC not available, using fallback: {e}")
        return _load_matches_fallback()
    return pd.DataFrame()


def _load_matches_fallback() -> pd.DataFrame:
    """Fallback if RPC not set up yet"""
    supabase = get_supabase()
    # Get unique game_ids with team info (minimal query)
    response = supabase.table('defined_actions').select(
        'game_id, team_id'
    ).execute()

    if not response.data:
        return pd.DataFrame()

    df = pd.DataFrame(response.data)
    game_ids = df['game_id'].unique()

    # Get team names
    teams_response = supabase.table('teams').select('team_id, team_name').execute()
    teams = {t['team_id']: t['team_name'] for t in teams_response.data}

    matches = []
    for gid in game_ids:
        team_ids = df[df['game_id'] == gid]['team_id'].unique()[:2]
        team_names = [teams.get(tid, f"Team {tid}") for tid in team_ids]
        if len(team_names) >= 2:
            matches.append({
                'game_id': gid,
                'home_team': team_names[0],
                'away_team': team_names[1],
                'match_label': f"{team_names[0]} vs {team_names[1]}"
            })
    return pd.DataFrame(matches)


# =============================================================================
# SINGLE MATCH LOADING (for Match Analysis page)
# =============================================================================

@st.cache_data(ttl=600)
def load_match_events(game_id: int) -> pd.DataFrame:
    """Load events for a single match - uses direct query with pagination to avoid row limits"""
    # Use direct query instead of RPC to handle pagination properly
    return _load_match_fallback(game_id)


def _load_match_fallback(game_id: int) -> pd.DataFrame:
    """Fallback match loading if RPC not available"""
    supabase = get_supabase()

    try:
        # Load only this match's actions with pagination
        # Use smaller batch size to ensure we get all data
        data = []
        last_action_id = -1
        batch_size = 500
        max_iterations = 20  # Safety limit

        for i in range(max_iterations):
            response = (
                supabase
                .table('defined_actions')
                .select('*')
                .eq('game_id', game_id)
                .gt('action_id', last_action_id)
                .order('action_id')
                .limit(batch_size)
                .execute()
            )
            if not response.data:
                break
            data.extend(response.data)
            last_action_id = response.data[-1]['action_id']
            if len(response.data) < batch_size:
                break

        if not data:
            st.warning(f"No events found for match {game_id}")
            return pd.DataFrame()

        df = pd.DataFrame(data)

        # Get team and player names
        team_ids = df['team_id'].unique().tolist()
        player_ids = df['player_id'].dropna().unique().tolist()

        teams_resp = supabase.table('teams').select('team_id, team_name').in_('team_id', team_ids).execute()
        teams_map = {t['team_id']: t['team_name'] for t in teams_resp.data} if teams_resp.data else {}

        if player_ids:
            players_resp = supabase.table('players').select('player_id, player_name').in_('player_id', player_ids).execute()
            players_map = {p['player_id']: p['player_name'] for p in players_resp.data} if players_resp.data else {}
        else:
            players_map = {}

        df['team_name'] = df['team_id'].map(teams_map).fillna('Unknown Team')
        df['player_name'] = df['player_id'].map(players_map).fillna('Unknown')

        # Handle column name variations
        if 'xt' in df.columns:
            df = df.rename(columns={'xt': 'xT'})

        # Add computed columns with safety checks
        if 'time_seconds_overall' in df.columns:
            df['minute'] = (df['time_seconds_overall'].fillna(0) / 60).astype(int)
            df['time_display'] = (
                df['minute'].astype(str) + "'" +
                ((df['time_seconds_overall'].fillna(0) % 60).astype(int)).astype(str).str.zfill(2)
            )
        else:
            df['minute'] = 0
            df['time_display'] = "0'00"

        return df

    except Exception as e:
        st.error(f"Failed to load match data: {str(e)[:200]}")
        return pd.DataFrame()


# =============================================================================
# SEASON STATS (pre-aggregated from database)
# =============================================================================

@st.cache_data(ttl=600)
def load_player_season_stats(min_actions: int = 10) -> pd.DataFrame:
    """Load pre-aggregated player season stats from database"""
    supabase = get_supabase()

    try:
        response = supabase.rpc('get_player_season_stats', {'min_actions': min_actions}).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            # Handle column name variations
            rename_map = {'xt_total': 'xT', 'final_third_passes': 'final_third', 'box_passes': 'box_entries'}
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            # Calculate per 90 values
            df['nineties'] = df['minutes'] / 90
            df['nineties'] = df['nineties'].replace(0, 0.01)  # Avoid division by zero
            return df
    except Exception as e:
        st.warning(f"Player stats RPC not available: {e}")
        return pd.DataFrame()

    return pd.DataFrame()


@st.cache_data(ttl=600)
def load_team_season_stats() -> pd.DataFrame:
    """Load pre-aggregated team season stats from database"""
    supabase = get_supabase()

    try:
        response = supabase.rpc('get_team_season_stats', {}).execute()
        if response.data:
            df = pd.DataFrame(response.data)
            # Handle column name variations
            if 'xt_total' in df.columns:
                df = df.rename(columns={'xt_total': 'xT'})
            # Calculate per game values
            df['shots_per_game'] = (df['shots'] / df['games']).round(1)
            df['prog_passes_per_game'] = (df['prog_passes'] / df['games']).round(1)
            df['prog_carries_per_game'] = (df['prog_carries'] / df['games']).round(1)
            df['final_third_per_game'] = (df['final_third_entries'] / df['games']).round(1)
            df['box_entries_per_game'] = (df['box_entries'] / df['games']).round(1)
            df['high_press_per_game'] = (df['high_press_actions'] / df['games']).round(1)
            df['xT_per_game'] = (df['xT'] / df['games']).round(2)
            return df
    except Exception as e:
        st.warning(f"Team stats RPC not available: {e}")
        return pd.DataFrame()

    return pd.DataFrame()


@st.cache_data(ttl=600)
def load_team_ppda() -> pd.DataFrame:
    """Load PPDA stats from database"""
    supabase = get_supabase()

    try:
        response = supabase.rpc('get_team_ppda', {}).execute()
        if response.data:
            return pd.DataFrame(response.data)
    except Exception as e:
        st.warning(f"PPDA RPC not available: {e}")
        return pd.DataFrame()

    return pd.DataFrame()


# =============================================================================
# LEGACY SUPPORT (for backwards compatibility during transition)
# =============================================================================

@st.cache_data(ttl=600)
def load_data():
    """
    Legacy function - loads all data.
    DEPRECATED: Use load_match_events() for match analysis
    and load_player_season_stats() for season stats.
    """
    matches_df = load_matches_list()
    matches = {row['game_id']: row['match_label'] for _, row in matches_df.iterrows()}

    # Return empty actions DataFrame - individual matches should be loaded on demand
    return pd.DataFrame(), matches


def get_available_teams() -> list:
    """Get list of team names"""
    supabase = get_supabase()
    response = supabase.table('teams').select('team_name').execute()
    if response.data:
        return sorted(list(set(t['team_name'] for t in response.data)))
    return []


# =============================================================================
# UTILITY FUNCTIONS (operate on already-loaded data)
# =============================================================================

def flip_coords(df: pd.DataFrame, away_team: str) -> pd.DataFrame:
    """Flip coordinates for away team to show both teams attacking right"""
    df = df.copy()
    mask = df['team_name'] == away_team
    df.loc[mask, 'start_x'] = 100 - df.loc[mask, 'start_x']
    df.loc[mask, 'end_x'] = 100 - df.loc[mask, 'end_x']
    df.loc[mask, 'start_y'] = 100 - df.loc[mask, 'start_y']
    df.loc[mask, 'end_y'] = 100 - df.loc[mask, 'end_y']
    return df


def is_progressive_advanced(row) -> bool:
    """Check if action is progressive (for local filtering only)"""
    if pd.isna(row.get('end_x')) or pd.isna(row.get('start_x')):
        return False
    if row.get('result_name') != 'success':
        return False

    start_x, end_x = row['start_x'], row['end_x']

    if start_x <= 50 and end_x <= 50 and (end_x - start_x) >= 30:
        return True
    if start_x <= 50 and end_x > 50 and (end_x - start_x) >= 15:
        return True
    if start_x > 50 and end_x > 50 and (end_x - start_x) >= 10:
        return True

    return False


def estimate_player_minutes(actions, player_id):
    """Estimate total minutes played based on action timestamps"""
    p_actions = actions[actions['player_id'] == player_id]
    if len(p_actions) == 0:
        return 0

    total_minutes = 0
    for game_id in p_actions['game_id'].unique():
        game_actions = p_actions[p_actions['game_id'] == game_id]
        first_action = game_actions['time_seconds_overall'].min()
        last_action = game_actions['time_seconds_overall'].max()
        start_min = max(0, (first_action / 60) - 3)
        end_min = min(95, (last_action / 60) + 3)
        total_minutes += (end_min - start_min)

    return total_minutes
