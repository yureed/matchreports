"""
Statistics Calculation Functions
"""

import pandas as pd
import numpy as np
from .constants import SHOT_EVENTS, ARROW_EVENTS, CARRY_EVENTS, DEFENSIVE_EVENTS
from .data import is_progressive_advanced, estimate_player_minutes


def compute_match_stats(df, home, away, home_id=None, away_id=None):
    """Compute match statistics for scoreboard display"""
    h = df[df['team_name'] == home]
    a = df[df['team_name'] == away]

    # Shots
    shots_h = len(h[h['type_name'].isin(SHOT_EVENTS)])
    shots_a = len(a[a['type_name'].isin(SHOT_EVENTS)])

    # Goals (including own goals)
    goals_h = len(h[h['type_name'].isin(SHOT_EVENTS) & (h['goal_from_shot'] == True)])
    goals_a = len(a[a['type_name'].isin(SHOT_EVENTS) & (a['goal_from_shot'] == True)])

    # Own goals
    if home_id and away_id:
        og_for_home = len(df[(df['result_name'] == 'owngoal') & (df['team_id'] == away_id)])
        og_for_away = len(df[(df['result_name'] == 'owngoal') & (df['team_id'] == home_id)])
        goals_h += og_for_home
        goals_a += og_for_away

    # Pass accuracy
    pass_h = h[h['type_name'] == 'pass']
    pass_a = a[a['type_name'] == 'pass']
    pass_pct_h = (pass_h['result_name'] == 'success').mean() * 100 if len(pass_h) > 0 else 0
    pass_pct_a = (pass_a['result_name'] == 'success').mean() * 100 if len(pass_a) > 0 else 0

    return {
        'shots_h': shots_h, 'shots_a': shots_a,
        'goals_h': goals_h, 'goals_a': goals_a,
        'pass_pct_h': pass_pct_h, 'pass_pct_a': pass_pct_a,
    }


def calculate_ppda(actions, team):
    """
    Calculate Passes Per Defensive Action (PPDA).
    Lower = more intense pressing.
    """
    team_games = actions[actions['team_name'] == team]['game_id'].unique()

    total_opp_passes = 0
    total_def_actions = 0

    for game_id in team_games:
        game_df = actions[actions['game_id'] == game_id]
        teams_in_game = game_df['team_name'].unique()
        opponent = [t for t in teams_in_game if t != team]
        if not opponent:
            continue
        opponent = opponent[0]

        # Opponent passes in their own defensive 60%
        opp_passes = game_df[
            (game_df['team_name'] == opponent) &
            (game_df['type_name'] == 'pass') &
            (game_df['start_x'] < 60)
        ]
        total_opp_passes += len(opp_passes)

        # Team's defensive actions in opponent's defensive 60%
        def_actions = game_df[
            (game_df['team_name'] == team) &
            (game_df['type_name'].isin(DEFENSIVE_EVENTS)) &
            (game_df['start_x'] > 40)
        ]
        total_def_actions += len(def_actions)

    if total_def_actions == 0:
        return 0
    return total_opp_passes / total_def_actions


def calculate_team_stats(actions, team):
    """Calculate comprehensive team statistics"""
    team_df = actions[actions['team_name'] == team]
    games = team_df['game_id'].nunique()

    if games == 0:
        return None

    # Basic counts
    passes = team_df[team_df['type_name'] == 'pass']
    shots = team_df[team_df['type_name'].isin(SHOT_EVENTS)]
    goals = len(shots[shots['goal_from_shot'] == True])

    # Progressive actions
    prog_passes = team_df[
        (team_df['type_name'].isin(ARROW_EVENTS)) &
        (team_df.apply(is_progressive_advanced, axis=1))
    ]
    prog_carries = team_df[
        (team_df['type_name'].isin(CARRY_EVENTS)) &
        (team_df.apply(is_progressive_advanced, axis=1))
    ]

    # Final third entries
    final_third_passes = passes[
        (passes['end_x'] > 66.67) &
        (passes['start_x'] <= 66.67) &
        (passes['result_name'] == 'success')
    ]

    # Box entries
    box_entries = passes[
        (passes['end_x'] > 83) &
        (passes['end_y'] >= 21) & (passes['end_y'] <= 79) &
        (passes['start_x'] <= 83) &
        (passes['result_name'] == 'success')
    ]

    # Defensive stats
    tackles = team_df[team_df['type_name'] == 'tackle']
    interceptions = team_df[team_df['type_name'] == 'interception']
    high_press = team_df[
        (team_df['type_name'].isin(DEFENSIVE_EVENTS)) &
        (team_df['start_x'] > 66.67)
    ]

    return {
        'team': team,
        'games': games,
        'goals': goals,
        'shots': len(shots),
        'shots_per_game': len(shots) / games,
        'pass_pct': (passes['result_name'] == 'success').mean() * 100 if len(passes) > 0 else 0,
        'prog_passes': len(prog_passes),
        'prog_passes_per_game': len(prog_passes) / games,
        'prog_carries': len(prog_carries),
        'prog_carries_per_game': len(prog_carries) / games,
        'final_third_entries': len(final_third_passes),
        'final_third_per_game': len(final_third_passes) / games,
        'box_entries': len(box_entries),
        'box_entries_per_game': len(box_entries) / games,
        'tackles': len(tackles),
        'interceptions': len(interceptions),
        'high_press_actions': len(high_press),
        'high_press_per_game': len(high_press) / games,
    }


def calculate_player_stats(actions, min_minutes=0):
    """Calculate comprehensive player statistics with minutes played"""
    player_stats = []

    for player_id in actions['player_id'].dropna().unique():
        p = actions[actions['player_id'] == player_id]
        name = p['player_name'].iloc[0]
        team = p['team_name'].iloc[0]
        games = p['game_id'].nunique()

        # Estimate minutes
        minutes = estimate_player_minutes(actions, player_id)
        if minutes < min_minutes:
            continue

        nineties = minutes / 90 if minutes > 0 else 0.01

        # Passes
        passes = p[p['type_name'] == 'pass']
        pass_success = len(passes[passes['result_name'] == 'success'])

        # Progressive
        prog_passes = len(p[
            (p['type_name'].isin(ARROW_EVENTS)) &
            (p.apply(is_progressive_advanced, axis=1))
        ])
        prog_carries = len(p[
            (p['type_name'].isin(CARRY_EVENTS)) &
            (p.apply(is_progressive_advanced, axis=1))
        ])

        # Shots & Goals
        shots_df = p[p['type_name'].isin(SHOT_EVENTS)]
        goals = len(shots_df[shots_df['goal_from_shot'] == True])

        # Defensive
        tackles = len(p[p['type_name'] == 'tackle'])
        interceptions = len(p[p['type_name'] == 'interception'])
        clearances = len(p[p['type_name'] == 'clearance'])

        # Dribbles & Crosses
        dribbles = p[p['type_name'].isin(['dribble', 'take_on'])]
        crosses = p[p['type_name'] == 'cross']

        # Final third passes
        ft_passes = len(passes[
            (passes['end_x'] > 66.67) &
            (passes['start_x'] <= 66.67) &
            (passes['result_name'] == 'success')
        ])

        # Final third carries
        ft_carries = len(p[
            (p['type_name'].isin(CARRY_EVENTS)) &
            (p['start_x'] <= 66.67) &
            (p['end_x'] > 66.67) &
            (p['result_name'] == 'success')
        ])

        # Box entries (passes)
        box_passes = len(passes[
            (passes['end_x'] > 83) &
            (passes['end_y'] >= 21) & (passes['end_y'] <= 79) &
            (passes['start_x'] <= 83) &
            (passes['result_name'] == 'success')
        ])

        # Box entries (carries)
        box_carries = len(p[
            (p['type_name'].isin(CARRY_EVENTS)) &
            (p['start_x'] <= 83) &
            (p['end_x'] > 83) &
            (p['end_y'] >= 21) & (p['end_y'] <= 79) &
            (p['result_name'] == 'success')
        ])

        player_stats.append({
            'player_id': player_id,
            'player': name,
            'team': team,
            'games': games,
            'minutes': round(minutes),
            'nineties': round(nineties, 2),
            'goals': goals,
            'shots': len(shots_df),
            'passes': len(passes),
            'pass_success': pass_success,
            'pass_pct': (pass_success / len(passes) * 100) if len(passes) > 0 else 0,
            'prog_passes': prog_passes,
            'prog_carries': prog_carries,
            'tackles': tackles,
            'interceptions': interceptions,
            'clearances': clearances,
            'dribbles': len(dribbles),
            'dribble_success': len(dribbles[dribbles['result_name'] == 'success']),
            'crosses': len(crosses),
            'final_third_passes': ft_passes,
            'final_third_carries': ft_carries,
            'box_passes': box_passes,
            'box_carries': box_carries,
            'defensive': tackles + interceptions + clearances,
        })

    return pd.DataFrame(player_stats)
