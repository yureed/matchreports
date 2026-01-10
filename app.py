"""
Matchday Analytics - Football Event Visualization Platform
Professional match analysis for journalists and analysts.
"""

import streamlit as st
import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from io import BytesIO
import os

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="Match Analysis - Matchday Analytics",
    page_icon="M",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SIMPLE CLEAN CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    /* Hide footer only */
    footer {display: none;}

    /* Dark theme */
    .stApp {
        background: #0a0a0f;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #111118;
    }

    /* Match Header */
    .match-header-card {
        background: #16161d;
        border: 3px solid #2a2a35;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin-bottom: 1rem;
    }

    .match-teams {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 2rem;
    }

    .team-info { flex: 1; max-width: 1000px; }
    .team-info.home { text-align: right; }
    .team-info.away { text-align: left; }

    .team-name-display {
        font-family: 'Inter', sans-serif;
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0;
    }
    .team-name-display.home { color: #f87171; }
    .team-name-display.away { color: #60a5fa; }

    .score-center {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.5rem 1rem;
        background: #0a0a0f;
        border-radius: 8px;
    }

    .score-num {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.5rem;
        font-weight: 600;
        color: #fff;
    }

    .score-divider { color: #444; font-size: 1.5rem; }

    .match-meta {
        text-align: center;
        margin-top: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        color: #666;
    }

    .stats-row {
        display: flex;
        gap: 0.5rem;
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #2a2a35;
    }

    .stat-box {
        flex: 1;
        background: #1a1a22;
        border-radius: 6px;
        padding: 0.75rem;
    }

    .stat-values {
        display: flex;
        justify-content: space-between;
    }

    .stat-val {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.1rem;
        font-weight: 600;
    }
    .stat-val.home { color: #f87171; }
    .stat-val.away { color: #60a5fa; }

    .stat-name {
        font-size: 0.6rem;
        color: #666;
        text-transform: uppercase;
        text-align: center;
        margin-top: 0.25rem;
    }

    /* Panel Header */
    .panel-header {
        font-size: 0.7rem;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2a2a35;
        margin-bottom: 0.75rem;
    }

    .player-name {
        font-size: 1rem;
        font-weight: 600;
        color: #fff;
    }

    .event-meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #888;
    }

    .team-badge {
        font-size: 0.6rem;
        font-weight: 600;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .team-badge.home { background: rgba(248,113,113,0.15); color: #fca5a5; }
    .team-badge.away { background: rgba(96,165,250,0.15); color: #93c5fd; }

    .result-badge {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.6rem;
        padding: 0.15rem 0.4rem;
        border-radius: 3px;
    }
    .result-badge.success { background: rgba(52,211,153,0.15); color: #34d399; }
    .result-badge.fail { background: rgba(248,113,113,0.1); color: #fca5a5; }

    .event-detail { font-size: 0.8rem; color: #888; margin-top: 0.4rem; }

    .goal-flash {
        font-size: 1.2rem;
        font-weight: 700;
        color: #fbbf24;
        margin-top: 0.5rem;
    }

    .mode-stats { font-size: 0.85rem; color: #ddd; margin-bottom: 0.4rem; }
    .mode-stats strong { color: #60a5fa; }

    .team-stat-line {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        padding: 0.2rem 0;
    }

    .event-counter {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: #888;
    }
    .event-counter strong { color: #60a5fa; }

    .context-item {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.7rem;
        padding: 0.25rem 0.5rem;
        color: #666;
        border-left: 2px solid #333;
        margin: 0.1rem 0;
    }
    .context-item.active {
        color: #fff;
        background: rgba(59,130,246,0.1);
        border-left-color: #3b82f6;
    }

    .section-label {
        font-size: 0.65rem;
        font-weight: 600;
        color: #555;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 1rem 0 0.4rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# CONSTANTS
# =============================================================================

COLORS = {
    'home': '#f87171',
    'away': '#60a5fa',
    'pitch': '#0f1c14',
    'pitch_lines': '#2d5a3d',
    'lines': 'rgba(255,255,255,0.7)',
    'bg': '#06080a',
    'bg_card': '#12151a',
    'text': '#f0f2f5',
    'muted': '#4b5563',
    'accent': '#3b82f6',
    'success': '#34d399',
    'gold': '#fbbf24',
}

EVENT_MARKERS = {
    'pass': 'o', 'cross': 'o', 'corner_crossed': 'o', 'corner_short': 'o',
    'freekick_crossed': 'o', 'freekick_short': 'o',
    'shot': '^', 'shot_freekick': '^', 'shot_penalty': '^',
    'tackle': 's', 'interception': 's', 'foul': 'X',
    'dribble': 'D', 'take_on': 'D', 'bad_touch': 'D',
    'clearance': 'p', 'goalkick': 'p', 'throw_in': 'p',
    'keeper_save': 'H', 'keeper_pick_up': 'H', 'keeper_claim': 'H', 'keeper_punch': 'H',
}

ARROW_EVENTS = ['pass', 'cross', 'corner_crossed', 'freekick_crossed', 'corner_short', 'freekick_short']
CARRY_EVENTS = ['dribble', 'take_on']
SHOT_EVENTS = ['shot', 'shot_freekick', 'shot_penalty']
DEFENSIVE_EVENTS = ['tackle', 'interception', 'clearance', 'foul']

# View modes for the app
VIEW_MODES = ["Sequence", "Shot Map", "Pass Network", "Heatmap", "Progressive", "Pressing", "Momentum"]

# =============================================================================
# UTILITIES
# =============================================================================

def is_carry(row):
    if row['type_name'] not in CARRY_EVENTS:
        return False
    if pd.isna(row['end_x']) or pd.isna(row['end_y']):
        return False
    return np.sqrt((row['end_x'] - row['start_x'])**2 + (row['end_y'] - row['start_y'])**2) > 1.0


def is_progressive(row, threshold=10):
    """Check if action moves ball significantly toward goal (x increases by threshold)"""
    if pd.isna(row['end_x']) or pd.isna(row['start_x']):
        return False
    x_gain = row['end_x'] - row['start_x']
    return x_gain >= threshold and row['result_name'] == 'success'


def compute_pass_network(df, team):
    """Compute pass connections between players for a team"""
    team_df = df[(df['team_name'] == team) & (df['type_name'].isin(ARROW_EVENTS))]
    team_df = team_df[team_df['result_name'] == 'success']

    # Get average positions
    avg_pos = team_df.groupby('player_name').agg({
        'start_x': 'mean',
        'start_y': 'mean',
        'player_name': 'count'
    }).rename(columns={'player_name': 'count'}).reset_index()

    # Get pass connections (need to identify receiver from next action)
    connections = {}
    team_actions = df[df['team_name'] == team].sort_values('action_id')

    for i in range(len(team_actions) - 1):
        curr = team_actions.iloc[i]
        next_act = team_actions.iloc[i + 1]

        if curr['type_name'] in ARROW_EVENTS and curr['result_name'] == 'success':
            passer = curr['player_name']
            receiver = next_act['player_name']
            if passer != receiver and passer != 'Unknown' and receiver != 'Unknown':
                key = (passer, receiver)
                connections[key] = connections.get(key, 0) + 1

    return avg_pos, connections


def compute_momentum(df, home, away, interval=5):
    """Compute momentum (action count) per time interval for each team"""
    max_min = int(df['minute'].max()) + 1
    intervals = range(0, max_min, interval)

    momentum = []
    for start in intervals:
        end = start + interval
        period_df = df[(df['minute'] >= start) & (df['minute'] < end)]
        h_actions = len(period_df[period_df['team_name'] == home])
        a_actions = len(period_df[period_df['team_name'] == away])
        total = h_actions + a_actions
        if total > 0:
            h_pct = h_actions / total
        else:
            h_pct = 0.5
        momentum.append({'start': start, 'end': end, 'home_pct': h_pct,
                        'home': h_actions, 'away': a_actions})

    return momentum


def flip_coords(df, away_team):
    df = df.copy()
    mask = df['team_name'] == away_team
    for col in ['start_x', 'end_x', 'start_y', 'end_y']:
        df.loc[mask, col] = 100 - df.loc[mask, col]
    return df


def fig_to_buffer(fig, dpi=200):
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor(), edgecolor='none')
    buf.seek(0)
    return buf


@st.cache_data
def load_data():
    actions = pd.read_csv('consolidated_defined_actions.csv')
    players = pd.read_csv('consolidated_players.csv')
    teams = pd.read_csv('consolidated_teams.csv').drop_duplicates(subset=['team_id'])

    actions = actions.merge(teams[['team_id', 'team_name']], on='team_id', how='left')
    actions = actions.merge(
        players[['player_id', 'player_name']].drop_duplicates(subset=['player_id']),
        on='player_id', how='left'
    )

    actions['player_name'] = actions['player_name'].fillna('Unknown')
    actions['minute'] = (actions['time_seconds_overall'] / 60).astype(int)
    actions['second'] = (actions['time_seconds_overall'] % 60).astype(int)
    actions['time_display'] = actions.apply(lambda x: f"{x['minute']}'{x['second']:02d}\"", axis=1)

    game_teams = actions.groupby('game_id').agg({'team_name': lambda x: list(x.unique())}).reset_index()
    game_teams['label'] = game_teams['team_name'].apply(lambda x: f"{x[0]} vs {x[1]}" if len(x) >= 2 else str(x))

    return actions, dict(zip(game_teams['game_id'], game_teams['label']))


def compute_stats(df, home, away, home_id=None, away_id=None):
    h, a = df[df['team_name'] == home], df[df['team_name'] == away]
    shots_h = h[h['type_name'].isin(SHOT_EVENTS)]
    shots_a = a[a['type_name'].isin(SHOT_EVENTS)]
    passes_h = h[h['type_name'] == 'pass']
    passes_a = a[a['type_name'] == 'pass']

    # Count regular goals
    goals_h = len(shots_h[shots_h['goal_from_shot'] == True])
    goals_a = len(shots_a[shots_a['goal_from_shot'] == True])

    # Handle own goals - credit to opposing team
    if home_id is not None and away_id is not None:
        owngoals = df[df['result_name'] == 'owngoal']
        for _, row in owngoals.iterrows():
            if row['team_id'] == home_id:
                goals_a += 1  # Own goal by home team = goal for away
            else:
                goals_h += 1  # Own goal by away team = goal for home

    return {
        'goals_h': goals_h, 'goals_a': goals_a,
        'shots_h': len(shots_h), 'shots_a': len(shots_a),
        'pass_pct_h': (passes_h['result_name'] == 'success').mean() * 100 if len(passes_h) else 0,
        'pass_pct_a': (passes_a['result_name'] == 'success').mean() * 100 if len(passes_a) else 0,
    }


def get_passes_into_areas(df, team, home_team):
    """
    Get passes into final third and penalty area for a team.
    After flip_coords: home attacks RIGHT (x=100), away attacks LEFT (x=0)
    """
    team_df = df[(df['team_name'] == team) & (df['type_name'].isin(ARROW_EVENTS))]
    successful = team_df[team_df['result_name'] == 'success']

    is_home = (team == home_team)

    if is_home:
        # Home attacks RIGHT: final third is x > 66.67, penalty area x > 83
        final_third = successful[
            (successful['end_x'] >= 66.67) & (successful['start_x'] < 66.67)
        ]
        penalty_area = successful[
            (successful['end_x'] >= 83) &
            (successful['end_y'] >= 21) & (successful['end_y'] <= 79) &
            (successful['start_x'] < 83)
        ]
    else:
        # Away attacks LEFT: final third is x < 33.33, penalty area x < 17
        final_third = successful[
            (successful['end_x'] <= 33.33) & (successful['start_x'] > 33.33)
        ]
        penalty_area = successful[
            (successful['end_x'] <= 17) &
            (successful['end_y'] >= 21) & (successful['end_y'] <= 79) &
            (successful['start_x'] > 17)
        ]

    return final_third, penalty_area


def is_progressive_advanced(row):
    """Advanced progressive detection based on pitch position (from matchreport.py)"""
    if pd.isna(row['end_x']) or pd.isna(row['start_x']):
        return False
    if row['result_name'] != 'success':
        return False

    start_x, end_x = row['start_x'], row['end_x']
    own_half = 50

    # Condition 1: Both in own half, moved 30+ yards forward
    if start_x <= own_half and end_x <= own_half and (end_x - start_x) >= 30:
        return True
    # Condition 2: Started own half, ended opponent half, moved 15+ yards
    if start_x <= own_half and end_x > own_half and (end_x - start_x) >= 15:
        return True
    # Condition 3: Both in opponent half, moved 10+ yards forward
    if start_x > own_half and end_x > own_half and (end_x - start_x) >= 10:
        return True

    return False


# =============================================================================
# MATCH REPORT GENERATOR
# =============================================================================

def generate_match_report(df, home, away, stats, home_id=None, away_id=None):
    """
    Generate a comprehensive match report image with:
    - Scoreline header
    - Pass networks for both teams
    - Shot map
    - Momentum chart
    - Key statistics
    """
    from mplsoccer import VerticalPitch

    # Premium dark theme colors
    BG = '#06080a'
    BG_CARD = '#12151a'
    TEXT = '#f0f2f5'
    TEXT_MUTED = '#6b7280'
    HOME_COLOR = '#f87171'
    AWAY_COLOR = '#60a5fa'
    GOLD = '#fbbf24'
    PITCH_COLOR = '#0f1c14'
    PITCH_LINES = '#2d5a3d'

    # Create figure with grid
    fig = plt.figure(figsize=(20, 28), facecolor=BG)
    gs = fig.add_gridspec(14, 12, hspace=0.3, wspace=0.3,
                          left=0.04, right=0.96, top=0.95, bottom=0.02)

    # === HEADER - Scoreline ===
    ax_header = fig.add_subplot(gs[0:2, :])
    ax_header.set_facecolor(BG)
    ax_header.axis('off')

    # Title
    ax_header.text(0.5, 0.85, 'MATCH REPORT', transform=ax_header.transAxes,
                   fontsize=14, color=TEXT_MUTED, ha='center', va='center',
                   fontweight='bold', fontfamily='sans-serif')

    # Team names and score
    ax_header.text(0.35, 0.45, home.upper(), transform=ax_header.transAxes,
                   fontsize=32, color=HOME_COLOR, ha='right', va='center',
                   fontweight='bold', fontfamily='sans-serif')
    ax_header.text(0.5, 0.45, f"{stats['goals_h']}  -  {stats['goals_a']}",
                   transform=ax_header.transAxes, fontsize=38, color=TEXT,
                   ha='center', va='center', fontweight='bold', fontfamily='monospace')
    ax_header.text(0.65, 0.45, away.upper(), transform=ax_header.transAxes,
                   fontsize=32, color=AWAY_COLOR, ha='left', va='center',
                   fontweight='bold', fontfamily='sans-serif')

    # Subtitle
    ax_header.text(0.5, 0.1, 'Full Match Analysis', transform=ax_header.transAxes,
                   fontsize=11, color=TEXT_MUTED, ha='center', va='center',
                   fontfamily='sans-serif')

    # === PASS NETWORKS ===
    # Home pass network
    ax_home_net = fig.add_subplot(gs[2:6, 0:5])
    ax_home_net.set_facecolor(BG_CARD)
    _draw_pass_network_report(ax_home_net, df, home, HOME_COLOR, PITCH_COLOR, PITCH_LINES)
    ax_home_net.set_title(f'{home} Pass Network', fontsize=12, color=HOME_COLOR,
                          fontweight='bold', pad=10, fontfamily='sans-serif')

    # Away pass network
    ax_away_net = fig.add_subplot(gs[2:6, 7:12])
    ax_away_net.set_facecolor(BG_CARD)
    _draw_pass_network_report(ax_away_net, df, away, AWAY_COLOR, PITCH_COLOR, PITCH_LINES)
    ax_away_net.set_title(f'{away} Pass Network', fontsize=12, color=AWAY_COLOR,
                          fontweight='bold', pad=10, fontfamily='sans-serif')

    # === STATS PANEL (center) ===
    ax_stats = fig.add_subplot(gs[2:6, 5:7])
    ax_stats.set_facecolor(BG)
    ax_stats.axis('off')

    # Calculate additional stats
    h_df = df[df['team_name'] == home]
    a_df = df[df['team_name'] == away]

    h_passes = len(h_df[h_df['type_name'] == 'pass'])
    a_passes = len(a_df[a_df['type_name'] == 'pass'])
    h_prog = len(h_df[(h_df['type_name'].isin(ARROW_EVENTS)) & (h_df.apply(is_progressive_advanced, axis=1))])
    a_prog = len(a_df[(a_df['type_name'].isin(ARROW_EVENTS)) & (a_df.apply(is_progressive_advanced, axis=1))])
    h_tackles = len(h_df[h_df['type_name'] == 'tackle'])
    a_tackles = len(a_df[a_df['type_name'] == 'tackle'])

    stat_items = [
        ('Shots', stats['shots_h'], stats['shots_a']),
        ('Pass %', f"{stats['pass_pct_h']:.0f}", f"{stats['pass_pct_a']:.0f}"),
        ('Passes', h_passes, a_passes),
        ('Progressive', h_prog, a_prog),
        ('Tackles', h_tackles, a_tackles),
    ]

    for i, (label, h_val, a_val) in enumerate(stat_items):
        y = 0.85 - i * 0.18
        ax_stats.text(0.5, y, label.upper(), transform=ax_stats.transAxes,
                      fontsize=8, color=TEXT_MUTED, ha='center', va='center',
                      fontweight='bold', fontfamily='sans-serif')
        ax_stats.text(0.15, y - 0.06, str(h_val), transform=ax_stats.transAxes,
                      fontsize=16, color=HOME_COLOR, ha='center', va='center',
                      fontweight='bold', fontfamily='monospace')
        ax_stats.text(0.85, y - 0.06, str(a_val), transform=ax_stats.transAxes,
                      fontsize=16, color=AWAY_COLOR, ha='center', va='center',
                      fontweight='bold', fontfamily='monospace')

    # === SHOT MAP ===
    ax_shots = fig.add_subplot(gs[6:10, 0:6])
    ax_shots.set_facecolor(BG_CARD)
    pitch = Pitch(pitch_type='opta', pitch_color=PITCH_COLOR, line_color=PITCH_LINES,
                  linewidth=1.0, goal_type='box')
    pitch.draw(ax=ax_shots)

    # Draw shots for both teams
    shots_df = df[df['type_name'].isin(SHOT_EVENTS)]
    for _, r in shots_df.iterrows():
        is_home = r['team_name'] == home
        c = HOME_COLOR if is_home else AWAY_COLOR
        goal = r.get('goal_from_shot', False)
        m = '*' if goal else '^'
        sz = 400 if goal else 200
        ec = GOLD if goal else 'white'
        ax_shots.scatter(r['start_x'], r['start_y'], s=sz, c=c, marker=m,
                        alpha=1.0 if goal else 0.8, edgecolors=ec, linewidths=2, zorder=10)

    ax_shots.set_title('Shot Map', fontsize=12, color=TEXT, fontweight='bold',
                       pad=10, fontfamily='sans-serif')

    # === MOMENTUM CHART ===
    ax_momentum = fig.add_subplot(gs[6:8, 6:12])
    ax_momentum.set_facecolor(BG_CARD)

    momentum = compute_momentum(df, home, away, interval=5)

    for i, m in enumerate(momentum):
        ax_momentum.bar(i, m['home_pct'], width=0.85, color=HOME_COLOR, alpha=0.8, bottom=0)
        ax_momentum.bar(i, 1 - m['home_pct'], width=0.85, color=AWAY_COLOR, alpha=0.8, bottom=m['home_pct'])

    ax_momentum.axhline(0.5, color='white', linestyle='--', alpha=0.3, linewidth=1)
    ax_momentum.set_xlim(-0.5, len(momentum) - 0.5)
    ax_momentum.set_ylim(0, 1)
    ax_momentum.set_xticks(range(0, len(momentum), 2))
    ax_momentum.set_xticklabels([f"{m['start']}'" for m in momentum[::2]], fontsize=7, color=TEXT_MUTED)
    ax_momentum.set_yticks([])
    for spine in ax_momentum.spines.values():
        spine.set_visible(False)
    ax_momentum.set_title('Match Momentum', fontsize=12, color=TEXT, fontweight='bold',
                          pad=10, fontfamily='sans-serif')

    # === PROGRESSIVE ACTIONS ===
    ax_prog = fig.add_subplot(gs[8:10, 6:12])
    ax_prog.set_facecolor(BG_CARD)
    pitch2 = Pitch(pitch_type='opta', pitch_color=PITCH_COLOR, line_color=PITCH_LINES,
                   linewidth=1.0, goal_type='box')
    pitch2.draw(ax=ax_prog)

    prog_passes = df[(df['type_name'].isin(ARROW_EVENTS)) & (df.apply(is_progressive_advanced, axis=1))]
    for _, r in prog_passes.iterrows():
        c = HOME_COLOR if r['team_name'] == home else AWAY_COLOR
        pitch2.arrows(r['start_x'], r['start_y'], r['end_x'], r['end_y'],
                     ax=ax_prog, color=c, alpha=0.7, width=1.5, headwidth=4, headlength=4)

    ax_prog.set_title('Progressive Passes', fontsize=12, color=TEXT, fontweight='bold',
                      pad=10, fontfamily='sans-serif')

    # === PASSES INTO BOX ===
    ax_box_h = fig.add_subplot(gs[10:12, 0:6])
    ax_box_h.set_facecolor(BG_CARD)
    pitch3 = Pitch(pitch_type='opta', pitch_color=PITCH_COLOR, line_color=PITCH_LINES,
                   linewidth=1.0, goal_type='box')
    pitch3.draw(ax=ax_box_h)

    _, h_box = get_passes_into_areas(df, home, home)
    _, a_box = get_passes_into_areas(df, away, home)

    for _, r in h_box.iterrows():
        pitch3.arrows(r['start_x'], r['start_y'], r['end_x'], r['end_y'],
                     ax=ax_box_h, color=HOME_COLOR, alpha=0.8, width=2, headwidth=5, headlength=5)
    for _, r in a_box.iterrows():
        pitch3.arrows(r['start_x'], r['start_y'], r['end_x'], r['end_y'],
                     ax=ax_box_h, color=AWAY_COLOR, alpha=0.8, width=2, headwidth=5, headlength=5)

    ax_box_h.set_title('Passes Into Penalty Area', fontsize=12, color=TEXT, fontweight='bold',
                       pad=10, fontfamily='sans-serif')

    # === FINAL THIRD PASSES ===
    ax_ft = fig.add_subplot(gs[10:12, 6:12])
    ax_ft.set_facecolor(BG_CARD)
    pitch4 = Pitch(pitch_type='opta', pitch_color=PITCH_COLOR, line_color=PITCH_LINES,
                   linewidth=1.0, goal_type='box')
    pitch4.draw(ax=ax_ft)

    h_ft, _ = get_passes_into_areas(df, home, home)
    a_ft, _ = get_passes_into_areas(df, away, home)

    for _, r in h_ft.iterrows():
        pitch4.arrows(r['start_x'], r['start_y'], r['end_x'], r['end_y'],
                     ax=ax_ft, color=HOME_COLOR, alpha=0.7, width=1.5, headwidth=4, headlength=4)
    for _, r in a_ft.iterrows():
        pitch4.arrows(r['start_x'], r['start_y'], r['end_x'], r['end_y'],
                     ax=ax_ft, color=AWAY_COLOR, alpha=0.7, width=1.5, headwidth=4, headlength=4)

    ax_ft.axvline(66.67, color=GOLD, linestyle='--', alpha=0.5, linewidth=2)
    ax_ft.set_title('Passes Into Final Third', fontsize=12, color=TEXT, fontweight='bold',
                    pad=10, fontfamily='sans-serif')

    # === FOOTER ===
    ax_footer = fig.add_subplot(gs[13:14, :])
    ax_footer.set_facecolor(BG)
    ax_footer.axis('off')

    ax_footer.text(0.5, 0.7, 'Generated by Matchday Analytics', transform=ax_footer.transAxes,
                   fontsize=10, color=TEXT_MUTED, ha='center', va='center',
                   fontfamily='sans-serif')
    ax_footer.text(0.5, 0.3, '@yureedelahi', transform=ax_footer.transAxes,
                   fontsize=9, color=TEXT_MUTED, ha='center', va='center',
                   fontfamily='sans-serif', alpha=0.7)

    # Legend
    ax_footer.scatter(0.35, 0.5, s=100, c=HOME_COLOR, marker='o', transform=ax_footer.transAxes)
    ax_footer.text(0.37, 0.5, home[:12], transform=ax_footer.transAxes, fontsize=9,
                   color=HOME_COLOR, va='center', fontfamily='sans-serif')
    ax_footer.scatter(0.55, 0.5, s=100, c=AWAY_COLOR, marker='o', transform=ax_footer.transAxes)
    ax_footer.text(0.57, 0.5, away[:12], transform=ax_footer.transAxes, fontsize=9,
                   color=AWAY_COLOR, va='center', fontfamily='sans-serif')

    return fig


def _draw_pass_network_report(ax, df, team, color, pitch_color, line_color):
    """Draw pass network for match report"""
    from mplsoccer import VerticalPitch

    pitch = VerticalPitch(pitch_type='opta', pitch_color=pitch_color, line_color=line_color,
                          linewidth=1.0, goal_type='box')
    pitch.draw(ax=ax)

    avg_pos, connections = compute_pass_network(df, team)

    if avg_pos.empty:
        ax.text(50, 50, "No data", fontsize=10, color='white', ha='center', va='center')
        return

    # Draw connections
    max_passes = max(connections.values()) if connections else 1
    for (passer, receiver), count in connections.items():
        p1 = avg_pos[avg_pos['player_name'] == passer]
        p2 = avg_pos[avg_pos['player_name'] == receiver]
        if len(p1) > 0 and len(p2) > 0:
            lw = 0.5 + (count / max_passes) * 6
            alpha = 0.2 + (count / max_passes) * 0.6
            ax.plot([p1['start_x'].values[0], p2['start_x'].values[0]],
                   [p1['start_y'].values[0], p2['start_y'].values[0]],
                   color=color, alpha=alpha, linewidth=lw, zorder=5)

    # Draw player nodes
    max_actions = avg_pos['count'].max() if len(avg_pos) > 0 else 1
    for _, p in avg_pos.iterrows():
        sz = 150 + (p['count'] / max_actions) * 400
        ax.scatter(p['start_x'], p['start_y'], s=sz, c=color, alpha=0.9,
                  edgecolors='white', linewidths=1.5, zorder=10)
        name = p['player_name'].split()[-1][:8]
        ax.text(p['start_x'], p['start_y'] - 4, name, fontsize=6,
               color='white', ha='center', va='top', fontweight='bold', zorder=11)


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_viz(df, home, away, title, subtitle, mode='sequence', sel_idx=None,
               arrows=True, numbers=True, selected_team='home', full_match_df=None):
    """
    Main visualization function supporting multiple modes:
    - sequence: Event flow with markers and arrows
    - shots: Shot map with goal lines
    - pass_network: Player connections
    - heatmap: Activity zones
    - progressive: Progressive passes/carries
    - pressing: Defensive actions by zone
    - momentum: Timeline of match control
    - final_third: Passes into final third
    - penalty_area: Passes into penalty area
    """
    # For momentum, use a different layout
    if mode == 'momentum':
        fig = plt.figure(figsize=(14, 4), facecolor=COLORS['bg'])
        ax = fig.add_subplot(111)
        use_df = full_match_df if full_match_df is not None else df
        _draw_momentum(ax, use_df, home, away)
        return fig

    fig = plt.figure(figsize=(14, 10), facecolor=COLORS['bg'])
    gs = fig.add_gridspec(20, 20, left=0.02, right=0.98, top=0.92, bottom=0.08)

    # Header
    ax_hdr = fig.add_subplot(gs[0:2, :])
    ax_hdr.set_facecolor(COLORS['bg'])
    ax_hdr.axis('off')

    # Title with custom font styling
    ax_hdr.text(0.5, 0.8, title.upper(), transform=ax_hdr.transAxes, fontsize=28,
                fontweight='bold', color=COLORS['text'], ha='center', va='center',
                fontfamily='sans-serif')
    ax_hdr.text(0.5, 0.25, subtitle, transform=ax_hdr.transAxes, fontsize=10,
                color=COLORS['muted'], ha='center', va='center', fontfamily='monospace')

    # Team indicators with direction arrows
    ax_hdr.text(0.04, 0.55, home, transform=ax_hdr.transAxes, fontsize=13,
                fontweight='bold', color=COLORS['home'], ha='left', va='center',
                fontfamily='sans-serif')
    ax_hdr.annotate('', xy=(0.15, 0.55), xytext=(0.045 + len(home)*0.012, 0.55),
                    transform=ax_hdr.transAxes,
                    arrowprops=dict(arrowstyle='->', color=COLORS['home'], lw=1.5))

    ax_hdr.text(0.96, 0.55, away, transform=ax_hdr.transAxes, fontsize=13,
                fontweight='bold', color=COLORS['away'], ha='right', va='center',
                fontfamily='sans-serif')
    ax_hdr.annotate('', xy=(0.955 - len(away)*0.012, 0.55), xytext=(0.85, 0.55),
                    transform=ax_hdr.transAxes,
                    arrowprops=dict(arrowstyle='<-', color=COLORS['away'], lw=1.5))

    # Premium dark pitch - deep forest green with subtle lines
    ax = fig.add_subplot(gs[2:18, 1:19])
    pitch = Pitch(pitch_type='opta', pitch_color=COLORS['pitch'],
                  line_color=COLORS['pitch_lines'], linewidth=1.0,
                  goal_type='box', corner_arcs=True)
    pitch.draw(ax=ax)

    if df.empty:
        ax.text(50, 50, "No events match your filters", fontsize=12,
                color=COLORS['muted'], ha='center', va='center', fontfamily='monospace')
        return fig

    df = df.sort_values('action_id').reset_index(drop=True)

    # Draw based on mode
    if mode == 'shots':
        _draw_shots(ax, pitch, df, home)
    elif mode == 'pass_network':
        _draw_pass_network(ax, pitch, df, home, away, selected_team)
    elif mode == 'heatmap':
        _draw_heatmap(ax, pitch, df, home, away, selected_team)
    elif mode == 'progressive':
        _draw_progressive(ax, pitch, df, home)
    elif mode == 'pressing':
        _draw_pressing(ax, pitch, df, home, away, selected_team)
    elif mode == 'final_third':
        _draw_final_third(ax, pitch, df, home, away)
    elif mode == 'penalty_area':
        _draw_penalty_area(ax, pitch, df, home, away)
    else:
        _draw_sequence(ax, pitch, df, home, sel_idx, arrows, numbers)

    # Footer with legend
    ax_ftr = fig.add_subplot(gs[18:20, :])
    ax_ftr.set_facecolor(COLORS['bg'])
    ax_ftr.axis('off')

    # Dynamic legend based on mode
    if mode in ['pass_network', 'heatmap', 'pressing']:
        team_shown = home if selected_team == 'home' else away
        color = COLORS['home'] if selected_team == 'home' else COLORS['away']
        ax_ftr.text(0.05, 0.6, f"Showing: {team_shown}", fontsize=10,
                   color=color, fontweight='bold', transform=ax_ftr.transAxes)
    elif mode == 'progressive':
        ax_ftr.scatter(0.05, 0.6, s=50, c=COLORS['gold'], marker='o', transform=ax_ftr.transAxes)
        ax_ftr.text(0.07, 0.6, "Progressive Pass", fontsize=8, color='#8b8178',
                   va='center', transform=ax_ftr.transAxes)
        ax_ftr.scatter(0.25, 0.6, s=50, c=COLORS['gold'], marker='D', transform=ax_ftr.transAxes)
        ax_ftr.text(0.27, 0.6, "Progressive Carry", fontsize=8, color='#8b8178',
                   va='center', transform=ax_ftr.transAxes)
    else:
        legend = [(COLORS['home'], 'o', home), (COLORS['away'], 'o', away),
                  ('#888888', '^', 'Shot'), (COLORS['gold'], '*', 'Goal')]
        for i, (c, m, lbl) in enumerate(legend):
            x = 0.05 + i * 0.15
            ax_ftr.scatter(x, 0.6, s=50, c=c, marker=m, transform=ax_ftr.transAxes)
            ax_ftr.text(x + 0.018, 0.6, lbl, transform=ax_ftr.transAxes, fontsize=8,
                        color='#8b8178', va='center', fontfamily='monospace')

        ax_ftr.text(0.02, 0.15, "— passes    - - carries", transform=ax_ftr.transAxes,
                    fontsize=7, color='#5c554f', fontfamily='monospace')

    return fig


def _draw_sequence(ax, pitch, df, home, sel_idx, arrows, numbers):
    if arrows:
        for idx, r in df[df['type_name'].isin(ARROW_EVENTS)].iterrows():
            c = COLORS['home'] if r['team_name'] == home else COLORS['away']
            a = 0.85 if r['result_name'] == 'success' else 0.3
            lw = 2.5 if sel_idx == idx else 1.2
            if sel_idx is not None and sel_idx != idx:
                a *= 0.5
            pitch.arrows(r['start_x'], r['start_y'], r['end_x'], r['end_y'],
                         ax=ax, color=c, alpha=a, width=lw, headwidth=5, headlength=5)

        for idx, r in df.iterrows():
            if is_carry(r):
                c = COLORS['home'] if r['team_name'] == home else COLORS['away']
                a = 0.7 if r['result_name'] == 'success' else 0.3
                lw = 2.5 if sel_idx == idx else 1.5
                if sel_idx is not None and sel_idx != idx:
                    a *= 0.5
                ax.annotate('', xy=(r['end_x'], r['end_y']), xytext=(r['start_x'], r['start_y']),
                            arrowprops=dict(arrowstyle='-|>', color=c, alpha=a, lw=lw,
                                            linestyle='--', mutation_scale=12), zorder=5)

    for idx, r in df.iterrows():
        c = COLORS['home'] if r['team_name'] == home else COLORS['away']
        m = EVENT_MARKERS.get(r['type_name'], 'o')
        selected = sel_idx == idx
        sz = 280 if selected else (120 if sel_idx is None else 70)
        a = 1.0 if selected or sel_idx is None else 0.4
        ec = COLORS['gold'] if selected else 'white'
        lw = 3 if selected else 1.2

        if r['type_name'] in SHOT_EVENTS and r.get('goal_from_shot', False):
            m, sz, ec, lw = '*', sz * 2, COLORS['gold'], 2

        ax.scatter(r['start_x'], r['start_y'], s=sz, c=c, marker=m, alpha=a,
                   edgecolors=ec, linewidths=lw, zorder=10)

        if numbers:
            na = 1.0 if selected or sel_idx is None else 0.35
            ax.scatter(r['start_x'] + 2.8, r['start_y'] + 2.8, s=160,
                       c=COLORS['bg'], alpha=na * 0.85, marker='o', zorder=11)
            ax.text(r['start_x'] + 2.8, r['start_y'] + 2.8, str(idx + 1), fontsize=7,
                    color='white', alpha=na, fontweight='bold', ha='center', va='center', zorder=12)


def _draw_shots(ax, pitch, df, home):
    """Draw shots - home attacks right (x=100), away attacks left (x=0) after flip_coords"""
    for _, r in df[df['type_name'].isin(SHOT_EVENTS)].iterrows():
        is_home = r['team_name'] == home
        c = COLORS['home'] if is_home else COLORS['away']
        goal = r.get('goal_from_shot', False)
        m = '*' if goal else '^'
        sz = 500 if goal else 250
        ec = COLORS['gold'] if goal else 'white'

        ax.scatter(r['start_x'], r['start_y'], s=sz, c=c, marker=m,
                   alpha=1.0 if goal else 0.85, edgecolors=ec, linewidths=2, zorder=10)

        # Home attacks right goal (x=100), Away attacks left goal (x=0)
        goal_x = 100 if is_home else 0
        ax.plot([r['start_x'], goal_x], [r['start_y'], 50], color=c,
                alpha=0.15, linestyle='--', linewidth=0.8, zorder=5)


def _draw_passes(ax, pitch, df, home, sel_idx):
    for idx, r in df[df['type_name'].isin(ARROW_EVENTS)].iterrows():
        c = COLORS['home'] if r['team_name'] == home else COLORS['away']
        success = r['result_name'] == 'success'
        a = 0.8 if success else 0.25
        lw = 1.5 if success else 0.8

        if sel_idx is not None:
            a = 1.0 if idx == sel_idx else a * 0.4
            lw = 3 if idx == sel_idx else lw

        pitch.arrows(r['start_x'], r['start_y'], r['end_x'], r['end_y'],
                     ax=ax, color=c, alpha=a, width=lw, headwidth=4, headlength=4)


def _draw_pass_network(ax, pitch, df, home, away, selected_team):
    """Draw pass network showing player connections"""
    team = home if selected_team == 'home' else away
    color = COLORS['home'] if selected_team == 'home' else COLORS['away']

    avg_pos, connections = compute_pass_network(df, team)

    if avg_pos.empty:
        ax.text(50, 50, f"No passing data for {team}", fontsize=12,
                color=COLORS['muted'], ha='center', va='center')
        return

    # Draw connections
    max_passes = max(connections.values()) if connections else 1
    for (passer, receiver), count in connections.items():
        p1 = avg_pos[avg_pos['player_name'] == passer]
        p2 = avg_pos[avg_pos['player_name'] == receiver]
        if len(p1) > 0 and len(p2) > 0:
            lw = 1 + (count / max_passes) * 8
            alpha = 0.3 + (count / max_passes) * 0.5
            ax.plot([p1['start_x'].values[0], p2['start_x'].values[0]],
                   [p1['start_y'].values[0], p2['start_y'].values[0]],
                   color=color, alpha=alpha, linewidth=lw, zorder=5)

    # Draw player nodes
    max_actions = avg_pos['count'].max() if len(avg_pos) > 0 else 1
    for _, p in avg_pos.iterrows():
        sz = 200 + (p['count'] / max_actions) * 600
        ax.scatter(p['start_x'], p['start_y'], s=sz, c=color, alpha=0.9,
                  edgecolors='white', linewidths=2, zorder=10)
        # Player name (shortened)
        name = p['player_name'].split()[-1][:10]
        ax.text(p['start_x'], p['start_y'] - 5, name, fontsize=7,
               color='white', ha='center', va='top', fontweight='bold', zorder=11)


def _draw_heatmap(ax, pitch, df, home, away, selected_team):
    """Draw activity heatmap for a team"""
    if selected_team == 'both':
        team_df = df
        cmap = 'YlOrRd'
    else:
        team = home if selected_team == 'home' else away
        team_df = df[df['team_name'] == team]
        cmap = 'Reds' if selected_team == 'home' else 'Blues'

    if team_df.empty:
        ax.text(50, 50, "No data for heatmap", fontsize=12,
                color=COLORS['muted'], ha='center', va='center')
        return

    # Create 2D histogram
    x = team_df['start_x'].values
    y = team_df['start_y'].values

    # Use pitch heatmap
    from scipy.ndimage import gaussian_filter
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=[20, 16], range=[[0, 100], [0, 100]])
    heatmap = gaussian_filter(heatmap.T, sigma=1.5)

    ax.imshow(heatmap, extent=[0, 100, 0, 100], origin='lower', cmap=cmap,
              alpha=0.7, aspect='auto', zorder=2)


def _draw_progressive(ax, pitch, df, home):
    """Draw progressive passes and carries using advanced detection"""
    # Use advanced progressive detection from matchreport.py
    prog_passes = df[(df['type_name'].isin(ARROW_EVENTS)) & (df.apply(is_progressive_advanced, axis=1))]
    prog_carries = df[df.apply(lambda r: is_carry(r) and is_progressive_advanced(r), axis=1)]

    # Draw progressive passes
    for _, r in prog_passes.iterrows():
        c = COLORS['home'] if r['team_name'] == home else COLORS['away']
        pitch.arrows(r['start_x'], r['start_y'], r['end_x'], r['end_y'],
                    ax=ax, color=COLORS['gold'], alpha=0.85, width=2,
                    headwidth=5, headlength=5, zorder=6)
        ax.scatter(r['start_x'], r['start_y'], s=80, c=c, alpha=0.9,
                  edgecolors=COLORS['gold'], linewidths=2, zorder=7)

    # Draw progressive carries (dashed)
    for _, r in prog_carries.iterrows():
        c = COLORS['home'] if r['team_name'] == home else COLORS['away']
        ax.annotate('', xy=(r['end_x'], r['end_y']), xytext=(r['start_x'], r['start_y']),
                   arrowprops=dict(arrowstyle='-|>', color=COLORS['gold'], alpha=0.7,
                                  lw=2, linestyle='--', mutation_scale=12), zorder=6)
        ax.scatter(r['start_x'], r['start_y'], s=80, c=c, marker='D', alpha=0.9,
                  edgecolors=COLORS['gold'], linewidths=2, zorder=7)


def _draw_pressing(ax, pitch, df, home, away, selected_team):
    """Draw pressing/defensive actions in opponent's third"""
    team = home if selected_team == 'home' else away
    color = COLORS['home'] if selected_team == 'home' else COLORS['away']

    # Filter defensive actions in attacking third (x > 66.67 for home attacking right)
    team_df = df[(df['team_name'] == team) & (df['type_name'].isin(DEFENSIVE_EVENTS))]

    # High press = defensive action in opponent's third
    # For home team attacking right, opponent's third is x > 66.67
    # For away team attacking left, opponent's third is x < 33.33
    if selected_team == 'home':
        high_press = team_df[team_df['start_x'] > 66.67]
        mid_press = team_df[(team_df['start_x'] >= 33.33) & (team_df['start_x'] <= 66.67)]
        low_press = team_df[team_df['start_x'] < 33.33]
    else:
        high_press = team_df[team_df['start_x'] < 33.33]
        mid_press = team_df[(team_df['start_x'] >= 33.33) & (team_df['start_x'] <= 66.67)]
        low_press = team_df[team_df['start_x'] > 66.67]

    # Draw zones with shading
    ax.axvspan(66.67, 100, alpha=0.1, color=COLORS['gold'], zorder=1)
    ax.axvspan(0, 33.33, alpha=0.1, color=COLORS['gold'], zorder=1)

    # Draw all defensive actions
    for _, r in team_df.iterrows():
        is_high = r['start_x'] > 66.67 if selected_team == 'home' else r['start_x'] < 33.33
        alpha = 1.0 if is_high else 0.5
        sz = 180 if is_high else 100
        ec = COLORS['gold'] if is_high else 'white'
        ax.scatter(r['start_x'], r['start_y'], s=sz, c=color, marker='s',
                  alpha=alpha, edgecolors=ec, linewidths=2, zorder=8)


def _draw_momentum(ax, df, home, away):
    """Draw momentum timeline"""
    momentum = compute_momentum(df, home, away, interval=5)

    ax.set_facecolor(COLORS['bg'])
    ax.set_xlim(-0.5, len(momentum) - 0.5)
    ax.set_ylim(0, 1)

    for i, m in enumerate(momentum):
        # Home portion (bottom)
        ax.bar(i, m['home_pct'], width=0.9, color=COLORS['home'], alpha=0.8, bottom=0)
        # Away portion (top)
        ax.bar(i, 1 - m['home_pct'], width=0.9, color=COLORS['away'], alpha=0.8, bottom=m['home_pct'])

    # Center line
    ax.axhline(0.5, color='white', linestyle='--', alpha=0.3, linewidth=1)

    # Labels
    ax.set_xticks(range(len(momentum)))
    ax.set_xticklabels([f"{m['start']}'" for m in momentum], fontsize=7, color=COLORS['muted'])
    ax.set_yticks([])

    # Team labels
    ax.text(-0.02, 0.25, home[:3].upper(), transform=ax.transAxes, fontsize=9,
            color=COLORS['home'], fontweight='bold', ha='right', va='center')
    ax.text(-0.02, 0.75, away[:3].upper(), transform=ax.transAxes, fontsize=9,
            color=COLORS['away'], fontweight='bold', ha='right', va='center')

    # Remove spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title("MATCH MOMENTUM", fontsize=12, color=COLORS['text'],
                fontweight='bold', fontfamily='sans-serif', pad=10)


def _draw_final_third(ax, pitch, df, home, away):
    """Draw passes into final third for both teams"""
    for team in [home, away]:
        color = COLORS['home'] if team == home else COLORS['away']
        final_third, _ = get_passes_into_areas(df, team, home)

        for _, r in final_third.iterrows():
            pitch.arrows(r['start_x'], r['start_y'], r['end_x'], r['end_y'],
                        ax=ax, color=color, alpha=0.75, width=1.5,
                        headwidth=4, headlength=4, zorder=6)

    # Draw final third lines for BOTH sides
    ax.axvline(66.67, color=COLORS['home'], linestyle='--', alpha=0.4, linewidth=2, zorder=3)
    ax.axvline(33.33, color=COLORS['away'], linestyle='--', alpha=0.4, linewidth=2, zorder=3)
    ax.text(68, 95, f"{home[:3].upper()}", fontsize=8, color=COLORS['home'],
           fontweight='bold', alpha=0.7, rotation=90, va='top')
    ax.text(32, 95, f"{away[:3].upper()}", fontsize=8, color=COLORS['away'],
           fontweight='bold', alpha=0.7, rotation=90, va='top')


def _draw_penalty_area(ax, pitch, df, home, away):
    """Draw passes into penalty area for both teams"""
    from matplotlib.patches import Rectangle

    # Highlight BOTH penalty areas
    # Home team's target (right side)
    penalty_rect_right = Rectangle((83, 21), 17, 58, fill=True, facecolor=COLORS['home'],
                                    alpha=0.12, zorder=2)
    ax.add_patch(penalty_rect_right)

    # Away team's target (left side)
    penalty_rect_left = Rectangle((0, 21), 17, 58, fill=True, facecolor=COLORS['away'],
                                   alpha=0.12, zorder=2)
    ax.add_patch(penalty_rect_left)

    for team in [home, away]:
        color = COLORS['home'] if team == home else COLORS['away']
        _, penalty_area = get_passes_into_areas(df, team, home)

        for _, r in penalty_area.iterrows():
            pitch.arrows(r['start_x'], r['start_y'], r['end_x'], r['end_y'],
                        ax=ax, color=color, alpha=0.85, width=2,
                        headwidth=5, headlength=5, zorder=6)
            # End point marker
            ax.scatter(r['end_x'], r['end_y'], s=60, c=color, alpha=0.9,
                      edgecolors='white', linewidths=1, zorder=7)


# =============================================================================
# SEASON STATISTICS PAGE - PROFESSIONAL ANALYTICS
# =============================================================================

def calculate_ppda(actions, team):
    """Calculate Passes Per Defensive Action (PPDA) - lower = more intense press"""
    # Get all games this team played
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
        opp_passes = game_df[(game_df['team_name'] == opponent) &
                             (game_df['type_name'] == 'pass') &
                             (game_df['start_x'] < 60)]
        total_opp_passes += len(opp_passes)

        # Team's defensive actions in opponent's defensive 60%
        def_actions = game_df[(game_df['team_name'] == team) &
                              (game_df['type_name'].isin(DEFENSIVE_EVENTS)) &
                              (game_df['start_x'] > 40)]
        total_def_actions += len(def_actions)

    if total_def_actions == 0:
        return 0
    return total_opp_passes / total_def_actions


def calculate_team_advanced_stats(actions, team):
    """Calculate advanced stats for a team"""
    team_df = actions[actions['team_name'] == team]
    games = team_df['game_id'].nunique()

    # Basic counts
    passes = team_df[team_df['type_name'] == 'pass']
    shots = team_df[team_df['type_name'].isin(SHOT_EVENTS)]
    goals = len(shots[shots['goal_from_shot'] == True])

    # Progressive actions
    prog_passes = team_df[(team_df['type_name'].isin(ARROW_EVENTS)) &
                          (team_df.apply(is_progressive_advanced, axis=1))]
    prog_carries = team_df[(team_df['type_name'].isin(CARRY_EVENTS)) &
                           (team_df.apply(is_progressive_advanced, axis=1))]

    # Final third entries (estimate: passes ending x > 66.67)
    final_third_passes = passes[(passes['end_x'] > 66.67) & (passes['start_x'] <= 66.67) &
                                (passes['result_name'] == 'success')]

    # Box entries (passes ending x > 83, y 21-79)
    box_entries = passes[(passes['end_x'] > 83) & (passes['end_y'] >= 21) & (passes['end_y'] <= 79) &
                         (passes['start_x'] <= 83) & (passes['result_name'] == 'success')]

    # Defensive stats
    tackles = team_df[team_df['type_name'] == 'tackle']
    interceptions = team_df[team_df['type_name'] == 'interception']

    # High press actions (defensive actions in opponent's third)
    high_press = team_df[(team_df['type_name'].isin(DEFENSIVE_EVENTS)) & (team_df['start_x'] > 66.67)]

    return {
        'team': team,
        'games': games,
        'goals': goals,
        'shots': len(shots),
        'shots_per_game': len(shots) / games if games > 0 else 0,
        'pass_pct': (passes['result_name'] == 'success').mean() * 100 if len(passes) > 0 else 0,
        'prog_passes': len(prog_passes),
        'prog_passes_per_game': len(prog_passes) / games if games > 0 else 0,
        'prog_carries': len(prog_carries),
        'prog_carries_per_game': len(prog_carries) / games if games > 0 else 0,
        'final_third_entries': len(final_third_passes),
        'final_third_per_game': len(final_third_passes) / games if games > 0 else 0,
        'box_entries': len(box_entries),
        'box_entries_per_game': len(box_entries) / games if games > 0 else 0,
        'tackles': len(tackles),
        'interceptions': len(interceptions),
        'high_press_actions': len(high_press),
        'high_press_per_game': len(high_press) / games if games > 0 else 0,
    }


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
        # Estimate: add buffer before first and after last action
        start_min = max(0, (first_action / 60) - 3)
        end_min = min(95, (last_action / 60) + 3)
        total_minutes += (end_min - start_min)

    return total_minutes


def get_plotly_layout(title="", height=450):
    """Get consistent Plotly layout for dark theme"""
    return dict(
        title=dict(text=title, font=dict(size=14, color='#f5f5f7'), x=0, xanchor='left') if title else None,
        paper_bgcolor='#141416',
        plot_bgcolor='#0a0a0c',
        font=dict(family="Inter, sans-serif", color='#a1a1a6'),
        height=height,
        margin=dict(l=60, r=40, t=50 if title else 30, b=60),
        xaxis=dict(gridcolor='#1f1f23', zerolinecolor='#1f1f23', tickfont=dict(size=11)),
        yaxis=dict(gridcolor='#1f1f23', zerolinecolor='#1f1f23', tickfont=dict(size=11)),
        hoverlabel=dict(bgcolor='#141416', font_size=12, font_family="Inter, monospace"),
    )


def create_interactive_scatter(df, x_col, y_col, title, x_label, y_label, hover_name='player'):
    """Create interactive scatter plot with Plotly"""
    if len(df) == 0:
        return None

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        marker=dict(
            size=df['minutes'] / df['minutes'].max() * 25 + 8,
            color='#2997ff',
            opacity=0.75,
            line=dict(width=1, color='white')
        ),
        text=df[hover_name],
        customdata=np.stack((df['team'], df['minutes'], df[x_col], df[y_col]), axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Team: %{customdata[0]}<br>" +
            "Minutes: %{customdata[1]:.0f}<br>" +
            f"{x_label}: %{{customdata[2]:.2f}}<br>" +
            f"{y_label}: %{{customdata[3]:.2f}}<br>" +
            "<extra></extra>"
        ),
    ))

    # Annotate top performers
    for _, row in df.nlargest(5, x_col).iterrows():
        fig.add_annotation(
            x=row[x_col], y=row[y_col],
            text=row[hover_name].split()[-1][:10],
            showarrow=True, arrowhead=0, arrowsize=0.5,
            arrowcolor='#6e6e73', ax=15, ay=-15,
            font=dict(size=9, color='#f5f5f7'),
            bgcolor='rgba(20,20,22,0.8)',
        )

    # Median lines
    fig.add_hline(y=df[y_col].median(), line_dash="dash", line_color='#6e6e73', opacity=0.3)
    fig.add_vline(x=df[x_col].median(), line_dash="dash", line_color='#6e6e73', opacity=0.3)

    layout = get_plotly_layout(title, height=500)
    layout['xaxis']['title'] = x_label
    layout['yaxis']['title'] = y_label
    fig.update_layout(**layout)

    return fig


def show_season_stats(actions):
    """Display professional-grade season statistics with per 90 metrics"""

    # Sidebar filters
    with st.sidebar:
        st.markdown("### Filters")
        available_teams = sorted(actions['team_name'].dropna().unique())
        selected_teams = st.multiselect(
            "Select Teams",
            available_teams,
            default=available_teams
        )

    if not selected_teams:
        selected_teams = available_teams

    filtered = actions[actions['team_name'].isin(selected_teams)]

    # Header
    st.markdown("""
    <div style="padding: 1.5rem 0; border-bottom: 1px solid #1a1a1e; margin-bottom: 1.5rem;">
        <h1 style="font-size: 1.75rem; margin: 0; color: #f5f5f7; font-weight: 700; letter-spacing: -0.02em;">Season Analytics</h1>
        <p style="color: #6e6e73; margin-top: 0.4rem; font-size: 0.9rem;">Player and team statistics with per 90 normalization</p>
    </div>
    """, unsafe_allow_html=True)

    # Main tabs - 5 tabs now
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Player Rankings", "Team Analytics", "Progressive", "On Ball", "Scoring"])

    # ============ PLAYER RANKINGS TAB ============
    with tab1:
        # Controls row
        ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 1.5, 1, 1])

        with ctrl1:
            ranking_type = st.selectbox(
                "Rank By",
                [
                    "Goals",
                    "Shots",
                    "Passes",
                    "Pass Accuracy",
                    "Progressive Passes",
                    "Progressive Carries",
                    "Tackles",
                    "Interceptions",
                    "Dribbles",
                    "Crosses",
                    "Final Third Passes",
                    "Box Entries",
                    "Defensive Actions"
                ]
            )

        with ctrl2:
            min_minutes = st.slider("Min. Minutes", 0, 600, 90, step=30)

        with ctrl3:
            per_90 = st.toggle("Per 90", value=False)

        with ctrl4:
            show_table = st.toggle("Table View", value=False)

        # Calculate comprehensive player stats
        player_data = []
        for player_id in filtered['player_id'].dropna().unique():
            p = filtered[filtered['player_id'] == player_id]
            name = p['player_name'].iloc[0]
            team = p['team_name'].iloc[0]
            games = p['game_id'].nunique()

            # Estimate minutes
            minutes = estimate_player_minutes(filtered, player_id)
            nineties = minutes / 90 if minutes > 0 else 0.01

            passes = p[p['type_name'] == 'pass']
            shots_df = p[p['type_name'].isin(SHOT_EVENTS)]
            goals = len(shots_df[shots_df['goal_from_shot'] == True])

            prog_p = len(p[(p['type_name'].isin(ARROW_EVENTS)) & (p.apply(is_progressive_advanced, axis=1))])
            prog_c = len(p[(p['type_name'].isin(CARRY_EVENTS)) & (p.apply(is_progressive_advanced, axis=1))])

            tackles = len(p[p['type_name'] == 'tackle'])
            interceptions = len(p[p['type_name'] == 'interception'])
            clearances = len(p[p['type_name'] == 'clearance'])

            dribbles = p[p['type_name'].isin(['dribble', 'take_on'])]
            crosses = p[p['type_name'] == 'cross']

            # Final third and box entries
            ft_passes = len(passes[(passes['end_x'] > 66.67) & (passes['start_x'] <= 66.67) & (passes['result_name'] == 'success')])
            box_entries = len(passes[(passes['end_x'] > 83) & (passes['end_y'] >= 21) & (passes['end_y'] <= 79) & (passes['start_x'] <= 83) & (passes['result_name'] == 'success')])

            player_data.append({
                'player': name,
                'team': team,
                'games': games,
                'minutes': round(minutes),
                'nineties': round(nineties, 2),
                'goals': goals,
                'shots': len(shots_df),
                'passes': len(passes),
                'pass_pct': (passes['result_name'] == 'success').mean() * 100 if len(passes) > 0 else 0,
                'prog_passes': prog_p,
                'prog_carries': prog_c,
                'tackles': tackles,
                'interceptions': interceptions,
                'clearances': clearances,
                'dribbles': len(dribbles),
                'crosses': len(crosses),
                'final_third': ft_passes,
                'box_entries': box_entries,
                'defensive': tackles + interceptions + clearances,
            })

        player_df = pd.DataFrame(player_data)

        # Filter by minutes
        player_df = player_df[player_df['minutes'] >= min_minutes]

        # Map ranking type to column
        sort_map = {
            "Goals": "goals",
            "Shots": "shots",
            "Passes": "passes",
            "Pass Accuracy": "pass_pct",
            "Progressive Passes": "prog_passes",
            "Progressive Carries": "prog_carries",
            "Tackles": "tackles",
            "Interceptions": "interceptions",
            "Dribbles": "dribbles",
            "Crosses": "crosses",
            "Final Third Passes": "final_third",
            "Box Entries": "box_entries",
            "Defensive Actions": "defensive",
        }
        sort_col = sort_map[ranking_type]

        # Calculate per 90 if needed
        display_df = player_df.copy()
        if per_90:
            per90_cols = ['goals', 'shots', 'passes', 'prog_passes', 'prog_carries',
                         'tackles', 'interceptions', 'clearances', 'dribbles', 'crosses',
                         'final_third', 'box_entries', 'defensive']
            for col in per90_cols:
                display_df[col] = (display_df[col] / display_df['nineties']).round(2)

        display_df = display_df.sort_values(sort_col, ascending=False).head(25)
        suffix = " /90" if per_90 else ""

        # Summary stats row
        st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin: 1rem 0 1.5rem 0; flex-wrap: wrap;">
            <div style="background: #141416; border: 1px solid #1f1f23; border-radius: 6px; padding: 1rem 1.25rem; min-width: 120px;">
                <div style="font-size: 0.65rem; color: #6e6e73; text-transform: uppercase; letter-spacing: 0.08em;">Players</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #f5f5f7; font-family: monospace;">{len(display_df)}</div>
            </div>
            <div style="background: #141416; border: 1px solid #1f1f23; border-radius: 6px; padding: 1rem 1.25rem; min-width: 120px;">
                <div style="font-size: 0.65rem; color: #6e6e73; text-transform: uppercase; letter-spacing: 0.08em;">Avg Minutes</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #2997ff; font-family: monospace;">{player_df['minutes'].mean():.0f}</div>
            </div>
            <div style="background: #141416; border: 1px solid #1f1f23; border-radius: 6px; padding: 1rem 1.25rem; min-width: 120px;">
                <div style="font-size: 0.65rem; color: #6e6e73; text-transform: uppercase; letter-spacing: 0.08em;">Avg 90s</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #30d158; font-family: monospace;">{player_df['nineties'].mean():.1f}</div>
            </div>
            <div style="background: #141416; border: 1px solid #1f1f23; border-radius: 6px; padding: 1rem 1.25rem; min-width: 120px;">
                <div style="font-size: 0.65rem; color: #6e6e73; text-transform: uppercase; letter-spacing: 0.08em;">Total Goals</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #ff9f0a; font-family: monospace;">{player_df['goals'].sum()}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if show_table:
            # Table view
            table_df = display_df[['player', 'team', 'minutes', 'nineties', 'goals', 'shots', 'passes',
                                   'pass_pct', 'prog_passes', 'prog_carries', 'tackles', 'interceptions']].copy()
            table_df.columns = ['Player', 'Team', 'Mins', '90s', f'G{suffix}', f'Sh{suffix}', f'Pass{suffix}',
                               'Pass%', f'ProgP{suffix}', f'ProgC{suffix}', f'Tkl{suffix}', f'Int{suffix}']
            table_df['Pass%'] = table_df['Pass%'].round(1)
            st.dataframe(table_df, use_container_width=True, hide_index=True, height=600)
        else:
            # Card view
            for idx, (_, row) in enumerate(display_df.iterrows()):
                val = row[sort_col]
                val_display = f"{val:.2f}" if per_90 else str(int(val)) if sort_col != 'pass_pct' else f"{val:.1f}%"

                st.markdown(f"""
                <div style="background: #141416; border: 1px solid #1f1f23; border-radius: 6px; padding: 0.875rem 1rem; margin-bottom: 0.4rem; display: flex; align-items: center; justify-content: space-between;">
                    <div style="display: flex; align-items: center; gap: 1rem;">
                        <div style="background: {'#ff9f0a' if idx < 3 else '#1f1f23'}; color: {'#000' if idx < 3 else '#a1a1a6'}; width: 26px; height: 26px; border-radius: 4px; display: flex; align-items: center; justify-content: center; font-weight: 600; font-size: 0.8rem; font-family: monospace;">
                            {idx + 1}
                        </div>
                        <div>
                            <div style="font-weight: 600; color: #f5f5f7; font-size: 0.95rem;">{row['player']}</div>
                            <div style="font-size: 0.75rem; color: #6e6e73;">{row['team']} | {row['minutes']} min ({row['nineties']:.1f} x 90)</div>
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="font-size: 1.25rem; font-weight: 600; color: #2997ff; font-family: monospace;">{val_display}</div>
                        <div style="font-size: 0.65rem; color: #6e6e73;">{ranking_type}{suffix}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ============ TEAM ANALYTICS TAB ============
    with tab2:
        # Calculate stats for all teams
        team_stats = []
        for team in selected_teams:
            stats = calculate_team_advanced_stats(filtered, team)
            ppda = calculate_ppda(actions, team)
            stats['ppda'] = ppda
            team_stats.append(stats)

        team_df = pd.DataFrame(team_stats)

        team_per_game = st.toggle("Show Per Game", value=True, key="team_pg")

        # Summary cards
        st.markdown(f"""
        <div style="display: flex; gap: 1rem; margin: 1rem 0 1.5rem 0; flex-wrap: wrap;">
            <div style="background: #141416; border: 1px solid #1f1f23; border-radius: 6px; padding: 1rem 1.25rem;">
                <div style="font-size: 0.65rem; color: #6e6e73; text-transform: uppercase;">Total Goals</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #30d158; font-family: monospace;">{team_df['goals'].sum()}</div>
            </div>
            <div style="background: #141416; border: 1px solid #1f1f23; border-radius: 6px; padding: 1rem 1.25rem;">
                <div style="font-size: 0.65rem; color: #6e6e73; text-transform: uppercase;">Total Shots</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #f5f5f7; font-family: monospace;">{team_df['shots'].sum()}</div>
            </div>
            <div style="background: #141416; border: 1px solid #1f1f23; border-radius: 6px; padding: 1rem 1.25rem;">
                <div style="font-size: 0.65rem; color: #6e6e73; text-transform: uppercase;">Avg Pass%</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #2997ff; font-family: monospace;">{team_df['pass_pct'].mean():.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Team comparison table
        team_display = team_df.copy()
        suffix = " /G" if team_per_game else ""

        if team_per_game:
            for col in ['shots', 'prog_passes', 'prog_carries', 'final_third_entries', 'box_entries', 'tackles', 'interceptions', 'high_press_actions']:
                if col in team_display.columns:
                    team_display[col] = (team_display[col] / team_display['games']).round(1)
                col_pg = col + '_per_game'
                if col_pg in team_display.columns:
                    pass  # Already per game

        comparison_df = team_display[['team', 'games', 'goals', 'shots' if not team_per_game else 'shots_per_game',
                                      'pass_pct', 'prog_passes_per_game', 'prog_carries_per_game',
                                      'box_entries_per_game', 'ppda', 'high_press_per_game']].copy()
        comparison_df.columns = ['Team', 'Games', 'Goals', f'Shots{suffix}', 'Pass%', f'ProgP{suffix}',
                                f'ProgC{suffix}', f'Box{suffix}', 'PPDA', f'HiPress{suffix}']
        comparison_df = comparison_df.round(2).sort_values('Goals', ascending=False)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        # Charts
        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Pressing Intensity (PPDA)**")
            ppda_df = team_df[['team', 'ppda']].sort_values('ppda')

            fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0a0a0c')
            ax.set_facecolor('#0a0a0c')

            colors = ['#30d158' if x < 10 else '#ff9f0a' if x < 15 else '#ff453a' for x in ppda_df['ppda']]
            ax.barh(range(len(ppda_df)), ppda_df['ppda'], color=colors, alpha=0.9)

            ax.set_yticks(range(len(ppda_df)))
            ax.set_yticklabels(ppda_df['team'], color='#f5f5f7', fontsize=9)
            ax.tick_params(axis='x', colors='#6e6e73')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.grid(True, color='#1f1f23', alpha=0.5)
            ax.axvline(10, color='#30d158', linestyle='--', alpha=0.4, linewidth=1)
            ax.axvline(15, color='#ff9f0a', linestyle='--', alpha=0.4, linewidth=1)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col2:
            st.markdown("**Goals by Team**")
            goals_df = team_df[['team', 'goals']].sort_values('goals', ascending=True)

            fig, ax = plt.subplots(figsize=(10, 5), facecolor='#0a0a0c')
            ax.set_facecolor('#0a0a0c')

            ax.barh(range(len(goals_df)), goals_df['goals'], color='#30d158', alpha=0.9)

            ax.set_yticks(range(len(goals_df)))
            ax.set_yticklabels(goals_df['team'], color='#f5f5f7', fontsize=9)
            ax.tick_params(axis='x', colors='#6e6e73')
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.xaxis.grid(True, color='#1f1f23', alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    # ============ PROGRESSIVE ANALYSIS TAB ============
    with tab3:
        min_mins_prog = st.slider("Minimum Minutes", 0, 600, 180, step=30, key="prog_mins")
        per_90_prog = st.toggle("Show Per 90", value=True, key="prog_p90")

        prog_player_df = player_df[player_df['minutes'] >= min_mins_prog].copy()

        if per_90_prog:
            prog_player_df['prog_passes_p90'] = (prog_player_df['prog_passes'] / prog_player_df['nineties']).round(2)
            prog_player_df['prog_carries_p90'] = (prog_player_df['prog_carries'] / prog_player_df['nineties']).round(2)
            pass_col, carry_col = 'prog_passes_p90', 'prog_carries_p90'
            suffix = " /90"
        else:
            pass_col, carry_col = 'prog_passes', 'prog_carries'
            suffix = ""

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"**Top Progressive Passers{suffix}**")
            top_pp = prog_player_df.nlargest(12, pass_col)[['player', 'team', 'minutes', pass_col, carry_col]]
            top_pp.columns = ['Player', 'Team', 'Mins', f'ProgP{suffix}', f'ProgC{suffix}']
            st.dataframe(top_pp, use_container_width=True, hide_index=True)

        with col2:
            st.markdown(f"**Top Progressive Carriers{suffix}**")
            top_pc = prog_player_df.nlargest(12, carry_col)[['player', 'team', 'minutes', pass_col, carry_col]]
            top_pc.columns = ['Player', 'Team', 'Mins', f'ProgP{suffix}', f'ProgC{suffix}']
            st.dataframe(top_pc, use_container_width=True, hide_index=True)

        st.markdown("---")

        # Interactive Plotly scatter plot
        if len(prog_player_df) > 0:
            st.markdown(f"**Progressive Passes vs Carries{suffix}** (hover for player info)")

            fig = create_interactive_scatter(
                prog_player_df, pass_col, carry_col,
                "", f"Progressive Passes{suffix}", f"Progressive Carries{suffix}"
            )
            if fig:
                st.plotly_chart(fig, use_container_width=True)

        # Team progressive actions chart with Plotly
        if len(team_df) > 0:
            st.markdown("---")
            st.markdown("**Team Progressive Actions per Game**")

            prog_team_df = team_df[['team', 'prog_passes_per_game', 'prog_carries_per_game']].copy()
            prog_team_df = prog_team_df.sort_values('prog_passes_per_game', ascending=False)

            fig = go.Figure()
            fig.add_trace(go.Bar(x=prog_team_df['team'], y=prog_team_df['prog_passes_per_game'],
                                 name='Prog Passes', marker_color='#2997ff', opacity=0.9))
            fig.add_trace(go.Bar(x=prog_team_df['team'], y=prog_team_df['prog_carries_per_game'],
                                 name='Prog Carries', marker_color='#30d158', opacity=0.9))

            layout = get_plotly_layout("", height=400)
            layout['barmode'] = 'group'
            layout['xaxis']['tickangle'] = -45
            layout['legend'] = dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1,
                                    font=dict(size=10), bgcolor='rgba(20,20,22,0.8)')
            fig.update_layout(**layout)
            st.plotly_chart(fig, use_container_width=True)

    # ============ ON BALL ANALYSIS TAB ============
    with tab4:
        st.markdown("""
        <p style="color: #6e6e73; font-size: 0.85rem; margin-bottom: 1rem;">
            Analyze ball progression into dangerous areas - Final Third and Penalty Box entries via passes and carries.
        </p>
        """, unsafe_allow_html=True)

        min_mins_ob = st.slider("Minimum Minutes", 0, 600, 180, step=30, key="ob_mins")
        per_90_ob = st.toggle("Show Per 90", value=True, key="ob_p90")

        # Calculate on-ball metrics for each player
        ob_data = []
        for player_id in filtered['player_id'].dropna().unique():
            p = filtered[filtered['player_id'] == player_id]
            name = p['player_name'].iloc[0]
            team = p['team_name'].iloc[0]
            minutes = estimate_player_minutes(filtered, player_id)

            if minutes < min_mins_ob:
                continue

            nineties = minutes / 90 if minutes > 0 else 0.01
            passes = p[p['type_name'] == 'pass']

            # Final third entries
            ft_passes = len(passes[(passes['end_x'] > 66.67) & (passes['start_x'] <= 66.67) & (passes['result_name'] == 'success')])
            ft_carries = len(p[(p['type_name'].isin(CARRY_EVENTS)) & (p['start_x'] <= 66.67) & (p['end_x'] > 66.67) & (p['result_name'] == 'success')])

            # Box entries
            box_passes = len(passes[(passes['end_x'] > 83) & (passes['end_y'] >= 21) & (passes['end_y'] <= 79) & (passes['start_x'] <= 83) & (passes['result_name'] == 'success')])
            box_carries = len(p[(p['type_name'].isin(CARRY_EVENTS)) & (p['start_x'] <= 83) & (p['end_x'] > 83) & (p['end_y'] >= 21) & (p['end_y'] <= 79) & (p['result_name'] == 'success')])

            ob_data.append({
                'player': name, 'team': team, 'minutes': round(minutes), 'nineties': round(nineties, 2),
                'ft_passes': ft_passes, 'ft_carries': ft_carries,
                'box_passes': box_passes, 'box_carries': box_carries,
            })

        ob_df = pd.DataFrame(ob_data)

        if len(ob_df) > 0:
            if per_90_ob:
                ob_df['ft_passes_p90'] = (ob_df['ft_passes'] / ob_df['nineties']).round(2)
                ob_df['ft_carries_p90'] = (ob_df['ft_carries'] / ob_df['nineties']).round(2)
                ob_df['box_passes_p90'] = (ob_df['box_passes'] / ob_df['nineties']).round(2)
                ob_df['box_carries_p90'] = (ob_df['box_carries'] / ob_df['nineties']).round(2)
                ft_pass_col, ft_carry_col = 'ft_passes_p90', 'ft_carries_p90'
                box_pass_col, box_carry_col = 'box_passes_p90', 'box_carries_p90'
                suffix = " /90"
            else:
                ft_pass_col, ft_carry_col = 'ft_passes', 'ft_carries'
                box_pass_col, box_carry_col = 'box_passes', 'box_carries'
                suffix = ""

            # Final Third section
            st.markdown("### Final Third Entries")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Top Final Third Passers{suffix}**")
                ft_passers = ob_df.nlargest(10, ft_pass_col)[['player', 'team', 'minutes', ft_pass_col, ft_carry_col]]
                ft_passers.columns = ['Player', 'Team', 'Mins', f'FT Pass{suffix}', f'FT Carry{suffix}']
                st.dataframe(ft_passers, use_container_width=True, hide_index=True)

            with col2:
                st.markdown(f"**Top Final Third Carriers{suffix}**")
                ft_carriers = ob_df.nlargest(10, ft_carry_col)[['player', 'team', 'minutes', ft_pass_col, ft_carry_col]]
                ft_carriers.columns = ['Player', 'Team', 'Mins', f'FT Pass{suffix}', f'FT Carry{suffix}']
                st.dataframe(ft_carriers, use_container_width=True, hide_index=True)

            # Final Third scatter
            st.markdown(f"**Final Third: Passes vs Carries{suffix}** (hover for details)")
            fig = create_interactive_scatter(ob_df, ft_pass_col, ft_carry_col, "", f"FT Passes{suffix}", f"FT Carries{suffix}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Box Entries section
            st.markdown("### Penalty Box Entries")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown(f"**Top Box Passers{suffix}**")
                box_passers = ob_df.nlargest(10, box_pass_col)[['player', 'team', 'minutes', box_pass_col, box_carry_col]]
                box_passers.columns = ['Player', 'Team', 'Mins', f'Box Pass{suffix}', f'Box Carry{suffix}']
                st.dataframe(box_passers, use_container_width=True, hide_index=True)

            with col2:
                st.markdown(f"**Top Box Carriers{suffix}**")
                box_carriers = ob_df.nlargest(10, box_carry_col)[['player', 'team', 'minutes', box_pass_col, box_carry_col]]
                box_carriers.columns = ['Player', 'Team', 'Mins', f'Box Pass{suffix}', f'Box Carry{suffix}']
                st.dataframe(box_carriers, use_container_width=True, hide_index=True)

            # Box entries scatter
            st.markdown(f"**Penalty Box: Passes vs Carries{suffix}** (hover for details)")
            fig = create_interactive_scatter(ob_df, box_pass_col, box_carry_col, "", f"Box Passes{suffix}", f"Box Carries{suffix}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            # Combined profile
            st.markdown("### Combined On Ball Profile")
            ob_df['total_ft'] = ob_df[ft_pass_col] + ob_df[ft_carry_col]
            ob_df['total_box'] = ob_df[box_pass_col] + ob_df[box_carry_col]

            fig = create_interactive_scatter(ob_df, 'total_ft', 'total_box', "", f"Total Final Third{suffix}", f"Total Box{suffix}")
            if fig:
                st.plotly_chart(fig, use_container_width=True)

    # ============ SCORING TAB ============
    with tab5:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Top Scorers**")
            shots = filtered[filtered['type_name'].isin(SHOT_EVENTS)]
            goals = shots[shots['goal_from_shot'] == True]

            scorer_stats = goals.groupby(['player_name', 'team_name']).agg({
                'action_id': 'count',
                'game_id': 'nunique'
            }).reset_index()
            scorer_stats.columns = ['Player', 'Team', 'Goals', 'Games']
            scorer_stats = scorer_stats.sort_values('Goals', ascending=False).head(15)

            for idx, (_, row) in enumerate(scorer_stats.iterrows()):
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; padding: 0.6rem 0; border-bottom: 1px solid #1f1f23;">
                    <div>
                        <span style="color: {'#ff9f0a' if idx < 3 else '#6e6e73'}; font-weight: 600; margin-right: 0.5rem; font-family: monospace;">{idx + 1}.</span>
                        <span style="color: #f5f5f7; font-weight: 500;">{row['Player']}</span>
                        <span style="color: #6e6e73; font-size: 0.8rem;"> ({row['Team']})</span>
                    </div>
                    <div style="color: #30d158; font-weight: 600; font-size: 1rem; font-family: monospace;">{row['Goals']}</div>
                </div>
                """, unsafe_allow_html=True)

        with col2:
            st.markdown("**Shot Conversion Rate**")
            conv_df = team_df[['team', 'goals', 'shots']].copy()
            conv_df['conversion'] = (conv_df['goals'] / conv_df['shots'] * 100).round(1)
            conv_df = conv_df.sort_values('conversion', ascending=False)

            fig, ax = plt.subplots(figsize=(10, 6), facecolor='#0a0a0c')
            ax.set_facecolor('#0a0a0c')

            colors = ['#30d158' if x > 15 else '#ff9f0a' if x > 10 else '#ff453a' for x in conv_df['conversion']]
            ax.barh(range(len(conv_df)), conv_df['conversion'], color=colors, alpha=0.9)

            ax.set_yticks(range(len(conv_df)))
            ax.set_yticklabels(conv_df['team'], color='#f5f5f7', fontsize=9)
            ax.tick_params(axis='x', colors='#6e6e73')

            for spine in ax.spines.values():
                spine.set_visible(False)

            ax.xaxis.grid(True, color='#1f1f23', alpha=0.5)
            ax.axvline(10, color='#ff9f0a', linestyle='--', alpha=0.4)
            ax.axvline(15, color='#30d158', linestyle='--', alpha=0.4)

            # Value labels
            for i, v in enumerate(conv_df['conversion']):
                ax.text(v + 0.5, i, f'{v}%', va='center', fontsize=8, color='#a1a1a6')

            ax.set_xlabel('Conversion %', color='#6e6e73', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")

        # Goals vs Shots scatter
        st.markdown("**Goals vs Shots (Player)**")

        min_shots = st.slider("Minimum Shots", 1, 20, 3, key="min_shots")
        shooter_df = player_df[(player_df['shots'] >= min_shots) & (player_df['minutes'] >= 90)].copy()

        if len(shooter_df) > 0:
            fig, ax = plt.subplots(figsize=(12, 6), facecolor='#0a0a0c')
            ax.set_facecolor('#0a0a0c')

            ax.scatter(shooter_df['shots'], shooter_df['goals'],
                      s=shooter_df['minutes']/3, c='#2997ff', alpha=0.7,
                      edgecolors='white', linewidths=0.5)

            # Annotate top scorers
            for _, row in shooter_df.nlargest(5, 'goals').iterrows():
                ax.annotate(row['player'].split()[-1][:10],
                           (row['shots'], row['goals']),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, color='#f5f5f7')

            ax.set_xlabel('Shots', fontsize=10, color='#a1a1a6')
            ax.set_ylabel('Goals', fontsize=10, color='#a1a1a6')
            ax.tick_params(colors='#6e6e73')

            for spine in ax.spines.values():
                spine.set_color('#1f1f23')

            ax.xaxis.grid(True, color='#1f1f23', alpha=0.5)
            ax.yaxis.grid(True, color='#1f1f23', alpha=0.5)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    if 'sel_idx' not in st.session_state:
        st.session_state.sel_idx = None

    # Sidebar
    with st.sidebar:
        st.markdown("## Matchday Analytics")

        # Page selector
        page = st.radio(
            "Navigate",
            ["Match Analysis", "Season Statistics"],
            label_visibility="collapsed"
        )

        st.markdown("---")

        actions, matches = load_data()

    # Route to different pages
    if page == "Season Statistics":
        show_season_stats(actions)
        return

    # Continue with Match Analysis
    with st.sidebar:

        st.markdown('<p class="section-label">Match</p>', unsafe_allow_html=True)
        match = st.selectbox("Match", list(matches.values()), label_visibility="collapsed")
        gid = [k for k, v in matches.items() if v == match][0]
        mdf = actions[actions['game_id'] == gid].copy()

        teams = mdf['team_name'].unique()
        home, away = teams[0], teams[1] if len(teams) > 1 else "Away"

        st.markdown('<p class="section-label">Time Range</p>', unsafe_allow_html=True)
        max_min = int(mdf['minute'].max()) + 1
        mins = st.slider("Minutes", 0, max_min, (0, max_min), label_visibility="collapsed")

        fdf = mdf[(mdf['minute'] >= mins[0]) & (mdf['minute'] <= mins[1])].copy()
        all_types = sorted(mdf['type_name'].unique())
        sel_types = all_types  # Default

        with st.expander("Filters", expanded=False):
            sel_types = st.multiselect("Event Types", all_types, default=all_types)

        if sel_types:
            fdf = fdf[fdf['type_name'].isin(sel_types)]

        with st.expander("Team & Player", expanded=False):
            team_opt = st.radio("Team", ["Both", home, away], horizontal=True)
            if team_opt == home:
                fdf = fdf[fdf['team_name'] == home]
            elif team_opt == away:
                fdf = fdf[fdf['team_name'] == away]

            players = sorted(fdf['player_name'].unique())
            sel_players = st.multiselect("Players", players)
            if sel_players:
                fdf = fdf[fdf['player_name'].isin(sel_players)]

        c1, c2 = st.columns(2)
        arrows = c1.checkbox("Arrows", True)
        numbers = c2.checkbox("Numbers", True)

        st.markdown("---")
        st.markdown("""
        <div style="display: flex; gap: 0.75rem; flex-wrap: wrap;">
            <a href="https://twitter.com/yureedelahi" target="_blank" style="
                font-size: 0.75rem; color: #a1a1aa; text-decoration: none;
            ">𝕏 @yureedelahi</a>
            <a href="https://yureedelahi.substack.com/" target="_blank" style="
                font-size: 0.75rem; color: #a1a1aa; text-decoration: none;
            ">Substack</a>
        </div>
        """, unsafe_allow_html=True)

    # Get team IDs for own goal handling
    home_id = mdf[mdf['team_name'] == home]['team_id'].iloc[0] if len(mdf[mdf['team_name'] == home]) > 0 else None
    away_id = mdf[mdf['team_name'] == away]['team_id'].iloc[0] if len(mdf[mdf['team_name'] == away]) > 0 else None

    # Process data
    fdf = flip_coords(fdf, away).sort_values('action_id').reset_index(drop=True)
    full_match_flipped = flip_coords(mdf, away)  # Full match for momentum

    # IMPORTANT: Calculate stats from FILTERED data (fdf), not full match (mdf)
    stats = compute_stats(fdf, home, away, home_id, away_id)

    # Match Header - Scoreboard Style with stats that update with time filter
    st.markdown(f"""
    <div class="match-header-card">
        <div class="match-teams">
            <div class="team-info home">
                <h2 class="team-name-display home">{home}</h2>
            </div>
            <div class="score-center">
                <span class="score-num">{stats['goals_h']}</span>
                <span class="score-divider">-</span>
                <span class="score-num">{stats['goals_a']}</span>
            </div>
            <div class="team-info away">
                <h2 class="team-name-display away">{away}</h2>
            </div>
        </div>
        <div class="match-meta">{mins[0]}' — {mins[1]}' · {len(fdf)} events</div>
        <div class="stats-row">
            <div class="stat-box">
                <div class="stat-values">
                    <span class="stat-val home">{stats['shots_h']}</span>
                    <span class="stat-val away">{stats['shots_a']}</span>
                </div>
                <div class="stat-name">Shots</div>
            </div>
            <div class="stat-box">
                <div class="stat-values">
                    <span class="stat-val home">{stats['pass_pct_h']:.0f}%</span>
                    <span class="stat-val away">{stats['pass_pct_a']:.0f}%</span>
                </div>
                <div class="stat-name">Pass Accuracy</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Match Report Button (prominent at top)
    report_col1, report_col2, report_col3 = st.columns([1, 2, 1])
    with report_col2:
        if st.button("GENERATE MATCH REPORT", use_container_width=True, type="primary"):
            with st.spinner("Generating comprehensive match report..."):
                full_stats = compute_stats(mdf, home, away, home_id, away_id)  # Use full match for report
                report_fig = generate_match_report(mdf, home, away, full_stats, home_id, away_id)
                report_buf = fig_to_buffer(report_fig, dpi=250)
                plt.close(report_fig)
                st.session_state.report_buf = report_buf
                st.session_state.report_name = f"{home}_vs_{away}_match_report.png"
                st.success("Match report generated. Download button below.")

    # Show report download if generated
    if 'report_buf' in st.session_state:
        with report_col2:
            st.download_button(
                "DOWNLOAD REPORT",
                st.session_state.report_buf,
                st.session_state.report_name,
                "image/png",
                use_container_width=True,
                type="secondary"
            )

    st.markdown("---")

    # View Mode Selector - Clean horizontal tabs
    if 'view_mode' not in st.session_state:
        st.session_state.view_mode = "Sequence"

    mode = st.radio(
        "Analysis View",
        ["Sequence", "Shots", "Pass Network", "Heatmap", "Progressive", "Pressing", "Final Third", "Box Entry", "Momentum"],
        horizontal=True,
        index=["Sequence", "Shots", "Pass Network", "Heatmap", "Progressive", "Pressing", "Final Third", "Box Entry", "Momentum"].index(st.session_state.view_mode)
    )

    if mode != st.session_state.view_mode:
        st.session_state.view_mode = mode
        st.rerun()

    # Team selector for modes that need it
    selected_team = 'home'
    if mode in ["Pass Network", "Heatmap", "Pressing"]:
        team_sel_col1, team_sel_col2 = st.columns([1, 3])
        with team_sel_col1:
            team_choice = st.radio("Show Team", [home, away], horizontal=True, label_visibility="collapsed")
            selected_team = 'home' if team_choice == home else 'away'

    # Playback controls for Sequence mode
    if mode == "Sequence" and len(fdf) > 0:
        ctrl_cols = st.columns([1, 1, 1, 3])
        with ctrl_cols[0]:
            if st.button("◂ PREV", disabled=st.session_state.sel_idx is None or st.session_state.sel_idx <= 0):
                st.session_state.sel_idx = max(0, st.session_state.sel_idx - 1)
        with ctrl_cols[1]:
            if st.button("NEXT ▸", disabled=st.session_state.sel_idx is not None and st.session_state.sel_idx >= len(fdf) - 1):
                st.session_state.sel_idx = 0 if st.session_state.sel_idx is None else min(len(fdf) - 1, st.session_state.sel_idx + 1)
        with ctrl_cols[2]:
            if st.button("CLEAR"):
                st.session_state.sel_idx = None
        with ctrl_cols[3]:
            if st.session_state.sel_idx is not None:
                st.markdown(f'<span class="event-counter">Event <strong>{st.session_state.sel_idx + 1}</strong> of {len(fdf)}</span>', unsafe_allow_html=True)

    # Main visualization
    main_col, side_col = st.columns([4, 1])

    with main_col:
        time_txt = f"Minute {mins[0]}" if mins[0] == mins[1] else f"Minutes {mins[0]}-{mins[1]}"
        title = f"{home} vs {away}"
        sub = f"{time_txt} · {len(fdf)} events"
        if sel_types and len(sel_types) < len(all_types):
            sub += f" · {', '.join(sel_types[:2])}" + (f" +{len(sel_types)-2}" if len(sel_types) > 2 else "")

        # Map UI modes to internal modes
        mode_map = {
            "Sequence": "sequence",
            "Shots": "shots",
            "Pass Network": "pass_network",
            "Heatmap": "heatmap",
            "Progressive": "progressive",
            "Pressing": "pressing",
            "Final Third": "final_third",
            "Box Entry": "penalty_area",
            "Momentum": "momentum"
        }

        fig = create_viz(
            fdf, home, away, title, sub,
            mode=mode_map.get(mode, "sequence"),
            sel_idx=st.session_state.sel_idx if mode == "Sequence" else None,
            arrows=arrows,
            numbers=numbers,
            selected_team=selected_team,
            full_match_df=full_match_flipped
        )

        st.pyplot(fig, use_container_width=True)

        # Simple download for current view
        buf = fig_to_buffer(fig)
        fname = f"{home}_vs_{away}_{mins[0]}-{mins[1]}min_{mode.lower().replace(' ', '_')}.png".replace(" ", "_")
        st.download_button("DOWNLOAD VIEW", buf, fname, "image/png", use_container_width=True)

        plt.close(fig)

    with side_col:
        st.markdown('<div class="panel-header">Event Details</div>', unsafe_allow_html=True)

        if len(fdf) > 0 and mode == "Sequence":
            opts = ["— Show All —"] + [f"{i+1}. {r['time_display']} {r['type_name']}" for i, r in fdf.iterrows()]

            if st.session_state.sel_idx is None:
                default = 0
            else:
                default = st.session_state.sel_idx + 1

            sel_str = st.selectbox("Event", opts, index=min(default, len(opts) - 1), label_visibility="collapsed")

            if sel_str == "— Show All —":
                if st.session_state.sel_idx is not None:
                    st.session_state.sel_idx = None
                    st.rerun()
            else:
                idx = opts.index(sel_str) - 1
                if st.session_state.sel_idx != idx:
                    st.session_state.sel_idx = idx
                    st.rerun()

            if st.session_state.sel_idx is None:
                st.markdown(f'<p class="mode-stats"><strong>{len(fdf)}</strong> events in view</p>', unsafe_allow_html=True)
                st.markdown('<p class="event-detail">Select an event to see details</p>', unsafe_allow_html=True)
            else:
                e = fdf.iloc[st.session_state.sel_idx]
                badge_cls = "home" if e['team_name'] == home else "away"
                st.markdown(f'<span class="team-badge {badge_cls}">{e["team_name"]}</span>', unsafe_allow_html=True)
                st.markdown(f'<p class="player-name">{e["player_name"]}</p>', unsafe_allow_html=True)
                st.markdown(f'<p class="event-meta">{e["time_display"]} · {e["type_name"].replace("_", " ").title()}</p>', unsafe_allow_html=True)

                res_cls = "success" if e['result_name'] == 'success' else "fail"
                st.markdown(f'<span class="result-badge {res_cls}">{e["result_name"]}</span>', unsafe_allow_html=True)
                st.markdown(f'<p class="event-detail">Body: {e["bodypart_name"].title()}</p>', unsafe_allow_html=True)

                if pd.notna(e.get('xT')) and e['xT'] != 0:
                    color = COLORS['success'] if e['xT'] > 0 else COLORS['home']
                    st.markdown(f'<p class="event-detail">xT: <span style="color:{color}">{e["xT"]:.3f}</span></p>', unsafe_allow_html=True)

                if e['type_name'] in SHOT_EVENTS and e.get('goal_from_shot', False):
                    st.markdown('<p class="goal-flash">GOAL!</p>', unsafe_allow_html=True)

                with st.expander("Context"):
                    idx = st.session_state.sel_idx
                    for i in range(max(0, idx - 2), min(len(fdf), idx + 3)):
                        r = fdf.iloc[i]
                        active = "active" if i == idx else ""
                        st.markdown(f'<div class="context-item {active}">{i+1}. {r["time_display"]} {r["team_name"][:3]} — {r["type_name"]}</div>', unsafe_allow_html=True)

        elif mode == "Shots":
            shots = fdf[fdf['type_name'].isin(SHOT_EVENTS)]
            goals = shots[shots['goal_from_shot'] == True]
            st.markdown(f'<p class="mode-stats"><strong>{len(shots)}</strong> shots · <strong>{len(goals)}</strong> goals</p>', unsafe_allow_html=True)
            for t in [home, away]:
                ts = shots[shots['team_name'] == t]
                tg = ts[ts['goal_from_shot'] == True]
                c = COLORS['home'] if t == home else COLORS['away']
                st.markdown(f'<p class="team-stat-line" style="color:{c}">{t}: {len(ts)} shots ({len(tg)}G)</p>', unsafe_allow_html=True)

        elif mode == "Pass Network":
            team = home if selected_team == 'home' else away
            team_passes = fdf[(fdf['team_name'] == team) & (fdf['type_name'].isin(ARROW_EVENTS))]
            succ = team_passes[team_passes['result_name'] == 'success']
            players = team_passes['player_name'].nunique()
            st.markdown(f'<p class="mode-stats"><strong>{len(succ)}</strong> completed passes</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="event-detail">{players} players involved</p>', unsafe_allow_html=True)
            st.markdown('<p class="event-detail" style="margin-top:1rem; font-size:0.6rem;">Node size = touches<br>Line width = pass frequency</p>', unsafe_allow_html=True)

        elif mode == "Heatmap":
            team = home if selected_team == 'home' else away
            team_actions = fdf[fdf['team_name'] == team]
            st.markdown(f'<p class="mode-stats"><strong>{len(team_actions)}</strong> actions</p>', unsafe_allow_html=True)
            st.markdown('<p class="event-detail">Activity concentration shown by color intensity</p>', unsafe_allow_html=True)

        elif mode == "Progressive":
            prog_passes = fdf[(fdf['type_name'].isin(ARROW_EVENTS)) & (fdf.apply(is_progressive_advanced, axis=1))]
            prog_carries = fdf[fdf.apply(lambda r: is_carry(r) and is_progressive_advanced(r), axis=1)]
            st.markdown(f'<p class="mode-stats"><strong>{len(prog_passes)}</strong> prog. passes<br><strong>{len(prog_carries)}</strong> prog. carries</p>', unsafe_allow_html=True)
            for t in [home, away]:
                pp = prog_passes[prog_passes['team_name'] == t]
                pc = prog_carries[prog_carries['team_name'] == t]
                c = COLORS['home'] if t == home else COLORS['away']
                st.markdown(f'<p class="team-stat-line" style="color:{c}">{t}: {len(pp)}P / {len(pc)}C</p>', unsafe_allow_html=True)

        elif mode == "Pressing":
            team = home if selected_team == 'home' else away
            team_def = fdf[(fdf['team_name'] == team) & (fdf['type_name'].isin(DEFENSIVE_EVENTS))]
            if selected_team == 'home':
                high = len(team_def[team_def['start_x'] > 66.67])
            else:
                high = len(team_def[team_def['start_x'] < 33.33])
            st.markdown(f'<p class="mode-stats"><strong>{len(team_def)}</strong> defensive actions</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="event-detail"><strong style="color:{COLORS["gold"]}">{high}</strong> in high press zone</p>', unsafe_allow_html=True)

        elif mode == "Final Third":
            for t in [home, away]:
                ft, _ = get_passes_into_areas(fdf, t, home)
                c = COLORS['home'] if t == home else COLORS['away']
                st.markdown(f'<p class="team-stat-line" style="color:{c}">{t}: {len(ft)} passes</p>', unsafe_allow_html=True)

        elif mode == "Box Entry":
            for t in [home, away]:
                _, pa = get_passes_into_areas(fdf, t, home)
                c = COLORS['home'] if t == home else COLORS['away']
                st.markdown(f'<p class="team-stat-line" style="color:{c}">{t}: {len(pa)} entries</p>', unsafe_allow_html=True)

        elif mode == "Momentum":
            st.markdown('<p class="mode-stats">Match control by 5-min intervals</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="team-stat-line" style="color:{COLORS["home"]}">{home[:3].upper()} = bottom</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="team-stat-line" style="color:{COLORS["away"]}">{away[:3].upper()} = top</p>', unsafe_allow_html=True)

        else:
            st.markdown('<p class="empty-state">Apply filters to see events</p>', unsafe_allow_html=True)

    # Table
    with st.expander("Full Event Table"):
        if len(fdf) > 0:
            tdf = fdf[['time_display', 'team_name', 'player_name', 'type_name', 'result_name']].copy()
            tdf.columns = ['Time', 'Team', 'Player', 'Event', 'Result']
            tdf['Event'] = tdf['Event'].str.replace('_', ' ').str.title()
            tdf['Result'] = tdf['Result'].str.title()
            st.dataframe(tdf, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
