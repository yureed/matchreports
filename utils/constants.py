"""
Constants - Colors, event types, and configuration
"""

# =============================================================================
# COLOR SCHEMES
# =============================================================================

COLORS = {
    'home': '#2997ff',      # Blue
    'away': '#ff453a',      # Red
    'success': '#30d158',   # Green
    'fail': '#6e6e73',      # Gray
    'gold': '#ff9f0a',      # Gold/Orange
    'purple': '#bf5af2',    # Purple
    'pitch': '#0d1117',
    'pitch_lines': '#1a3d2e',
    'bg': '#0a0a0c',
    'card': '#141416',
    'elevated': '#1a1a1e',
    'border': '#1f1f23',
    'text_primary': '#f5f5f7',
    'text_secondary': '#a1a1a6',
    'text_tertiary': '#6e6e73',
}

# Plotly-compatible color scheme
PLOTLY_COLORS = {
    'bg': '#0a0a0c',
    'paper': '#141416',
    'grid': '#1f1f23',
    'text': '#a1a1a6',
    'text_primary': '#f5f5f7',
    'accent': '#2997ff',
    'success': '#30d158',
    'warning': '#ff9f0a',
    'danger': '#ff453a',
    'purple': '#bf5af2',
}

# =============================================================================
# EVENT TYPE CATEGORIES
# =============================================================================

SHOT_EVENTS = ['shot', 'shot_freekick', 'shot_penalty']

ARROW_EVENTS = ['pass', 'cross', 'corner_crossed', 'freekick_crossed',
                'corner_short', 'freekick_short']

CARRY_EVENTS = ['dribble', 'take_on']

DEFENSIVE_EVENTS = ['tackle', 'interception', 'clearance', 'foul']

KEEPER_EVENTS = ['keeper_save', 'keeper_claim', 'keeper_punch',
                 'keeper_pick_up', 'keeper_drop']

# Event marker styles for pitch visualization
EVENT_MARKERS = {
    'pass': ('o', 8),
    'cross': ('o', 8),
    'shot': ('^', 10),
    'shot_freekick': ('^', 10),
    'shot_penalty': ('^', 10),
    'tackle': ('s', 8),
    'interception': ('s', 8),
    'dribble': ('D', 8),
    'take_on': ('D', 8),
    'foul': ('X', 8),
    'clearance': ('p', 8),
    'default': ('o', 7),
}

# =============================================================================
# CSS STYLES
# =============================================================================

PREMIUM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    :root {
        --bg-void: #0a0a0c;
        --bg-main: #0f0f11;
        --bg-card: #141416;
        --bg-elevated: #1a1a1e;
        --bg-hover: #222228;
        --border: rgba(255, 255, 255, 0.08);
        --border-subtle: rgba(255, 255, 255, 0.04);
        --text-primary: #f5f5f7;
        --text-secondary: #a1a1a6;
        --text-tertiary: #6e6e73;
        --accent: #2997ff;
        --accent-dim: rgba(41, 151, 255, 0.15);
        --success: #30d158;
        --warning: #ff9f0a;
        --danger: #ff453a;
    }

    footer {display: none !important;}
    header {visibility: hidden;}

    .stApp {
        background: var(--bg-void);
    }

    [data-testid="stSidebar"] {
        background: var(--bg-main);
        border-right: 1px solid var(--border);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 3px;
    }

    .stTabs [data-baseweb="tab"] {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        background: transparent !important;
        color: var(--text-tertiary) !important;
        border: none !important;
        border-radius: 4px !important;
        padding: 0.6rem 1.25rem !important;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
    }

    .stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) {
        background: var(--bg-hover) !important;
        color: var(--text-primary) !important;
    }

    /* Dataframe styling */
    .stDataFrame {
        font-family: 'JetBrains Mono', monospace !important;
    }

    [data-testid="stDataFrame"] > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    .block-container {
        padding: 1rem 2rem 2rem 2rem !important;
        max-width: 100% !important;
    }
</style>
"""
