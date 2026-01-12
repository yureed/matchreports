"""
Utils package for Matchday Analytics
"""

from .constants import (
    COLORS,
    PLOTLY_COLORS,
    SHOT_EVENTS,
    ARROW_EVENTS,
    CARRY_EVENTS,
    DEFENSIVE_EVENTS,
    KEEPER_EVENTS,
    EVENT_MARKERS,
    PREMIUM_CSS,
)

from .data import (
    load_data,
    load_matches_list,
    load_match_events,
    load_player_season_stats,
    load_team_season_stats,
    load_team_ppda,
    get_available_teams,
    flip_coords,
    is_progressive_advanced,
    estimate_player_minutes,
)

from .stats import (
    compute_match_stats,
    calculate_ppda,
    calculate_team_stats,
    calculate_player_stats,
)

from .charts import (
    create_scatter_plot,
    create_comparison_scatter,
    create_quadrant_chart,
    create_bar_chart,
    create_grouped_bar,
    create_on_ball_scatter,
    get_plotly_layout,
)

__all__ = [
    # Constants
    'COLORS', 'PLOTLY_COLORS', 'SHOT_EVENTS', 'ARROW_EVENTS',
    'CARRY_EVENTS', 'DEFENSIVE_EVENTS', 'KEEPER_EVENTS',
    'EVENT_MARKERS', 'PREMIUM_CSS',
    # Data (optimized)
    'load_data', 'load_matches_list', 'load_match_events',
    'load_player_season_stats', 'load_team_season_stats',
    'load_team_ppda', 'get_available_teams',
    'flip_coords', 'is_progressive_advanced', 'estimate_player_minutes',
    # Stats
    'compute_match_stats', 'calculate_ppda',
    'calculate_team_stats', 'calculate_player_stats',
    # Charts
    'create_scatter_plot', 'create_comparison_scatter',
    'create_quadrant_chart', 'create_bar_chart',
    'create_grouped_bar', 'create_on_ball_scatter',
    'get_plotly_layout',
]
