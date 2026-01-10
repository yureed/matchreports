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
    flip_coords,
    is_progressive_advanced,
    is_carry,
    estimate_player_minutes,
    get_final_third_entries,
    get_box_entries,
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
    # Data
    'load_data', 'flip_coords', 'is_progressive_advanced',
    'is_carry', 'estimate_player_minutes',
    'get_final_third_entries', 'get_box_entries',
    # Stats
    'compute_match_stats', 'calculate_ppda',
    'calculate_team_stats', 'calculate_player_stats',
    # Charts
    'create_scatter_plot', 'create_comparison_scatter',
    'create_quadrant_chart', 'create_bar_chart',
    'create_grouped_bar', 'create_on_ball_scatter',
    'get_plotly_layout',
]
