"""
Interactive Charts with Plotly
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from .constants import PLOTLY_COLORS


def get_plotly_layout(title="", height=500):
    """Get consistent Plotly layout for dark theme"""
    return dict(
        title=dict(
            text=title,
            font=dict(size=14, color=PLOTLY_COLORS['text_primary']),
            x=0,
            xanchor='left'
        ),
        paper_bgcolor=PLOTLY_COLORS['paper'],
        plot_bgcolor=PLOTLY_COLORS['bg'],
        font=dict(
            family="Inter, -apple-system, sans-serif",
            color=PLOTLY_COLORS['text']
        ),
        height=height,
        margin=dict(l=60, r=40, t=60, b=60),
        xaxis=dict(
            gridcolor=PLOTLY_COLORS['grid'],
            zerolinecolor=PLOTLY_COLORS['grid'],
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            gridcolor=PLOTLY_COLORS['grid'],
            zerolinecolor=PLOTLY_COLORS['grid'],
            tickfont=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor=PLOTLY_COLORS['paper'],
            font_size=12,
            font_family="Inter, monospace"
        ),
        legend=dict(
            bgcolor='rgba(20,20,22,0.9)',
            bordercolor=PLOTLY_COLORS['grid'],
            borderwidth=1,
            font=dict(size=11)
        )
    )


def create_scatter_plot(df, x_col, y_col, title, x_label, y_label,
                         size_col=None, color_col=None, hover_name='player'):
    """
    Create interactive scatter plot with hover info.

    Args:
        df: DataFrame with player data
        x_col: Column for x-axis
        y_col: Column for y-axis
        title: Chart title
        x_label: X-axis label
        y_label: Y-axis label
        size_col: Optional column for point sizes
        color_col: Optional column for color grouping
        hover_name: Column to show as main hover text
    """
    if len(df) == 0:
        return None

    # Prepare hover data
    hover_data = {
        'team': True,
        'minutes': True,
        x_col: ':.2f',
        y_col: ':.2f',
    }

    # Calculate sizes
    if size_col and size_col in df.columns:
        sizes = df[size_col] / df[size_col].max() * 30 + 10
    else:
        sizes = 15

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        hover_name=hover_name,
        hover_data=hover_data,
        size=sizes if isinstance(sizes, pd.Series) else None,
        color=color_col if color_col else None,
        color_discrete_sequence=[PLOTLY_COLORS['accent'], PLOTLY_COLORS['success'],
                                  PLOTLY_COLORS['warning'], PLOTLY_COLORS['purple']],
    )

    if not isinstance(sizes, pd.Series):
        fig.update_traces(marker=dict(size=sizes))

    fig.update_traces(
        marker=dict(
            line=dict(width=1, color='white'),
            opacity=0.8
        )
    )

    # Add median lines
    x_median = df[x_col].median()
    y_median = df[y_col].median()

    fig.add_hline(y=y_median, line_dash="dash",
                  line_color=PLOTLY_COLORS['text'], opacity=0.3,
                  annotation_text=f"Median: {y_median:.2f}",
                  annotation_position="right")

    fig.add_vline(x=x_median, line_dash="dash",
                  line_color=PLOTLY_COLORS['text'], opacity=0.3,
                  annotation_text=f"Median: {x_median:.2f}",
                  annotation_position="top")

    layout = get_plotly_layout(title, height=500)
    layout['xaxis']['title'] = x_label
    layout['yaxis']['title'] = y_label

    fig.update_layout(**layout)

    return fig


def create_comparison_scatter(df, x_col, y_col, title, x_label, y_label,
                               category_col='team', hover_name='player'):
    """
    Create scatter plot comparing two metrics with team colors.
    """
    if len(df) == 0:
        return None

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color=category_col,
        hover_name=hover_name,
        hover_data={
            'team': True,
            'minutes': True,
            x_col: ':.2f',
            y_col: ':.2f',
        },
        size='minutes',
        size_max=25,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_traces(
        marker=dict(
            line=dict(width=0.5, color='white'),
            opacity=0.75
        )
    )

    layout = get_plotly_layout(title, height=550)
    layout['xaxis']['title'] = x_label
    layout['yaxis']['title'] = y_label
    layout['showlegend'] = True
    layout['legend'] = dict(
        bgcolor='rgba(20,20,22,0.9)',
        bordercolor=PLOTLY_COLORS['grid'],
        borderwidth=1,
        font=dict(size=10),
        orientation='h',
        yanchor='bottom',
        y=1.02,
        xanchor='right',
        x=1
    )

    fig.update_layout(**layout)

    # Add median lines
    fig.add_hline(y=df[y_col].median(), line_dash="dot",
                  line_color=PLOTLY_COLORS['text'], opacity=0.2)
    fig.add_vline(x=df[x_col].median(), line_dash="dot",
                  line_color=PLOTLY_COLORS['text'], opacity=0.2)

    return fig


def create_quadrant_chart(df, x_col, y_col, title, x_label, y_label,
                           hover_name='player', annotate_top=5):
    """
    Create a quadrant analysis scatter plot with annotations for top performers.
    """
    if len(df) == 0:
        return None

    fig = go.Figure()

    # Add scatter points
    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='markers',
        marker=dict(
            size=df['minutes'] / df['minutes'].max() * 25 + 8,
            color=PLOTLY_COLORS['accent'],
            opacity=0.7,
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

    # Add annotations for top performers
    if annotate_top > 0:
        # Top by x
        for _, row in df.nlargest(annotate_top, x_col).iterrows():
            fig.add_annotation(
                x=row[x_col],
                y=row[y_col],
                text=row[hover_name].split()[-1][:12],
                showarrow=True,
                arrowhead=0,
                arrowsize=0.5,
                arrowwidth=1,
                arrowcolor=PLOTLY_COLORS['text'],
                ax=20,
                ay=-20,
                font=dict(size=9, color=PLOTLY_COLORS['text_primary']),
                bgcolor='rgba(20,20,22,0.8)',
                borderpad=3,
            )

    # Median lines
    x_med = df[x_col].median()
    y_med = df[y_col].median()

    fig.add_hline(y=y_med, line_dash="dash",
                  line_color=PLOTLY_COLORS['text'], opacity=0.25)
    fig.add_vline(x=x_med, line_dash="dash",
                  line_color=PLOTLY_COLORS['text'], opacity=0.25)

    # Quadrant labels
    x_max, y_max = df[x_col].max(), df[y_col].max()
    x_min, y_min = df[x_col].min(), df[y_col].min()

    quadrant_labels = [
        (x_max * 0.85, y_max * 0.95, "Elite", PLOTLY_COLORS['success']),
        (x_min + (x_med - x_min) * 0.1, y_max * 0.95, "Carriers", PLOTLY_COLORS['warning']),
        (x_max * 0.85, y_min + (y_med - y_min) * 0.1, "Passers", PLOTLY_COLORS['accent']),
    ]

    for x, y, text, color in quadrant_labels:
        fig.add_annotation(
            x=x, y=y, text=text,
            showarrow=False,
            font=dict(size=10, color=color),
            opacity=0.6
        )

    layout = get_plotly_layout(title, height=550)
    layout['xaxis']['title'] = x_label
    layout['yaxis']['title'] = y_label

    fig.update_layout(**layout)

    return fig


def create_bar_chart(df, x_col, y_col, title, color=None, horizontal=True):
    """Create a bar chart"""
    if len(df) == 0:
        return None

    color = color or PLOTLY_COLORS['accent']

    if horizontal:
        fig = go.Figure(go.Bar(
            x=df[y_col],
            y=df[x_col],
            orientation='h',
            marker_color=color,
            opacity=0.9,
            text=df[y_col].round(1),
            textposition='outside',
            textfont=dict(size=10, color=PLOTLY_COLORS['text']),
        ))
    else:
        fig = go.Figure(go.Bar(
            x=df[x_col],
            y=df[y_col],
            marker_color=color,
            opacity=0.9,
            text=df[y_col].round(1),
            textposition='outside',
            textfont=dict(size=10, color=PLOTLY_COLORS['text']),
        ))

    layout = get_plotly_layout(title, height=400)
    fig.update_layout(**layout)

    return fig


def create_grouped_bar(df, x_col, metrics, title, colors=None):
    """Create grouped bar chart for comparing multiple metrics"""
    if len(df) == 0:
        return None

    if colors is None:
        colors = [PLOTLY_COLORS['accent'], PLOTLY_COLORS['success'],
                  PLOTLY_COLORS['warning'], PLOTLY_COLORS['purple']]

    fig = go.Figure()

    for i, (metric, label) in enumerate(metrics):
        fig.add_trace(go.Bar(
            x=df[x_col],
            y=df[metric],
            name=label,
            marker_color=colors[i % len(colors)],
            opacity=0.9,
        ))

    layout = get_plotly_layout(title, height=450)
    layout['barmode'] = 'group'
    layout['xaxis']['tickangle'] = -45

    fig.update_layout(**layout)

    return fig


def create_on_ball_scatter(df, passes_col, carries_col, title, suffix=""):
    """
    Create specialized scatter for on-ball analysis (passes vs carries).
    """
    if len(df) == 0:
        return None

    fig = go.Figure()

    # Main scatter
    fig.add_trace(go.Scatter(
        x=df[passes_col],
        y=df[carries_col],
        mode='markers',
        marker=dict(
            size=df['minutes'] / df['minutes'].max() * 30 + 8,
            color=PLOTLY_COLORS['accent'],
            opacity=0.75,
            line=dict(width=1, color='white')
        ),
        text=df['player'],
        customdata=np.stack((
            df['team'],
            df['minutes'],
            df[passes_col],
            df[carries_col]
        ), axis=-1),
        hovertemplate=(
            "<b>%{text}</b><br>" +
            "Team: %{customdata[0]}<br>" +
            "Minutes: %{customdata[1]:.0f}<br>" +
            f"Passes{suffix}: %{{customdata[2]:.2f}}<br>" +
            f"Carries{suffix}: %{{customdata[3]:.2f}}<br>" +
            "<extra></extra>"
        ),
        name='Players'
    ))

    # Median lines
    x_med = df[passes_col].median()
    y_med = df[carries_col].median()

    fig.add_hline(y=y_med, line_dash="dash",
                  line_color=PLOTLY_COLORS['text'], opacity=0.3)
    fig.add_vline(x=x_med, line_dash="dash",
                  line_color=PLOTLY_COLORS['text'], opacity=0.3)

    # Annotate outliers
    total = df[passes_col] + df[carries_col]
    for _, row in df.nlargest(5, passes_col).iterrows():
        fig.add_annotation(
            x=row[passes_col], y=row[carries_col],
            text=row['player'].split()[-1][:10],
            showarrow=True, arrowhead=0, arrowsize=0.5,
            arrowcolor=PLOTLY_COLORS['text'],
            ax=15, ay=-15,
            font=dict(size=9, color=PLOTLY_COLORS['text_primary']),
            bgcolor='rgba(20,20,22,0.8)',
        )

    layout = get_plotly_layout(title, height=550)
    layout['xaxis']['title'] = f"Passes{suffix}"
    layout['yaxis']['title'] = f"Carries{suffix}"

    fig.update_layout(**layout)

    return fig
