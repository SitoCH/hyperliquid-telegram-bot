import io
from typing import List
import pandas as pd  # type: ignore[import]
import matplotlib.pyplot as plt
import mplfinance as mpf  # type: ignore[import]
import numpy as np  # type: ignore[import]

from utils import fmt_price
from .wyckoff_types import WyckoffState, Timeframe
from .significant_levels import find_significant_levels
from matplotlib.ticker import FuncFormatter


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df['o'] + df['h'] + df['l'] + df['c']) / 4

    ha_df = pd.DataFrame(dict(c=ha_close, v=df['v']))

    ha_df['o'] = [0.0] * len(df)

    prekey = df.index[0]
    ha_df.at[prekey, 'o'] = df.at[prekey, 'o']

    for key in df.index[1:]:
        ha_df.at[key, 'o'] = (ha_df.at[prekey, 'o'] + ha_df.at[prekey, 'c']) / 2.0
        prekey = key

    ha_df['h'] = pd.concat([ha_df.o, df.h], axis=1).max(axis=1)
    ha_df['l'] = pd.concat([ha_df.o, df.l], axis=1).min(axis=1)

    return ha_df

def save_to_buffer(df: pd.DataFrame, wyckoff: WyckoffState, title: str, timeframe: Timeframe, mid: float) -> io.BytesIO:
    """Use timeframe settings for chart window."""
    # Calculate thresholds from full dataset
    df['MACD_Hist'] = df['MACD_Hist'].fillna(0)
    
    # Using percentiles instead of fixed percentage of max/min
    # This makes the thresholds more robust against outliers
    positive_values = df['MACD_Hist'][df['MACD_Hist'] > 0]
    negative_values = df['MACD_Hist'][df['MACD_Hist'] < 0]
    
    # If we have enough data points, use percentiles
    if len(positive_values) > 10:
        strong_positive_threshold = max(positive_values.quantile(0.75), 0.000001)
    else:
        strong_positive_threshold = max(df['MACD_Hist'].max() * 0.4, 0.000001)
        
    if len(negative_values) > 10:
        strong_negative_threshold = min(negative_values.quantile(0.25), -0.000001)
    else:
        strong_negative_threshold = min(df['MACD_Hist'].min() * 0.4, -0.000001)
    
    # Calculate significant levels using full dataset
    resistance_levels, support_levels = find_significant_levels(df, wyckoff, mid, timeframe)

    # Now filter for plotting window using timeframe settings
    from_time = df['t'].max() - timeframe.settings.chart_image_time_delta
    df_plot = df.loc[df['t'] >= from_time].copy()

    buf = io.BytesIO()
    fig, ax = plt.subplots(3, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [7, 1, 1]})
    
    df_plot['SuperTrend_Green'] = np.where(
        df_plot['c'] > df_plot['SuperTrend'],
        df_plot['SuperTrend'],
        np.nan
    )
    df_plot['SuperTrend_Red'] = np.where(
        df_plot['c'] <= df_plot['SuperTrend'],
        df_plot['SuperTrend'],
        np.nan
    )

    ha_df = heikin_ashi(df_plot)

    def determine_color(value: float) -> str:
        if value > 0:
            if value >= strong_positive_threshold * 1.5:
                return '#006400'  # dark green
            elif value >= strong_positive_threshold:
                return '#008000'  # green
            elif value >= strong_positive_threshold * 0.5:
                return '#2E8B57'  # sea green
            else:
                return '#90EE90'  # light green
        elif value < 0:
            if value <= strong_negative_threshold * 1.5:
                return '#8B0000'  # dark red
            elif value <= strong_negative_threshold:
                return '#FF0000'  # red
            elif value <= strong_negative_threshold * 0.5:
                return '#CD5C5C'  # indian red
            else:
                return '#FFA07A'  # light salmon
        else:
            return '#D3D3D3'  # light gray for zero values

    # Apply colors based on full dataset thresholds but only to plotting window data
    macd_hist_colors = df_plot['MACD_Hist'].apply(determine_color).values

    def price_to_percent(price):
        return ((price / mid) - 1) * 100
    

    level_lines = []
    for level in reversed(resistance_levels):
        line = pd.Series([level] * len(df_plot), index=df_plot.index)
        pct_diff = price_to_percent(level)
        pct_str = f'+{pct_diff:.2f}%' if pct_diff > 0 else f'{pct_diff:.2f}%'
        level_lines.append(mpf.make_addplot(line, ax=ax[0], color='purple', width=0.5, 
                                            label=f'R {fmt_price(level)} ({pct_str})', linestyle='--'))
    
    is_ha_bullish = ha_df['c'].iloc[-1] >= ha_df['o'].iloc[-1]
    current_price_line = pd.Series([mid] * len(df_plot), index=df_plot.index)
    level_lines.append(mpf.make_addplot(current_price_line, ax=ax[0], 
                                        color='green' if is_ha_bullish else 'red', 
                                        width=0.5, label=f'Current {fmt_price(mid)}', 
                                        linestyle=':', alpha=0.6))

    for level in support_levels:
        line = pd.Series([level] * len(df_plot), index=df_plot.index)
        pct_diff = price_to_percent(level)
        pct_str = f'+{pct_diff:.2f}%' if pct_diff > 0 else f'{pct_diff:.2f}%'
        level_lines.append(mpf.make_addplot(line, ax=ax[0], color='purple', width=0.5, 
                                            label=f'S {fmt_price(level)} ({pct_str})', linestyle=':'))

    mpf.plot(ha_df,
            type='candle',
            columns=['o', 'h', 'l', 'c', 'v'],
            ax=ax[0],
            volume=ax[2],
            volume_panel=2,
            scale_width_adjustment={'volume': 0.7},
            axtitle=title,
            style='charles',
            ylabel='', 
            ylabel_lower='',
            addplot=[
                mpf.make_addplot(df_plot['SuperTrend'], ax=ax[0], color='green', width=0.5),
                mpf.make_addplot(df_plot['EMA'], ax=ax[0], color='orange', width=0.5),
                *level_lines,
                mpf.make_addplot(df_plot['MACD_Hist'], type='bar', panel=1, ax=ax[1], 
                                color=macd_hist_colors, width=0.7)
            ])

    x_min, x_max = ax[1].get_xlim()
    ax[0].set_xlim(x_min, x_max)
    ax[2].set_xlim(x_min, x_max)

    ax[0].set_ylabel('Price')
    
    ax[1].set_ylabel('MACD')
    ax[2].set_ylabel('Vol.')
    
    ax[1].yaxis.set_label_position('right')
    ax[1].tick_params(axis='y', length=0, labelright=False, labelleft=False) 
    
    ax[2].yaxis.set_label_position('right')
    ax[2].tick_params(axis='y', length=0, labelright=False, labelleft=False) 
    
    ax[0].set_xticklabels([])
    ax[0].set_xlabel('')
    ax[0].tick_params(axis='x', length=0)
    
    ax[1].set_xticklabels([])
    ax[1].set_xlabel('')
    ax[1].tick_params(axis='x', length=0) 
    
    plt.subplots_adjust(hspace=-0.05)
    
    ax[0].spines['bottom'].set_visible(False)
    ax[1].spines['top'].set_visible(True)
    ax[1].spines['bottom'].set_visible(False)
    ax[2].spines['top'].set_visible(True)

    # Add legend to candle panel
    ax[0].legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.subplots_adjust(hspace=-0.01) 
    
    plt.savefig(buf, format='png', dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_chart(dataframes: dict[Timeframe, pd.DataFrame], states: dict[Timeframe, WyckoffState], coin: str, mid: float) -> List[io.BytesIO]:
    chart_buffers = []

    plt.switch_backend('Agg')

    try:
        df_15m_plot = dataframes[Timeframe.MINUTES_15]
        chart_buffers.append(save_to_buffer(df_15m_plot, states[Timeframe.MINUTES_15], f"{coin} - 15M Chart", Timeframe.MINUTES_15, mid))

        df_1h_plot = dataframes[Timeframe.HOUR_1]
        chart_buffers.append(save_to_buffer(df_1h_plot, states[Timeframe.HOUR_1], f"{coin} - 1H Chart", Timeframe.HOUR_1, mid))

        df_4h_plot = dataframes[Timeframe.HOURS_4]
        chart_buffers.append(save_to_buffer(df_4h_plot, states[Timeframe.HOURS_4], f"{coin} - 4H Chart", Timeframe.HOURS_4, mid))

    except Exception as e:
        # Clean up on error
        plt.close('all')
        for buf in chart_buffers:
            if buf and not buf.closed:
                buf.close()
        raise e

    plt.close('all')
    return chart_buffers
