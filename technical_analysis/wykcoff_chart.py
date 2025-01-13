import io
from typing import List
import pandas as pd  # type: ignore[import]
import matplotlib.pyplot as plt
import mplfinance as mpf  # type: ignore[import]
import numpy as np  # type: ignore[import]

from utils import fmt_price
from technical_analysis.significant_levels import find_significant_levels


def heikin_ashi(df: pd.DataFrame) -> pd.DataFrame:
    ha_close = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4

    ha_df = pd.DataFrame(dict(Close=ha_close, Volume=df['Volume']))

    ha_df['Open'] = [0.0] * len(df)

    prekey = df.index[0]
    ha_df.at[prekey, 'Open'] = df.at[prekey, 'Open']

    for key in df.index[1:]:
        ha_df.at[key, 'Open'] = (ha_df.at[prekey, 'Open'] + ha_df.at[prekey, 'Close']) / 2.0
        prekey = key

    ha_df['High'] = pd.concat([ha_df.Open, df.High], axis=1).max(axis=1)
    ha_df['Low'] = pd.concat([ha_df.Open, df.Low], axis=1).min(axis=1)

    return ha_df

def save_to_buffer(df: pd.DataFrame, title: str, chart_image_time_delta) -> io.BytesIO:
    from_time = df['t'].max() - chart_image_time_delta
    df_plot = df.loc[df['t'] >= from_time].copy()

    buf = io.BytesIO()
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={'height_ratios': [3, 1]})

    df_plot['SuperTrend_Green'] = np.where(
        df_plot['Close'] > df_plot['SuperTrend'],
        df_plot['SuperTrend'],
        np.nan
    )
    df_plot['SuperTrend_Red'] = np.where(
        df_plot['Close'] <= df_plot['SuperTrend'],
        df_plot['SuperTrend'],
        np.nan
    )

    ha_df = heikin_ashi(df_plot)

    df_plot['MACD_Hist'] = df_plot['MACD_Hist'].fillna(0)
    strong_positive_threshold = max(df_plot['MACD_Hist'].max() * 0.4, 0.000001)
    strong_negative_threshold = min(df_plot['MACD_Hist'].min() * 0.4, -0.000001)

    def determine_color(value: float) -> str:
        if value >= strong_positive_threshold:
            return 'green'
        elif 0 < value < strong_positive_threshold:
            return 'lightgreen'
        elif strong_negative_threshold < value <= 0:
            return 'lightcoral'
        else:
            return 'red'

    macd_hist_colors = df_plot['MACD_Hist'].apply(determine_color).values

    resistance_levels, support_levels = find_significant_levels(df)
    
    level_lines = []
    for level in resistance_levels:
        line = pd.Series([level] * len(df_plot), index=df_plot.index)
        level_lines.append(mpf.make_addplot(line, ax=ax[0], color='purple', width=0.5, 
                                            label=f'R {fmt_price(level)}', linestyle='--'))
    
    for level in support_levels:
        line = pd.Series([level] * len(df_plot), index=df_plot.index)
        level_lines.append(mpf.make_addplot(line, ax=ax[0], color='purple', width=0.5, 
                                            label=f'S {fmt_price(level)}', linestyle=':'))

    current_price = df_plot['Close'].iloc[-1]
    is_ha_bullish = ha_df['Close'].iloc[-1] >= ha_df['Open'].iloc[-1]
    current_price_line = pd.Series([current_price] * len(df_plot), index=df_plot.index)
    level_lines.append(mpf.make_addplot(current_price_line, ax=ax[0], 
                                        color='green' if is_ha_bullish else 'red', 
                                        width=0.5, label=f'Current {fmt_price(current_price)}', 
                                        linestyle=':', alpha=0.6))

    mpf.plot(ha_df,
            type='candle',
            ax=ax[0],
            volume=False,
            axtitle=title,
            style='charles',
            addplot=[
                mpf.make_addplot(df_plot['SuperTrend'], ax=ax[0], color='green', width=0.5),
                mpf.make_addplot(df_plot['VWAP'], ax=ax[0], color='blue', width=0.5),
                mpf.make_addplot(df_plot['EMA'], ax=ax[0], color='orange', width=0.5),
                *level_lines,
                mpf.make_addplot(df_plot['MACD_Hist'], type='bar', width=0.7, 
                                color=macd_hist_colors, ax=ax[1], alpha=0.4)
            ])

    ax[0].legend(loc='upper left', fontsize='small')

    plt.tight_layout()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    return buf


def generate_chart(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, coin: str) -> List[io.BytesIO]:
    chart_buffers = []

    plt.switch_backend('Agg')

  

    try:
        df_15m_plot = df_15m.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        chart_buffers.append(save_to_buffer(df_15m_plot, f"{coin} - 15M Chart", pd.Timedelta(hours=48)))

        df_1h_plot = df_1h.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        chart_buffers.append(save_to_buffer(df_1h_plot, f"{coin} - 1H Chart", pd.Timedelta(days=7)))

        df_4h_plot = df_4h.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        chart_buffers.append(save_to_buffer(df_4h_plot, f"{coin} - 4H Chart", pd.Timedelta(days=21)))

        df_1d_plot = df_1d.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        chart_buffers.append(save_to_buffer(df_1d_plot, f"{coin} - 1D Chart", pd.Timedelta(days=180)))

    except Exception as e:
        # Clean up on error
        plt.close('all')
        for buf in chart_buffers:
            if buf and not buf.closed:
                buf.close()
        raise e

    plt.close('all')
    return chart_buffers
