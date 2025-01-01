import os
import io
import time
from tzlocal import get_localzone
from typing import Set, List, Dict, Any, Optional, cast, Tuple
import pandas as pd  # type: ignore[import]
import pandas_ta as ta  # type: ignore[import]
import matplotlib.pyplot as plt
import mplfinance as mpf  # type: ignore[import]
import numpy as np  # type: ignore[import]

from tabulate import tabulate, simple_separated_format
from telegram import Update, Message, CallbackQuery
from telegram.ext import CallbackContext, ContextTypes, ConversationHandler
from telegram.constants import ParseMode
from telegram.error import TelegramError

from logging_utils import logger
from telegram_utils import telegram_utils
from hyperliquid_utils import hyperliquid_utils
from utils import OPERATION_CANCELLED, fmt, fmt_price

SELECTING_COIN_FOR_TA = range(1)


async def execute_ta(update: Update, context: CallbackContext) -> int:
    if not update.message:
        return ConversationHandler.END
    
    await update.message.reply_text("Choose a coin to analyze:", reply_markup=hyperliquid_utils.get_coins_reply_markup())
    return cast(int, SELECTING_COIN_FOR_TA)


async def selected_coin_for_ta(update: Update, context: CallbackContext) -> int:
    if not update.callback_query:
        return ConversationHandler.END

    query = update.callback_query
    await query.answer()

    if not query.data:
        return ConversationHandler.END

    coin = query.data
    if coin == "cancel":
        await query.edit_message_text(text=OPERATION_CANCELLED)
        return ConversationHandler.END

    await query.edit_message_text(text=f"Analyzing {coin}...")
    await analyze_candles_for_coin(context, coin, hyperliquid_utils.info.all_mids(), always_notify=True)
    await query.delete_message()
    return ConversationHandler.END


async def analyze_candles(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Analyze candles for configured coins and categories."""

    all_mids = hyperliquid_utils.info.all_mids()

    coins_to_analyze = await get_coins_to_analyze(all_mids)
    
    if not coins_to_analyze:
        return
        
    for coin in coins_to_analyze:
        await analyze_candles_for_coin(context, coin, all_mids, always_notify=False)
        time.sleep(5)

async def get_coins_to_analyze(all_mids: Dict[str, Any]) -> Set[str]:
    """Get the set of coins to analyze based on configuration."""
    coins_to_analyze: Set[str] = set()
    
    # Add explicitly configured coins
    configured_coins = os.getenv("HTB_COINS_TO_ANALYZE", "").split(",")
    coins_to_analyze.update(coin for coin in configured_coins if coin and coin in all_mids)
    
    # Add coins from configured categories
    if categories := os.getenv("HTB_CATEGORIES_TO_ANALYZE"):
        for category in categories.split(","):
            if not category.strip():
                continue
                
            params = {
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 25,
                "sparkline": "false",
                "category": category.strip(),
                "price_change_percentage": "24h,30d,1y",
            }
            
            cryptos = hyperliquid_utils.fetch_cryptos(params)
            coins_to_analyze.update(
                crypto["symbol"] for crypto in cryptos 
                if crypto["symbol"] in all_mids
            )
    
    # Add coins with open orders if configured
    if os.getenv('HTB_ANALYZE_COINS_WITH_OPEN_ORDERS', 'False') == 'True':
        coins_to_analyze.update(
            coin for coin in hyperliquid_utils.get_coins_with_open_positions()
            if coin in all_mids
        )
    
    return coins_to_analyze


async def analyze_candles_for_coin(context: ContextTypes.DEFAULT_TYPE, coin: str, all_mids: Dict[str, Any], always_notify: bool) -> None:
    logger.info(f"Running TA for {coin}")
    try:
        now = int(time.time() * 1000)
        candles_15m = hyperliquid_utils.info.candles_snapshot(coin, "15m", now - 125 * 86400000, now)
        candles_1h = hyperliquid_utils.info.candles_snapshot(coin, "1h", now - 250 * 86400000, now)
        candles_4h = hyperliquid_utils.info.candles_snapshot(coin, "4h", now - 500 * 86400000, now)
        candles_1d = hyperliquid_utils.info.candles_snapshot(coin, "1d", now - 750 * 86400000, now)

        local_tz = get_localzone()
        df_15m = prepare_dataframe(candles_15m, local_tz)
        df_1h = prepare_dataframe(candles_1h, local_tz)
        df_4h = prepare_dataframe(candles_4h, local_tz)
        df_1d = prepare_dataframe(candles_1d, local_tz)

        mid = float(all_mids[coin])
        # Apply indicators but only store flip results for 1h and 4h
        apply_indicators(df_15m, mid)  # 15m indicators but no flip detection
        st_flip_1h, wyckoff_flip_1h = apply_indicators(df_1h, mid)
        st_flip_4h, wyckoff_flip_4h = apply_indicators(df_4h, mid)
        apply_indicators(df_1d, mid)  # 1d indicators but no flip detection
        
        # Check if 4h candle is recent enough
        is_4h_recent = 'T' in df_4h.columns and df_4h["T"].iloc[-1] >= pd.Timestamp.now(local_tz) - pd.Timedelta(hours=1)
        
        # Determine if we should notify based on either SuperTrend or Wyckoff flips
        flip_on_1h = st_flip_1h or wyckoff_flip_1h
        flip_on_4h = (st_flip_4h or wyckoff_flip_4h) and is_4h_recent

        if always_notify or flip_on_1h or flip_on_4h:
            await send_trend_change_message(context, mid, df_15m, df_1h, df_4h, df_1d, coin)
    except Exception as e:
        logger.critical(e, exc_info=True)
        await telegram_utils.send(f"Failed to analyze candles for {coin}: {str(e)}")


def prepare_dataframe(candles: List[Dict[str, Any]], local_tz) -> pd.DataFrame:
    df = pd.DataFrame(candles)
    df["T"] = pd.to_datetime(df["T"], unit="ms", utc=True).dt.tz_convert(local_tz)
    df["t"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(local_tz)
    df[["c", "h", "l", "o", "v"]] = df[["c", "h", "l", "o", "v"]].astype(float)
    df["n"] = df["n"].astype(int)
    return df


def detect_wyckoff_phase(df: pd.DataFrame) -> None:
    """Analyze and store Wyckoff phase data directly in the dataframe"""
    # Get ATR with safety checks
    atr: float = df.get('ATR', pd.Series([0.0])).iloc[-1]
    if atr == 0 or pd.isna(atr):
        df['wyckoff_phase'] = "Unknown"
        df['uncertain_phase'] = True
        df['wyckoff_volume'] = "unknown"
        df['wyckoff_pattern'] = "unknown"
        df['wyckoff_volatility'] = "unknown"
        return
    
    # Use last 50 candles for calculation (more responsive to recent market conditions)
    analysis_window = min(50, len(df) - 1)
    recent_df = df.iloc[-analysis_window:]

    volume_sma = recent_df['v'].rolling(window=20).mean()
    price_sma = recent_df['c'].rolling(window=20).mean()
    price_std = recent_df['c'].rolling(window=20).std()

    momentum = recent_df['c'].pct_change(periods=5).rolling(window=10).mean()
    volume_trend = recent_df['v'].pct_change().rolling(window=10).mean()
    
    # Current market conditions with volatility
    curr_price = recent_df['c'].iloc[-1]
    curr_volume = recent_df['v'].iloc[-1]
    avg_price = price_sma.iloc[-1]
    price_std_last = price_std.iloc[-1]
    volatility = price_std / avg_price
    
    # Enhanced market condition checks
    is_high_volume = (curr_volume > volume_sma.iloc[-1] * 1.1) and (volume_trend.iloc[-1] > 0)
    price_strength = (curr_price - avg_price) / (price_std_last + 1e-8)
    momentum_strength = momentum.iloc[-1] * 100

    strong_dev_threshold = 1.2 
    neutral_zone_threshold = 0.4
    momentum_threshold = 0.8

    uncertain_phase = False
    
    if price_strength > strong_dev_threshold:
        if momentum_strength < -momentum_threshold and is_high_volume:
            phase = "dist."
        else:
            uncertain_phase = True
            phase = "~ dist."
    
    elif price_strength < -strong_dev_threshold:
        if momentum_strength > momentum_threshold and is_high_volume:
            phase = "acc."
        else:
            uncertain_phase = True
            phase = "~ acc."
    
    elif -neutral_zone_threshold <= price_strength <= neutral_zone_threshold:
        if abs(momentum_strength) < momentum_threshold and volatility.iloc[-1] < volatility.mean():
            phase = "rang."
        else:
            uncertain_phase = True
            phase = "~ rang."
    
    else:  # Transitional zones
        if price_strength > 0:
            if is_high_volume and momentum_strength > momentum_threshold:
                phase = "markup"
            else:
                uncertain_phase = True
                phase = "~ mrkp"
        else:
            if is_high_volume and momentum_strength < -momentum_threshold:
                phase = "markdown"
            else:
                uncertain_phase = True
                phase = "~ mrkdwn"
    
    # Store results
    df.loc[:, 'wyckoff_phase'] = phase
    df.loc[:, 'uncertain_phase'] = uncertain_phase
    df.loc[:, 'wyckoff_volume'] = "high" if is_high_volume else "low"
    df.loc[:, 'wyckoff_pattern'] = "trending" if abs(momentum_strength) > momentum_threshold else "ranging"
    df.loc[:, 'wyckoff_volatility'] = "high" if volatility.iloc[-1] > volatility.mean() else "normal"


def apply_indicators(df: pd.DataFrame, mid: float) -> Tuple[bool, bool]:
    """Apply indicators and return (supertrend_flip, wyckoff_flip)"""
    # SuperTrend: shorter for faster response
    st_length = 10
    # ATR: standard setting
    atr_length = 14
    # EMA: longer for trend following
    ema_length = 21
    # Volume SMA: about one day in periods
    vol_length = 24

    df.set_index("T", inplace=True)
    df.sort_index(inplace=True)

    # ATR calculation with error handling
    atr_calc = ta.atr(df["h"], df["l"], df["c"], length=atr_length)
    if atr_calc is not None:
        df["ATR"] = atr_calc
    else:
        df["ATR"] = pd.Series([0.0] * len(df), index=df.index)
        logger.warning("ATR calculation failed, using fallback values")

    # Calculate Volume confirmation first
    df["Volume_SMA"] = df["v"].rolling(window=vol_length).mean()
    df["Volume_Confirm"] = df["v"] > df["Volume_SMA"]

    # Then handle SuperTrend with optimized settings and error handling
    supertrend = ta.supertrend(df["h"], df["l"], df["c"], length=st_length, multiplier=3.5)
    if supertrend is not None and len(df) > st_length:
        df["SuperTrend"] = supertrend[f"SUPERT_{st_length}_3.5"]
        df["SuperTrend_Flip_Detected"] = (
            supertrend[f"SUPERTd_{st_length}_3.5"].diff().abs() == 1) & df["Volume_Confirm"]
    else:
        df["SuperTrend"] = df["c"]  # Use close price as fallback
        df["SuperTrend_Flip_Detected"] = False
        logger.warning(f"Insufficient data for SuperTrend calculation (needed >{st_length} points, got {len(df)})")

    # Volume confirmation using longer period
    df["Volume_SMA"] = df["v"].rolling(window=vol_length).mean()
    df["Volume_Confirm"] = df["v"] > df["Volume_SMA"]
    
    # Use shorter length for VWAP with safety checks
    df["VWAP"] = ta.vwap(df["h"], df["l"], df["c"], df["v"])
    
    # Add safety check for VWAP flip detection
    if len(df["VWAP"].dropna()) >= 2:  # Ensure we have at least 2 valid VWAP values
        df["VWAP_Flip_Detected"] = (
            ((mid > df["VWAP"].iloc[-2]) & (mid <= df["VWAP"].iloc[-1])) | 
            ((mid < df["VWAP"].iloc[-2]) & (mid >= df["VWAP"].iloc[-1]))
        )
    else:
        df["VWAP_Flip_Detected"] = False
        logger.warning("Insufficient data for VWAP flip detection")

    # MACD
    macd = ta.macd(df["c"], fast=12, slow=26, signal=9)
    if macd is not None and not macd["MACD_12_26_9"].isna().all():
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_Signal"] = macd["MACDs_12_26_9"]
        df["MACD_Hist"] = macd["MACDh_12_26_9"]
    else:
        # Use zeros instead of NaN for better chart rendering
        df["MACD"] = 0.0
        df["MACD_Signal"] = 0.0
        df["MACD_Hist"] = 0.0
        logger.warning("MACD calculation failed, using zeros")

    if "SuperTrend_Flip_Detected" in df.columns:
        # Only consider flips with significant price movement (>0.5% from SuperTrend)
        price_deviation = abs(df["c"] - df["SuperTrend"]) / df["SuperTrend"] * 100
        significant_move = price_deviation > 0.5
        df["SuperTrend_Flip_Detected"] = df["SuperTrend_Flip_Detected"] & significant_move & df["Volume_Confirm"]

    # EMA with longer period for better trend following
    df["EMA"] = ta.ema(df["c"], length=ema_length)

    detect_wyckoff_phase(df)
    
    # Add safety check for Wyckoff flip detection
    if len(df['wyckoff_phase']) >= 2:
        wyckoff_flip = (
            df['wyckoff_phase'].iloc[-1] != df['wyckoff_phase'].iloc[-2] and
            not df['uncertain_phase'].iloc[-1]
        )
    else:
        wyckoff_flip = False
        logger.warning("Insufficient data for Wyckoff flip detection")
    
    return df["SuperTrend_Flip_Detected"].iloc[-1], wyckoff_flip


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


def generate_chart(df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, coin: str) -> List[io.BytesIO]:
    chart_buffers = []

    def find_significant_levels(df: pd.DataFrame, n_levels: int = 3) -> Tuple[List[float], List[float]]:
        """Enhanced support and resistance detection with dynamic lookback period"""
        current_price = df['Close'].iloc[-1]
        
        # Dynamic lookback based on timeframe
        timeframe_hours = (df.index[-1] - df.index[-2]).total_seconds() / 3600
        if timeframe_hours <= 1:  # 1h or less
            lookback = 200  # About 8 days for 1h
        elif timeframe_hours <= 4:  # 4h
            lookback = 180  # About 30 days for 4h
        else:  # daily
            lookback = 150  # About 5 months for daily
            
        # Use dynamic recent data points
        recent_df = df.iloc[-lookback:]
        highs = recent_df['High'].values
        lows = recent_df['Low'].values
        volumes = recent_df['Volume'].values
        closes = recent_df['Close'].values
        timestamps = np.arange(len(recent_df))
        
        # Dynamic tolerance based on recent volatility
        atr: float = recent_df['ATR'].iloc[-1]
        recent_volatility = recent_df['Close'].std() / recent_df['Close'].mean()
        base_tolerance = atr * np.log1p(recent_volatility) * 0.35
        
        # Price range limits based on current price
        max_deviation = 0.15  # 15% from current price
        min_price = current_price * (1 - max_deviation)
        max_price = current_price * (1 + max_deviation)
        
        # Price-volume profile with tighter range
        price_steps = 100
        volume_profile = np.zeros(price_steps)
        
        for price, vol in zip(closes, volumes):
            if min_price <= price <= max_price:
                idx = int((price - min_price) / (max_price - min_price) * (price_steps - 1))
                if 0 <= idx < price_steps:
                    volume_profile[idx] += vol
        
        # Smooth volume profile
        volume_profile = np.convolve(volume_profile, np.ones(5)/5, mode='same')
        
        def cluster_points(points: np.ndarray, volumes: np.ndarray, timestamps: np.ndarray) -> Dict[float, Dict[str, float]]:
            clusters: Dict[float, Dict[str, float]] = {}
            
            for idx, price in enumerate(points):
                # Skip prices outside our range of interest
                if not min_price <= price <= max_price:
                    continue
                    
                # Dynamic tolerance based on price level and distance from current price
                price_deviation = abs(price - current_price) / current_price
                local_tolerance = base_tolerance * (1 + price_deviation)
                
                nearby_prices = [p for p in clusters.keys() if abs(p - price) <= local_tolerance]
                
                # Exponential decay for older points
                recency_weight = np.exp(timestamps[idx] / len(timestamps) - 1)
                
                if nearby_prices:
                    main_price = nearby_prices[0]
                    weight = volumes[idx] * recency_weight
                    clusters[main_price]['weight'] += weight
                    clusters[main_price]['count'] += 1
                    clusters[main_price]['vol_sum'] += volumes[idx]
                    # Weighted average favoring recent prices
                    clusters[main_price]['price'] = (
                        clusters[main_price]['price'] * 0.7 +
                        price * 0.3
                    )
                else:
                    clusters[price] = {
                        'weight': volumes[idx] * recency_weight,
                        'count': 1,
                        'vol_sum': volumes[idx],
                        'price': price
                    }
            
            return clusters
        
        resistance_clusters = cluster_points(highs, volumes, timestamps)
        support_clusters = cluster_points(lows, volumes, timestamps)
        
        # Score and filter levels
        def score_level(cluster: Dict[str, float], max_vol: float) -> float:
            """
            Score a price level based on multiple weighted factors.
            Returns a score between 0 and 1.
            """
            current_price = df['Close'].iloc[-1]
            
            # Volume importance (30%)
            volume_ratio = cluster['vol_sum'] / max_vol
            volume_score = min(volume_ratio * 1.5, 1.0) * 0.30
            
            # Touch count importance (25%)
            # Logarithmic scaling with diminishing returns after 5 touches
            touch_count = cluster['count']
            touch_score = (1 - 1/(1 + np.log1p(touch_count))) * 0.25
            
            # Recency weight (20%)
            # Higher weight for more recent activity
            max_weight = max(c['weight'] for c in resistance_clusters.values())
            recency_score = (cluster['weight'] / max_weight) * 0.20
            
            # Price proximity (15%)
            # Exponential decay for prices further from current price
            price_deviation = abs(cluster['price'] - current_price) / current_price
            proximity_score = np.exp(-5 * price_deviation) * 0.15
            
            # Cluster density (10%)
            # Tighter clusters get higher scores
            if touch_count > 1:
                price_range = price_deviation * current_price
                density = cluster['vol_sum'] / (price_range * touch_count + 1e-8)
                density_score = min(density / (max_vol * 0.1), 1.0) * 0.10
            else:
                density_score = 0
            
            # Calculate base score
            base_score = volume_score + touch_score + recency_score + proximity_score + density_score
            
            # Multipliers for special conditions
            multiplier = 1.0
            
            # Strong level bonus (multiple touches with high volume)
            if touch_count >= 3 and volume_ratio > 0.15:
                multiplier *= 1.25
            
            # Recent activity bonus
            if cluster['weight'] > max_weight * 0.8:
                multiplier *= 1.15
            
            # Near current price bonus
            if price_deviation < 0.01:  # Within 1% of current price
                multiplier *= 1.2
            
            # Penalize very distant levels
            if price_deviation > 0.1:  # More than 10% away
                multiplier *= 0.8
            
            return min(base_score * multiplier, 1.0)

        max_vol = max(volumes.sum(), 1)
        
        resistance_levels = sorted(
            [cluster['price'] for cluster in resistance_clusters.values()
             if score_level(cluster, max_vol) > 0.2],  # Increased threshold
            reverse=True
        )[:n_levels]
        
        support_levels = sorted(
            [cluster['price'] for cluster in support_clusters.values()
             if score_level(cluster, max_vol) > 0.2],  # Increased threshold
        )[:n_levels]
        
        return sorted(resistance_levels), sorted(support_levels)

    def save_to_buffer(df: pd.DataFrame, title: str, chart_image_time_delta) -> io.BytesIO:

        from_time = df['t'].max() - chart_image_time_delta

        df_plot = df.loc[df['t'] >= from_time].copy()

        buf = io.BytesIO()
        fig, ax = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        df_plot.loc[:, 'SuperTrend_Green'] = df_plot.apply(
            lambda row: row['SuperTrend'] if row['Close'] > row['SuperTrend'] else float('nan'), 
            axis=1
        )
        df_plot.loc[:, 'SuperTrend_Red'] = df_plot.apply(
            lambda row: row['SuperTrend'] if row['Close'] <= row['SuperTrend'] else float('nan'), 
            axis=1
        )

        ha_df = heikin_ashi(df_plot)

        df_plot.loc[:, 'MACD_Hist'] = df_plot['MACD_Hist'].fillna(0)
        
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

        # Find significant price levels
        resistance_levels, support_levels = find_significant_levels(df)
        
        # Create horizontal lines for each level
        level_lines = []
        for level in resistance_levels:
            line = pd.Series([level] * len(df_plot), index=df_plot.index)
            level_lines.append(mpf.make_addplot(line, ax=ax[0], color='purple', width=1, 
                                              label=f'R {fmt_price(level)}', linestyle='--'))
        
        for level in support_levels:
            line = pd.Series([level] * len(df_plot), index=df_plot.index)
            level_lines.append(mpf.make_addplot(line, ax=ax[0], color='purple', width=1, 
                                              label=f'S {fmt_price(level)}', linestyle=':'))

        # Add current price line using Heikin-Ashi color
        current_price = df_plot['Close'].iloc[-1]
        is_ha_bullish = ha_df['Close'].iloc[-1] >= ha_df['Open'].iloc[-1]  # Use Heikin-Ashi values
        current_price_color = 'green' if is_ha_bullish else 'red'
        current_price_line = pd.Series([current_price] * len(df_plot), index=df_plot.index)
        level_lines.append(mpf.make_addplot(current_price_line, ax=ax[0], color=current_price_color, width=0.5, 
                                          label=f'Current {fmt_price(current_price)}', 
                                          linestyle=':', alpha=0.8))

        mpf.plot(ha_df,
                type='candle',
                ax=ax[0],
                volume=False,
                axtitle=title,
                style='charles',
                addplot=[
                    mpf.make_addplot(df_plot['SuperTrend'], ax=ax[0], color='green', label='SuperTrend', width=0.75),
                    mpf.make_addplot(df_plot['SuperTrend_Red'], ax=ax[0], color='red', width=0.75),
                    mpf.make_addplot(df_plot['VWAP'], ax=ax[0], color='blue', label='VWAP', width=0.75),
                    mpf.make_addplot(df_plot['EMA'], ax=ax[0], color='orange', label='EMA', width=0.75),
                    *level_lines,
                    mpf.make_addplot(df_plot['MACD_Hist'], type='bar', width=0.7, color=macd_hist_colors, ax=ax[1], alpha=0.5, secondary_y=False)
                ])

        ax[0].legend(loc='upper left')

        plt.tight_layout()
        plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
        buf.seek(0)
        plt.close(fig)
        return buf

    df_15m_plot = df_15m.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    chart_buffers.append(save_to_buffer(df_15m_plot, f"{coin} - 15M Chart", pd.Timedelta(hours=36)))

    df_1h_plot = df_1h.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    chart_buffers.append(save_to_buffer(df_1h_plot, f"{coin} - 1H Chart", pd.Timedelta(days=3)))

    df_4h_plot = df_4h.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    chart_buffers.append(save_to_buffer(df_4h_plot, f"{coin} - 4H Chart", pd.Timedelta(days=20)))

    df_1d_plot = df_1d.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
    chart_buffers.append(save_to_buffer(df_1d_plot, f"{coin} - 1D Chart", pd.Timedelta(days=180)))

    return chart_buffers


async def send_trend_change_message(context: ContextTypes.DEFAULT_TYPE, mid: float, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame, df_1d: pd.DataFrame, coin: str) -> None:
    charts = generate_chart(df_15m, df_1h, df_4h, df_1d, coin)

    results_1h = get_ta_results(df_1h, mid)
    results_4h = get_ta_results(df_4h, mid)
    results_1d = get_ta_results(df_1d, mid)

    table_1h = format_table(results_1h)
    table_4h = format_table(results_4h)
    table_1d = format_table(results_1d)

    # Send header
    await telegram_utils.send(
        f"<b>Indicators for {coin}</b>\n"
        f"Market price: {fmt_price(mid)} USDC",
        parse_mode=ParseMode.HTML
    )

    # Send 15m chart
    await telegram_utils.send(
        "15m indicators:",
        parse_mode=ParseMode.HTML
    )
    await context.bot.send_photo(chat_id=telegram_utils.telegram_chat_id, photo=charts[0])

    # Send 1h data and chart
    await telegram_utils.send(
        "1h indicators:\n"
        f"<pre>{table_1h}</pre>",
        parse_mode=ParseMode.HTML
    )
    await context.bot.send_photo(chat_id=telegram_utils.telegram_chat_id, photo=charts[1])

    # Send 4h data and chart
    await telegram_utils.send(
        "4h indicators:\n"
        f"<pre>{table_4h}</pre>",
        parse_mode=ParseMode.HTML
    )
    await context.bot.send_photo(chat_id=telegram_utils.telegram_chat_id, photo=charts[2])

    # Send 1d data and chart
    await telegram_utils.send(
        "1d indicators:\n"
        f"<pre>{table_1d}</pre>",
        parse_mode=ParseMode.HTML
    )
    await context.bot.send_photo(chat_id=telegram_utils.telegram_chat_id, photo=charts[3])


def get_ta_results(df: pd.DataFrame, mid: float) -> Dict[str, Any]:
    # Check if we have enough data points
    if len(df["SuperTrend"]) < 2 or len(df["VWAP"]) < 2:
        logger.warning("Insufficient data for technical analysis results")
        return {
            "supertrend_prev": 0,
            "supertrend": 0,
            "supertrend_trend_prev": "unknown",
            "supertrend_trend": "unknown",
            "vwap_prev": 0,
            "vwap": 0,
            "vwap_trend_prev": "unknown",
            "vwap_trend": "unknown",
            "wyckoff_phase_prev": "unknown",
            "wyckoff_phase": "unknown",
            "wyckoff_volume_prev": "unknown",
            "wyckoff_volume": "unknown",
            "wyckoff_pattern_prev": "unknown",
            "wyckoff_pattern": "unknown"
        }

    supertrend_prev, supertrend = df["SuperTrend"].iloc[-2], df["SuperTrend"].iloc[-1]
    vwap_prev, vwap = df["VWAP"].iloc[-2], df["VWAP"].iloc[-1]

    # Correctly get previous values by accessing index -2
    phase_prev = df['wyckoff_phase'].iloc[-2] if 'wyckoff_phase' in df.columns else "unknown"
    phase = df['wyckoff_phase'].iloc[-1] if 'wyckoff_phase' in df.columns else "unknown"
    volume_prev = df['wyckoff_volume'].iloc[-2] if 'wyckoff_volume' in df.columns else "unknown"
    volume = df['wyckoff_volume'].iloc[-1] if 'wyckoff_volume' in df.columns else "unknown"
    pattern_prev = df['wyckoff_pattern'].iloc[-2] if 'wyckoff_pattern' in df.columns else "unknown"
    pattern = df['wyckoff_pattern'].iloc[-1] if 'wyckoff_pattern' in df.columns else "unknown"

    return {
        "supertrend_prev": supertrend_prev,
        "supertrend": supertrend,
        "supertrend_trend_prev": "uptrend" if df["SuperTrend"].shift().gt(0).iloc[-2] else "downtrend",
        "supertrend_trend": "uptrend" if df["SuperTrend"].shift().gt(0).iloc[-1] else "downtrend",
        "vwap_prev": vwap_prev,
        "vwap": vwap,
        "vwap_trend_prev": "uptrend" if mid > vwap_prev else "downtrend",
        "vwap_trend": "uptrend" if mid > vwap else "downtrend",
        "wyckoff_phase_prev": phase_prev,
        "wyckoff_phase": phase,
        "wyckoff_volume_prev": volume_prev,
        "wyckoff_volume": volume,
        "wyckoff_pattern_prev": pattern_prev,
        "wyckoff_pattern": pattern
    }


def format_table(results: Dict[str, Any]) -> str:
    table_data = [
        ["Supertrend: ", "", ""],
        ["Trend ", results["supertrend_trend_prev"], results["supertrend_trend"]],
        ["Value ", fmt_price(results["supertrend_prev"]), fmt_price(results["supertrend"])],
        ["", "", ""],
        ["VWAP: ", "", ""],
        ["Trend ", results["vwap_trend_prev"], results["vwap_trend"]],
        ["Value ", fmt_price(results["vwap_prev"]), fmt_price(results["vwap"])],
        ["", "", ""],
        ["Wyckoff: ", "", ""],
        ["Phase ", results["wyckoff_phase_prev"], results["wyckoff_phase"]],
        ["Volume ", results["wyckoff_volume_prev"], results["wyckoff_volume"]],
        ["Pattern ", results["wyckoff_pattern_prev"], results["wyckoff_pattern"]]
    ]
    
    return tabulate(
        table_data,
        headers=["", "Previous", "Current"],
        tablefmt=simple_separated_format(" "),
        colalign=("right", "right", "right"),
    )
