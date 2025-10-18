import pandas as pd  # type: ignore[import]
import numpy as np  # type: ignore[import]
from typing import Final
from .wyckoff_types import WyckoffSign, Timeframe

# Constants for Wyckoff sign detection
STRONG_DEV_THRESHOLD: Final[float] = 2.1

def detect_wyckoff_signs(
    df: pd.DataFrame,
    price_strength: float,
    volume_trend: float,
    is_spring: bool,
    is_upthrust: bool,
    timeframe: Timeframe
) -> WyckoffSign:
    """
    Detect specific Wyckoff signs in market action optimized for crypto intraday trading.
    Uses dedicated parameters from timeframe settings for consistent sign detection.
    More conservative approach - only returns a sign when evidence is strong.
    """
    if len(df) < 5:
        return WyckoffSign.NONE

    # Calculate key metrics with noise reduction
    price_change = df['c'].pct_change()
    # Avoid using volume pct_change directly (noisy/unstable). We'll use ratio/z later.
    
    # Use timeframe-specific lookback periods from settings
    volatility_window = max(5, timeframe.settings.spring_upthrust_window)
    volume_ma_window = max(5, timeframe.settings.volume_ma_window // 3)
    price_ma_window = max(8, int(timeframe.settings.ema_length * 0.75))
    
    # Use timeframe settings for recent high/low lookback
    recent_window = timeframe.settings.swing_lookback
    
    # Calculate rolling metrics with adaptive windows (NaN-safe)
    # Price volatility based on last-N non-NaN returns
    price_ret = price_change
    valid_rets = price_ret.dropna()
    vol_win = min(max(3, volatility_window), len(valid_rets))
    price_volatility = float(valid_rets.iloc[-vol_win:].std(ddof=0)) if vol_win > 0 else 0.0
    if not np.isfinite(price_volatility) or price_volatility == 0:
        price_volatility = float(valid_rets.std(ddof=0)) if len(valid_rets) else 0.004
        if not np.isfinite(price_volatility) or price_volatility == 0:
            price_volatility = 0.004  # small fallback

    # Volume baselines
    volume_ma = df['v'].rolling(volume_ma_window, min_periods=max(3, volume_ma_window // 2)).mean()
    volume_std = df['v'].rolling(volume_ma_window, min_periods=max(3, volume_ma_window // 2)).std(ddof=0)
    price_ma = df['c'].rolling(price_ma_window, min_periods=max(4, price_ma_window // 2)).mean()
    
    # Use dedicated Wyckoff sign detection parameters from settings
    volatility_factor = timeframe.settings.wyckoff_volatility_factor
    trend_lookback = timeframe.settings.wyckoff_trend_lookback
    lps_volume_threshold = timeframe.settings.wyckoff_lps_volume_threshold
    lps_price_multiplier = timeframe.settings.wyckoff_lps_price_multiplier
    sos_multiplier = timeframe.settings.wyckoff_sos_multiplier
    ut_multiplier = timeframe.settings.wyckoff_ut_multiplier
    sc_multiplier = timeframe.settings.wyckoff_sc_multiplier
    ar_multiplier = timeframe.settings.wyckoff_ar_multiplier
    confirmation_threshold = timeframe.settings.wyckoff_confirmation_threshold

    # Use dynamic thresholds based on market volatility
    # Increase minimum thresholds by 25% to be more conservative
    min_price_move = max(0.004, price_volatility * 1.75 * volatility_factor)
    # Volume: prefer level-based metrics over pct_change
    current_volume = float(df['v'].iloc[-1])
    vol_ma_last = float(volume_ma.iloc[-1]) if np.isfinite(volume_ma.iloc[-1]) else float(df['v'].mean())
    if vol_ma_last == 0:
        vol_ma_last = max(1.0, float(df['v'].median()) if len(df) else 1.0)
    vol_sd_last = float(volume_std.iloc[-1]) if np.isfinite(volume_std.iloc[-1]) else float(df['v'].std(ddof=0))
    if not np.isfinite(vol_sd_last) or vol_sd_last == 0:
        vol_sd_last = max(1.0, float(df['v'].std(ddof=0)) if len(df) else 1.0)
    volume_ratio = current_volume / max(1e-9, vol_ma_last)
    volume_z = (current_volume - vol_ma_last) / max(1e-9, vol_sd_last)
    # Dynamic volume thresholds (tunable per timeframe)
    min_volume_ratio = 1.8 * volatility_factor
    min_volume_z = 1.5 * volatility_factor
    
    # Detect market context for better signal relevance
    baseline_std = price_ret.rolling(volatility_window * 3, min_periods=max(volatility_window, 3)).std(ddof=0).mean()
    baseline_std = float(baseline_std) if np.isfinite(baseline_std) else price_volatility
    is_high_volatility = price_volatility > baseline_std * 1.5
    recency_factor = 3 if is_high_volatility else 2

    # Improved confirmation functions with direct parameter usage and higher thresholds
    def confirm_trend(window: int, threshold: float, direction: int = 1) -> bool:
        """Check if price movement confirms a trend in the given direction with stricter, volatility-normalized criteria.
        threshold is an absolute move; we normalize returns by current volatility."""
        window = max(1, min(window, len(df) - 1))
        recent_changes = price_change.iloc[-window:]
        if len(recent_changes.dropna()) == 0:
            return False
        # Adjust sensitivity based on volatility - be more conservative
        sensitivity_factor = 0.9 if is_high_volatility else 1.0
        adjusted_threshold = threshold * sensitivity_factor
        # Normalize by current volatility
        k = adjusted_threshold / max(1e-9, price_volatility)
        weights = np.linspace(1, recency_factor, len(recent_changes))
        rc = recent_changes.fillna(0.0) / max(1e-9, price_volatility)
        if direction == 1:  # Up trend
            weighted_confirms = (rc > k) * weights
        else:  # Down trend
            weighted_confirms = (rc < -k) * weights

        return float(np.nansum(weighted_confirms)) >= float(np.sum(weights)) * confirmation_threshold
        
    def confirm_volume(window: int, min_ratio: float, min_z: float) -> bool:
        """Check if volume confirms a significant move using ratio and z-score with stricter criteria."""
        window = max(1, min(window, len(df)))
        recent_v = df['v'].iloc[-window:]
        # Rolling stats within the window
        recent_ma = recent_v.rolling(window, min_periods=max(2, window // 2)).mean()
        recent_sd = recent_v.rolling(window, min_periods=max(2, window // 2)).std(ddof=0)
        ratios = recent_v / recent_ma.replace(0, np.nan)
        zs = (recent_v - recent_ma) / recent_sd.replace(0, np.nan)
        # Adjust thresholds slightly in high vol regimes
        adj_ratio = min_ratio * (0.9 if is_high_volatility else 1.0)
        adj_z = min_z * (0.9 if is_high_volatility else 1.0)
        weights = np.linspace(1, recency_factor, len(recent_v))
        cond = ((ratios > adj_ratio) | (zs > adj_z)).fillna(False)
        weighted_confirms = (cond.astype(float) * weights)
        stricter_threshold = confirmation_threshold * 1.15
        return float(np.nansum(weighted_confirms)) >= float(np.sum(weights)) * stricter_threshold

    # Check for recent trend reversal
    recent_trend_changed = False
    if len(df) > (trend_lookback * 2):
        prev_trend = np.sign(df['c'].pct_change(trend_lookback).iloc[-trend_lookback-1])
        current_trend = np.sign(df['c'].pct_change(trend_lookback).iloc[-1])
        recent_trend_changed = prev_trend != current_trend and abs(current_trend) > 0

    # Create context variables using timeframe-specific recent window
    recent_low = df['l'].iloc[-recent_window:].min()
    recent_high = df['h'].iloc[-recent_window:].max()
    current_close = df['c'].iloc[-1]
    # Safe MAs and derived features
    price_ma_last = float(price_ma.iloc[-1]) if np.isfinite(price_ma.iloc[-1]) else float(df['c'].rolling(max(3, price_ma_window//2), min_periods=3).mean().iloc[-1])
    if not np.isfinite(price_ma_last) or price_ma_last == 0:
        price_ma_last = float(df['c'].mean()) if len(df) else current_close
    price_distance_from_ma = current_close / max(1e-9, price_ma_last) - 1
    vol_ratio_current = max(1.0, volume_ratio)

    # Candle shape metrics for wick-based patterns (handle missing 'o')
    open_col = 'o' if 'o' in df.columns else None
    open_price = float(df['o'].iloc[-1]) if open_col else (float(df['c'].iloc[-2]) if len(df) > 1 else float(df['c'].iloc[-1]))
    high = float(df['h'].iloc[-1])
    low = float(df['l'].iloc[-1])
    close = float(df['c'].iloc[-1])
    body = abs(close - open_price)
    rng = max(1e-9, high - low)
    upper_wick = max(0.0, high - max(close, open_price))
    lower_wick = max(0.0, min(close, open_price) - low)
    upper_wick_ratio = upper_wick / rng
    lower_wick_ratio = lower_wick / rng
    
    # Implement sign detection overlap prevention
    # Only return the strongest sign when multiple conditions are met
    sign_scores = {}
    
    # Selling Climax (SC) - more stringent conditions
    if (price_change.iloc[-1] < -min_price_move * sc_multiplier * 1.2 and
        ((volume_ratio > min_volume_ratio * 1.3) or (volume_z > min_volume_z)) and
        price_strength < -STRONG_DEV_THRESHOLD * 0.9 and
        price_distance_from_ma < -0.04 and
        lower_wick_ratio > 0.5 and  # long lower wick
        (low <= recent_low * 1.005) and  # near recent low
        confirm_volume(3, min_volume_ratio * 0.8, min_volume_z * 0.8)):
        sign_scores[WyckoffSign.SELLING_CLIMAX] = abs(float(price_change.iloc[-1])) * max(volume_ratio, 1.0)
        
    # Automatic Rally (AR) - more stringent conditions
    if (price_change.iloc[-1] > min_price_move * ar_multiplier * 1.2 and
        price_change.iloc[-2] < -min_price_move * 0.5 and  # Require prior decline
        df['l'].iloc[-1] > recent_low and
        price_strength > -0.1 and
        (volume_ratio > 1.1 or volume_z > 0.5) and
        recent_trend_changed and
        confirm_trend(2, min_price_move * 0.8)):
        sign_scores[WyckoffSign.AUTOMATIC_RALLY] = float(price_change.iloc[-1]) * max(1.0, volume_ratio)
        
    # Secondary Test (ST) - stricter tolerance range
    if (abs(price_change.iloc[-1]) < price_volatility * 0.6 and  # Stricter volatility threshold
        df['l'].iloc[-1] >= recent_low * 1.001 and  # Simplified and tighter tolerance
        df['l'].iloc[-1] <= recent_low * 1.015 and  # Upper bound more restrictive
        current_volume < vol_ma_last * 0.7 and  # Lower volume requirement
        price_strength < -0.5 and  # Stronger negative price strength
        df['v'].iloc[-1] < df['v'].iloc[-5:].min() * 1.2):  # Ensure truly decreased volume
        # Modified scoring formula to reduce sensitivity
        sign_scores[WyckoffSign.SECONDARY_TEST] = 0.7 / (abs(df['l'].iloc[-1] / recent_low - 1.0) + 0.03)

    # Last Point of Support (LPS)
    if (is_spring and
        volume_trend > lps_volume_threshold * 1.2 and
        price_change.iloc[-1] > min_price_move * lps_price_multiplier * 1.2 and
        price_strength < STRONG_DEV_THRESHOLD * 0.4 and
        confirm_trend(3, min_price_move * 0.6)):
        sign_scores[WyckoffSign.LAST_POINT_OF_SUPPORT] = price_change.iloc[-1] * volume_trend
        
    # Sign of Strength (SOS)
    if (price_change.iloc[-1] > min_price_move * sos_multiplier * 1.2 and
        confirm_trend(3, min_price_move * 0.8) and
        (volume_ratio > 1.2 or volume_z > 0.7) and  # Higher volume requirement
        price_strength > 0.4 and  # Higher strength threshold
        df['c'].iloc[-1] > price_ma_last * 1.01):  # Must be clearly above MA
        sign_scores[WyckoffSign.SIGN_OF_STRENGTH] = float(price_change.iloc[-1]) * max(1.0, volume_ratio)
        
    # Buying Climax (BC)
    if (price_change.iloc[-1] > min_price_move * 2.0 and  # Increased from 1.8
        ((volume_ratio > min_volume_ratio * 1.2) or (volume_z > min_volume_z)) and 
        price_strength > STRONG_DEV_THRESHOLD * 0.9 and  # Higher strength
        price_distance_from_ma > 0.05 and  # Further from MA
        confirm_volume(3, min_volume_ratio * 0.8, min_volume_z * 0.8)):
        sign_scores[WyckoffSign.BUYING_CLIMAX] = float(price_change.iloc[-1]) * max(1.0, volume_ratio)
        
    # Upthrust (UT)
    if (is_upthrust and
        ((volume_ratio > min_volume_ratio * 0.8) or (volume_z > min_volume_z * 0.8)) and
        price_change.iloc[-1] < -min_price_move * ut_multiplier * 1.1 and
        (df['c'].iloc[-1] / df['h'].iloc[-1]) < 0.985 and  # Stricter closing ratio
        upper_wick_ratio > 0.5 and  # long upper wick
        (high >= recent_high * 0.995)):
        sign_scores[WyckoffSign.UPTHRUST] = abs(float(price_change.iloc[-1])) * max(1.0, volume_ratio)

    # Secondary Test Resistance (STR) - stricter tolerance range
    if (abs(price_change.iloc[-1]) < price_volatility * 0.6 and  # Stricter volatility threshold
        df['h'].iloc[-1] <= recent_high * 0.999 and  # Simplified and tighter tolerance
        df['h'].iloc[-1] >= recent_high * 0.985 and  # Lower bound more restrictive
        current_volume < vol_ma_last * 0.7 and  # Lower volume requirement
        price_strength > 0.5 and  # Stronger positive price strength
        df['v'].iloc[-1] < df['v'].iloc[-5:].min() * 1.2):  # Ensure truly decreased volume
        # Modified scoring formula to reduce sensitivity
        sign_scores[WyckoffSign.SECONDARY_TEST_RESISTANCE] = 0.7 / (abs(df['h'].iloc[-1] / recent_high - 1.0) + 0.03)

    # Last Point of Supply (LPSY)
    if (is_upthrust and
        volume_trend > lps_volume_threshold * 1.2 and
        price_change.iloc[-1] < -min_price_move * lps_price_multiplier * 1.2 and
        price_strength > -STRONG_DEV_THRESHOLD * 0.4 and
        confirm_trend(3, min_price_move * 0.6, -1)):
        sign_scores[WyckoffSign.LAST_POINT_OF_RESISTANCE] = abs(float(price_change.iloc[-1])) * float(volume_trend)

    # Sign of Weakness (SOW)
    if (price_change.iloc[-1] < -min_price_move * sos_multiplier * 1.2 and
        confirm_trend(3, min_price_move * 0.8, -1) and
        (volume_ratio > 1.2 or volume_z > 0.7) and  # Higher volume requirement
        price_strength < -0.4 and  # Lower strength threshold
        df['c'].iloc[-1] < price_ma_last * 0.99):  # Must be clearly below MA
        sign_scores[WyckoffSign.SIGN_OF_WEAKNESS] = abs(float(price_change.iloc[-1])) * max(1.0, volume_ratio)

    # Return the strongest sign if it exists, otherwise NONE
    if sign_scores:
        # Normalize scores to reduce bias from absolute magnitudes
        norm = max(1e-9, price_volatility * vol_ratio_current)
        scored = {k: (float(v) / norm) for k, v in sign_scores.items()}
        best_sign, best_score = max(scored.items(), key=lambda x: x[1])
        # Require a minimum confidence depending on regime
        min_conf = 0.6 if is_high_volatility else 0.5
        if best_score >= min_conf:
            return best_sign

    return WyckoffSign.NONE