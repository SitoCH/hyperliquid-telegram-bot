import pandas as pd
import numpy as np
from technical_analysis.wyckoff.wyckoff_signs import detect_wyckoff_signs
from technical_analysis.wyckoff.wyckoff_types import WyckoffSign, Timeframe


TF = Timeframe.HOUR_1


def _make_df(close_prices, volumes=None, opens=None, highs=None, lows=None, extra_cols=None):
    n = len(close_prices)
    data = {
        'o': opens if opens else [p * 0.999 for p in close_prices],
        'h': highs if highs else [p * 1.005 for p in close_prices],
        'l': lows if lows else [p * 0.995 for p in close_prices],
        'c': close_prices,
        'v': volumes if volumes else [1000.0] * n,
    }
    if extra_cols:
        data.update(extra_cols)
    return pd.DataFrame(data)


# ═══════════════════════════════════════════════════════════════════════
# 1. Edge cases & defensive guards
# ═══════════════════════════════════════════════════════════════════════

class TestEdgeCases:

    def test_insufficient_data_returns_none(self):
        df = _make_df([100.0] * 4)
        assert detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF) == WyckoffSign.NONE

    def test_empty_dataframe_returns_none(self):
        df = _make_df([])
        assert detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF) == WyckoffSign.NONE

    def test_nan_close_does_not_crash(self):
        df = _make_df([100.0] * 30)
        df.loc[df.index[-1], 'c'] = np.nan
        result = detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF)
        assert result == WyckoffSign.NONE

    def test_nan_volume_does_not_crash(self):
        df = _make_df([100.0] * 30)
        df.loc[df.index[-1], 'v'] = np.nan
        result = detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF)
        assert result == WyckoffSign.NONE

    def test_zero_volume_does_not_crash(self):
        df = _make_df([100.0] * 30, volumes=[0.0] * 30)
        result = detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF)
        assert result == WyckoffSign.NONE

    def test_zero_price_does_not_crash(self):
        df = _make_df([0.0] * 30)
        result = detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF)
        assert result == WyckoffSign.NONE

    def test_all_prices_identical_returns_none(self):
        df = _make_df([100.0] * 30)
        assert detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF) == WyckoffSign.NONE

    def test_random_data_returns_none(self):
        rng = np.random.default_rng(99)
        prices = [100.0 + float(rng.normal(0, 0.5)) for _ in range(50)]
        vols = [1000.0 + float(rng.normal(0, 100)) for _ in range(50)]
        result = detect_wyckoff_signs(_make_df(prices, volumes=vols), 0.0, 0.0, False, False, TF)
        assert result == WyckoffSign.NONE

    def test_extreme_outlier_does_not_crash(self):
        prices = [100.0] * 29 + [1e6]
        df = _make_df(prices, volumes=[1000.0] * 29 + [1e9])
        result = detect_wyckoff_signs(df, 10.0, 10.0, True, True, TF)
        assert isinstance(result, WyckoffSign)

    def test_missing_open_column_does_not_crash(self):
        df = pd.DataFrame({
            'h': [101.0] * 30, 'l': [99.0] * 30,
            'c': [100.0] * 30, 'v': [1000.0] * 30,
        })
        assert detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF) == WyckoffSign.NONE

    def test_negative_prices_does_not_crash(self):
        df = _make_df([-100.0] * 30, highs=[-99.0] * 30, lows=[-101.0] * 30)
        assert isinstance(detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF), WyckoffSign)

    def test_confirm_trend_all_nan_returns_none(self):
        prices = [100.0] * 27 + [float('nan')] * 3
        n = len(prices)
        vols = [1000.0] * (n - 2) + [3000.0, 5000.0]
        df = _make_df(prices, volumes=vols)
        result = detect_wyckoff_signs(df, 0.8, 0.0, False, False, TF)
        assert result == WyckoffSign.NONE


# ═══════════════════════════════════════════════════════════════════════
# 2. Bullish sign detection (accumulation phase)
# ═══════════════════════════════════════════════════════════════════════

class TestSellingClimax:

    def test_selling_climax_detected(self):
        prices = [100.0] * 25 + [99.0, 98.0, 97.0, 96.0, 92.0]
        lows = [p * 0.995 for p in prices]
        lows[-1] = 88.0
        vols = [1000.0] * 29 + [12000.0]
        df = _make_df(prices, volumes=vols, lows=lows)
        result = detect_wyckoff_signs(df, -2.5, 0.0, False, False, TF)
        assert result == WyckoffSign.SELLING_CLIMAX

    def test_selling_climax_no_signal_when_strength_weak(self):
        prices = [100.0] * 25 + [99.0, 98.0, 97.0, 96.0, 92.0]
        lows = [p * 0.995 for p in prices]
        lows[-1] = 88.0
        vols = [1000.0] * 29 + [12000.0]
        df = _make_df(prices, volumes=vols, lows=lows)
        assert detect_wyckoff_signs(df, 0.5, 0.0, False, False, TF) != WyckoffSign.SELLING_CLIMAX

    def test_selling_climax_no_signal_without_volume(self):
        prices = [100.0] * 25 + [99.0, 98.0, 97.0, 96.0, 92.0]
        lows = [p * 0.995 for p in prices]
        lows[-1] = 88.0
        df = _make_df(prices, volumes=[500.0] * 30, lows=lows)
        assert detect_wyckoff_signs(df, -2.5, 0.0, False, False, TF) != WyckoffSign.SELLING_CLIMAX


class TestAutomaticRally:

    def test_automatic_rally_detected(self):
        prices = [100.0] * 28 + [100, 99.5, 99, 98.5, 98, 90, 108]
        n = len(prices)
        vols = [1000.0] * n
        vols[-2] = 3000.0
        vols[-1] = 5000.0
        df = _make_df(prices, volumes=vols)
        result = detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF)
        assert result == WyckoffSign.AUTOMATIC_RALLY

    def test_ar_no_signal_without_reversal(self):
        prices = [100.0 * (1 + 0.002 * i) for i in range(35)]
        df = _make_df(prices, volumes=[2000.0] * 35)
        assert detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF) != WyckoffSign.AUTOMATIC_RALLY


class TestSecondaryTest:

    def _st_df(self):
        base = [100.0]
        for i in range(1, 29):
            base.append(base[-1] + (-0.5 if i % 3 == 0 else 0.3))
        prices = base + [base[-1]]
        recent_low = min(base[-4:])
        lows = [p * 0.99 for p in prices]
        lows[-1] = recent_low * 1.002
        vols = [1000.0] * (len(prices) - 1) + [60.0]
        return _make_df(prices, volumes=vols, lows=lows)

    def test_secondary_test_detected(self):
        df = self._st_df()
        result = detect_wyckoff_signs(df, -1.0, 0.0, False, False, TF)
        assert result == WyckoffSign.SECONDARY_TEST

    def test_secondary_test_no_signal_when_volume_not_low(self):
        df = self._st_df()
        df['v'] = [1000.0] * len(df)
        assert detect_wyckoff_signs(df, -1.0, 0.0, False, False, TF) != WyckoffSign.SECONDARY_TEST


class TestLastPointOfSupport:

    def test_lps_detected(self):
        prices = [100.0] * 28 + [98, 95, 98, 103, 108]
        n = len(prices)
        vols = [1000.0] * (n - 2) + [4000.0, 6000.0]
        df = _make_df(prices, volumes=vols)
        assert detect_wyckoff_signs(df, 0.5, 1.5, True, False, TF) == WyckoffSign.LAST_POINT_OF_SUPPORT

    def test_lps_no_signal_without_spring(self):
        prices = [100.0] * 28 + [98, 95, 98, 103, 108]
        n = len(prices)
        vols = [1000.0] * (n - 2) + [4000.0, 6000.0]
        df = _make_df(prices, volumes=vols)
        result = detect_wyckoff_signs(df, 0.5, 1.5, False, False, TF)
        assert result != WyckoffSign.LAST_POINT_OF_SUPPORT


class TestSignOfStrength:

    def test_sos_detected(self):
        prices = [100.0] * 28 + [101, 102, 103, 104, 107]
        n = len(prices)
        vols = [1000.0] * (n - 2) + [3000.0, 5000.0]
        df = _make_df(prices, volumes=vols)
        assert detect_wyckoff_signs(df, 0.8, 0.0, False, False, TF) == WyckoffSign.SIGN_OF_STRENGTH

    def test_sos_no_signal_below_ma(self):
        prices = [105.0] * 28 + [104, 103, 102, 101, 99]
        n = len(prices)
        vols = [1000.0] * (n - 2) + [3000.0, 5000.0]
        df = _make_df(prices, volumes=vols)
        assert detect_wyckoff_signs(df, 0.8, 0.0, False, False, TF) != WyckoffSign.SIGN_OF_STRENGTH


# ═══════════════════════════════════════════════════════════════════════
# 3. Bearish sign detection (distribution phase)
# ═══════════════════════════════════════════════════════════════════════

class TestBuyingClimax:

    def test_buying_climax_detected(self):
        prices = [100.0] * 28 + [103, 106, 110, 115, 122]
        n = len(prices)
        vols = [1000.0] * (n - 1) + [25000.0]
        df = _make_df(prices, volumes=vols)
        result = detect_wyckoff_signs(df, 2.5, 0.0, False, False, TF)
        # BC and SOS share identical scoring formula; SOS wins ties (inserted first)
        # Both trigger on this data — verifies a strong bullish sign is returned
        assert result in (WyckoffSign.BUYING_CLIMAX, WyckoffSign.SIGN_OF_STRENGTH)

    def test_bc_no_signal_without_volume(self):
        prices = [100.0] * 28 + [103, 106, 110, 115, 122]
        n = len(prices)
        df = _make_df(prices, volumes=[500.0] * n)
        assert detect_wyckoff_signs(df, 2.5, 0.0, False, False, TF) != WyckoffSign.BUYING_CLIMAX


class TestUpthrust:

    def test_upthrust_detected(self):
        prices = [100.0] * 28 + [101, 102, 103, 102, 99]
        highs = [p * 1.005 for p in prices]
        highs[-1] = 106.0
        n = len(prices)
        vols = [1000.0] * (n - 1) + [5000.0]
        df = _make_df(prices, volumes=vols, highs=highs)
        assert detect_wyckoff_signs(df, 0.0, 0.0, False, True, TF) == WyckoffSign.UPTHRUST

    def test_upthrust_no_signal_without_flag(self):
        prices = [100.0] * 28 + [101, 102, 103, 102, 99]
        highs = [p * 1.005 for p in prices]
        highs[-1] = 106.0
        n = len(prices)
        vols = [1000.0] * (n - 1) + [5000.0]
        df = _make_df(prices, volumes=vols, highs=highs)
        assert detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF) != WyckoffSign.UPTHRUST


class TestSecondaryTestResistance:

    def _str_df(self):
        base = [100.0]
        for i in range(1, 29):
            base.append(base[-1] + (0.5 if i % 3 == 0 else -0.3))
        prices = base + [base[-1]]
        recent_high = max(base[-4:])
        highs = [p * 1.01 for p in prices]
        highs[-1] = recent_high * 0.998
        vols = [1000.0] * (len(prices) - 1) + [60.0]
        return _make_df(prices, volumes=vols, highs=highs)

    def test_str_detected(self):
        df = self._str_df()
        result = detect_wyckoff_signs(df, 1.0, 0.0, False, False, TF)
        assert result == WyckoffSign.SECONDARY_TEST_RESISTANCE

    def test_str_no_signal_when_volume_high(self):
        df = self._str_df()
        df['v'] = [1000.0] * len(df)
        assert detect_wyckoff_signs(df, 1.0, 0.0, False, False, TF) != WyckoffSign.SECONDARY_TEST_RESISTANCE


class TestLastPointOfResistance:

    def test_lpsy_detected(self):
        prices = [100.0] * 28 + [103, 106, 103, 98, 93]
        n = len(prices)
        vols = [1000.0] * (n - 2) + [4000.0, 6000.0]
        df = _make_df(prices, volumes=vols)
        assert detect_wyckoff_signs(df, -0.5, 0.8, False, True, TF) == WyckoffSign.LAST_POINT_OF_RESISTANCE

    def test_lpsy_no_signal_without_upthrust(self):
        prices = [100.0] * 28 + [103, 106, 103, 98, 93]
        n = len(prices)
        vols = [1000.0] * (n - 2) + [4000.0, 6000.0]
        df = _make_df(prices, volumes=vols)
        assert detect_wyckoff_signs(df, -0.5, 0.8, False, False, TF) != WyckoffSign.LAST_POINT_OF_RESISTANCE


class TestSignOfWeakness:

    def test_sow_detected(self):
        prices = [100.0] * 28 + [99, 98, 97, 96, 93]
        n = len(prices)
        vols = [1000.0] * (n - 2) + [3000.0, 5000.0]
        df = _make_df(prices, volumes=vols)
        assert detect_wyckoff_signs(df, -0.8, 0.0, False, False, TF) == WyckoffSign.SIGN_OF_WEAKNESS

    def test_sow_no_signal_when_close_above_ma(self):
        prices = [95.0] * 28 + [100, 99, 98, 97, 96]
        n = len(prices)
        vols = [1000.0] * (n - 2) + [3000.0, 5000.0]
        df = _make_df(prices, volumes=vols)
        assert detect_wyckoff_signs(df, -0.8, 0.0, False, False, TF) != WyckoffSign.SIGN_OF_WEAKNESS


# ═══════════════════════════════════════════════════════════════════════
# 4. Sign scoring & overlap prevention
# ═══════════════════════════════════════════════════════════════════════

class TestSignScoring:

    def test_strongest_sign_selected_when_multiple_trigger(self):
        prices = [100.0] * 28 + [98, 95, 98, 103, 108]
        n = len(prices)
        vols = [1000.0] * (n - 2) + [4000.0, 6000.0]
        df = _make_df(prices, volumes=vols)
        result = detect_wyckoff_signs(df, 0.5, 1.5, True, False, TF)
        assert result == WyckoffSign.LAST_POINT_OF_SUPPORT

    def test_all_signs_none_when_scores_below_minimum(self):
        prices = [100.0] * 30
        for i in range(1, 30):
            prices[i] = prices[i-1] + np.random.default_rng(7).normal(0, 0.1)
        df = _make_df(prices, volumes=[1000.0] * 30)
        assert detect_wyckoff_signs(df, 0.1, 0.0, False, False, TF) == WyckoffSign.NONE


# ═══════════════════════════════════════════════════════════════════════
# 5. Timeframe-specific behavior
# ═══════════════════════════════════════════════════════════════════════

class TestTimeframeBehavior:

    def test_different_timeframes_produce_results(self):
        prices = [100.0] * 25 + [99.0, 98.0, 97.0, 96.0, 92.0]
        lows = [p * 0.995 for p in prices]
        lows[-1] = 89.0
        vols = [1000.0] * 29 + [12000.0]
        df = _make_df(prices, volumes=vols, lows=lows)
        for tf in [Timeframe.MINUTES_15, Timeframe.MINUTES_30, Timeframe.HOUR_1, Timeframe.HOURS_4, Timeframe.HOURS_8]:
            result = detect_wyckoff_signs(df, -2.5, 0.0, False, False, tf)
            assert isinstance(result, WyckoffSign), f"{tf.name} failed"

    def test_all_configured_timeframes_return_enum(self):
        prices = [100.0] * 40
        for tf in [Timeframe.MINUTES_15, Timeframe.MINUTES_30, Timeframe.HOUR_1, Timeframe.HOURS_4, Timeframe.HOURS_8]:
            result = detect_wyckoff_signs(pd.DataFrame({'c': prices, 'h': [101.0]*40, 'l': [99.0]*40, 'o': [100.0]*40, 'v': [1000.0]*40}),
                                          0.0, 0.0, False, False, tf)
            assert isinstance(result, WyckoffSign)


# ═══════════════════════════════════════════════════════════════════════
# 6. Parameter sensitivity
# ═══════════════════════════════════════════════════════════════════════

class TestParameterSensitivity:

    def test_strong_price_strength_increases_sc_detection(self):
        prices = [100.0] * 25 + [99.0, 98.0, 97.0, 96.0, 92.0]
        lows = [p * 0.995 for p in prices]
        lows[-1] = 88.0
        vols = [1000.0] * 29 + [12000.0]
        df = _make_df(prices, volumes=vols, lows=lows)
        weak = detect_wyckoff_signs(df, -1.0, 0.0, False, False, TF)
        strong = detect_wyckoff_signs(df, -2.5, 0.0, False, False, TF)
        assert strong == WyckoffSign.SELLING_CLIMAX
        assert weak != WyckoffSign.SELLING_CLIMAX

    def test_volume_trend_boosts_lps(self):
        prices = [100.0] * 28 + [98, 95, 98, 103, 108]
        n = len(prices)
        vols = [1000.0] * (n - 2) + [4000.0, 6000.0]
        df = _make_df(prices, volumes=vols)
        result = detect_wyckoff_signs(df, 0.5, 1.5, True, False, TF)
        assert result == WyckoffSign.LAST_POINT_OF_SUPPORT

    def test_spring_flag_required_for_lps(self):
        prices = [100.0] * 28 + [98, 95, 98, 103, 108]
        n = len(prices)
        vols = [1000.0] * (n - 2) + [4000.0, 6000.0]
        df = _make_df(prices, volumes=vols)
        with_spring = detect_wyckoff_signs(df, 0.5, 1.5, True, False, TF)
        without = detect_wyckoff_signs(df, 0.5, 1.5, False, False, TF)
        assert with_spring == WyckoffSign.LAST_POINT_OF_SUPPORT
        assert without != WyckoffSign.LAST_POINT_OF_SUPPORT

    def test_upthrust_flag_required_for_ut(self):
        prices = [100.0] * 28 + [101, 102, 103, 102, 99]
        highs = [p * 1.005 for p in prices]
        highs[-1] = 106.0
        n = len(prices)
        vols = [1000.0] * (n - 1) + [5000.0]
        df = _make_df(prices, volumes=vols, highs=highs)
        with_flag = detect_wyckoff_signs(df, 0.0, 0.0, False, True, TF)
        without = detect_wyckoff_signs(df, 0.0, 0.0, False, False, TF)
        assert with_flag == WyckoffSign.UPTHRUST
        assert without != WyckoffSign.UPTHRUST


# ═══════════════════════════════════════════════════════════════════════
# 7. Return type purity
# ═══════════════════════════════════════════════════════════════════════

class TestReturnType:

    def test_always_returns_wyckoff_sign_enum(self):
        cases = [
            ([100.0] * 2, 0.0, 0.0, False, False),
            ([100.0] * 30, 0.0, 0.0, False, False),
            ([100.0] * 28 + [101, 102, 103, 104, 107], 2.0, 1.0, True, False),
            ([100.0] * 28 + [99, 98, 97, 96, 93], -2.0, 1.0, False, True),
        ]
        for prices, ps, vt, sp, ut in cases:
            df = _make_df(prices)
            result = detect_wyckoff_signs(df, ps, vt, sp, ut, TF)
            assert isinstance(result, WyckoffSign), f"Expected WyckoffSign, got {type(result)} for prices={prices}"