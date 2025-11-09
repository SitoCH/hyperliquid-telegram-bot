import os
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
from ..wyckoff.wyckoff_types import Timeframe
from ..funding_rates_cache import FundingRateEntry


class LLMPromptGenerator:
    """
    Handles generation of prompts for LLM analysis.
    """
    
    def __init__(self, timeframe_lookback_days: Dict[Timeframe, int]):
        self.timeframe_lookback_days = timeframe_lookback_days
    
    def generate_prediction_prompt(
        self, 
        coin: str, 
        dataframes: Dict[Timeframe, pd.DataFrame], 
        funding_rates: List[FundingRateEntry], 
        mid: float
    ) -> str:
        """
        Generate a prompt for LLM to predict price movements based on candles and funding rates.
        
        Args:
            coin: The cryptocurrency symbol
            dataframes: Dictionary of timeframes and their candle data
            funding_rates: List of funding rate entries
            mid: Current mid price
            
        Returns:
            Formatted prompt string for LLM
        """        
        
        current_time = datetime.now()
        prompt_parts = [
            f"TRADING SNAPSHOT for {coin}",
            f"Price: ${mid:.4f}",
            f"Timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "Objective: Produce a concise JSON trading decision for the next 3-4 hours (scalp/swing intraday).",
            "Default to HOLD unless a CLEAR, RISK-CONTROLLED LONG or SHORT setup exists.",
            "",
            "=== TIMEFRAME POLICY (weights) ===",
            "15m: 35% (entry + micro trend)",
            "30m: 33% (intraday directional backbone)",
            "1h: 22% (confirmation & context)",
            "4h: 7% (structural regime only; never blocks lower TF alignment)",
            "",
            "Data freshness: latest completed candles only; price may have moved since last close.",
            f"24h change (approx): {self._calculate_24h_change(dataframes.get(Timeframe.HOUR_1), mid):.2f}%",
            "",
            "Signal Criteria (simplified):",
            "Core REQUIREMENTS for LONG or SHORT (must have both A and B):",
            "A. Multi-timeframe alignment (30m + 15m direction agree) OR (30m + 1h agree) OR explosive 15m momentum with neutral 30m/1h.",
            "B. Valid R/R >= 1.4 with technical stop & target (levels, ATR, pivots, VWAP, Fib).",
            "Supporting Factors (each +1 score): volume expansion (v_ratio >1.3), MACD inflection, SuperTrend bias aligned, BB squeeze release, key level confluence ≥2, funding mean reversion edge, ATR context suitable.",
            "Penalties: conflicting TF direction, RSI extreme (>75 or <25) without reversal trigger, v_ratio <0.8, momentum divergence (MACD vs price), excessive chop (frequent reversals last 10 bars).",
            "Scoring: >=3 total (incl. core) -> consider LONG/SHORT, 5+ -> high confidence. <3 -> HOLD.",
            "Never force a trade. SHORT setups treated equal to LONG.",
            "",
            "Risk Logic: stop distance 1.3%-4.0%; target distance ≥1.4× stop (preferred ≥1.8×).",
            "Reject trade if logical price ordering fails (LONG: TP > price > SL; SHORT: SL > price > TP).",
            "",
            "Return strict JSON only (no commentary outside fields).",
            ""
        ]

        prompt_parts.extend(self._generate_candle_data_section(dataframes))
        prompt_parts.extend(self._generate_funding_rates_section(funding_rates))
        prompt_parts.extend(self._generate_market_sentiment_section(funding_rates))
        prompt_parts.extend(self._generate_funding_thresholds_section(funding_rates))

        # Response schema example
        prompt_parts.extend([
            "JSON Schema Example:",
            "{",
            '  "recap_heading": "Micro consolidation resolving upward from confluence zone",',
            '  "trading_insight": "30m + 15m bullish alignment with expanding volume and squeeze release; risk defined below.",',
            '  "signal": "long",  // long | short | hold',
            '  "confidence": 0.72,  // 0-1 scaled',
            '  "risk_level": "medium",',
            '  "time_horizon_hours": 3,',
            '  "key_drivers": ["15m+30m alignment", "Volume expansion", "Squeeze release"],',
            '  "score": 4,  // integer total factors (core + supporting - penalties)',
            '  "supporting_factors": {',
            '     "volume_expansion": true,',
            '     "macd_inflection": true,',
            '     "supertrend_bias": true,',
            '     "squeeze_release": false,',
            '     "level_confluence": 2',
            '  },',
            '  "penalties": { "rsi_extreme": false, "conflict_tf": false, "low_volume": false },',
            '  "trading_setup": {',
            f'     "stop_loss": {mid * 0.985:.4f},',
            f'     "take_profit": {mid * 1.015:.4f}',
            '  }',
            "}",
            "Return ONLY JSON. No prose outside fields.",
        ])
        
        return "\n".join(prompt_parts)
    
    def _calculate_24h_change(self, df: pd.DataFrame | None, current_price: float) -> float:
        """Calculate 24h price change percentage."""
        if df is None or df.empty or len(df) < 24:
            return 0.0
        
        price_24h_ago = df['c'].iloc[-24] if len(df) >= 24 else df['c'].iloc[0]
        return ((current_price - price_24h_ago) / price_24h_ago) * 100


    def _generate_candle_data_section(self, dataframes: Dict[Timeframe, pd.DataFrame]) -> List[str]:
        """Generate candle data section for the prompt."""
        prompt_parts = ["=== CANDLE DATA ==="]

        MAX_DAYS_CUTOFF = {
            Timeframe.MINUTES_15: 3,
            Timeframe.MINUTES_30: 3,
            Timeframe.HOUR_1: 7,
            Timeframe.HOURS_4: 14 
        }
        
        for timeframe in [Timeframe.MINUTES_15, Timeframe.MINUTES_30, Timeframe.HOUR_1, Timeframe.HOURS_4]:
            if timeframe not in dataframes:
                continue
                
            df = dataframes[timeframe]
            if df.empty:
                continue

            max_days = MAX_DAYS_CUTOFF.get(timeframe, 5.0)

            configured_days = self.timeframe_lookback_days.get(timeframe, 5.0)
            effective_days = min(configured_days, max_days)
            
            prompt_parts.extend(self._format_timeframe_data(df, timeframe, effective_days))
        
        return prompt_parts
    
    def _format_timeframe_data(self, df: pd.DataFrame, timeframe: Timeframe, effective_days: float) -> List[str]:
        """Format individual timeframe data with focused crypto-specific indicators."""
        candle_count = min(len(df), int((effective_days * 24 * 60) / timeframe.minutes))
        recent_candles = df.tail(candle_count)
        
        # Simplified header with core crypto indicators only
        sections = [
            f"\n{timeframe.name} Timeframe (last {len(recent_candles)} candles):",
            "Time | O | H | L | C | Vol | V_Ratio | ATR | MACD_Hist | SuperTrend | RSI | BB_Width | EMA | VWAP"
        ]

        for idx, row in recent_candles.iterrows():
            timestamp = idx.strftime("%m-%d %H:%M") if isinstance(idx, (pd.Timestamp, datetime)) else str(idx)
            sections.append(
                f"{timestamp} | {row.get('o', 0):.4f} | {row.get('h', 0):.4f} | "
                f"{row.get('l', 0):.4f} | {row.get('c', 0):.4f} | {row.get('v', 0):.0f} | "
                f"{row.get('v_ratio', 1):.2f} | {row.get('ATR', 0):.4f} | "
                f"{row.get('MACD_Hist', 0):.4f} | {row.get('SuperTrend', 0):.4f} | "
                f"{row.get('RSI', 50):.1f} | {row.get('BB_width', 0):.4f} | "
                f"{row.get('EMA', 0):.4f} | {row.get('VWAP', 0):.4f}"
            )

        # Only include core indicators summary for crypto trading
        sections.extend(self._generate_core_indicators_summary(recent_candles, timeframe))

        trend = "Bullish" if recent_candles['c'].iloc[-1] > recent_candles['SuperTrend'].iloc[-1] else "Bearish"
        atr_value = recent_candles.get('ATR', pd.Series([0])).iloc[-1]

        sections.extend([
            f"Summary for {timeframe.name}:",
            f"- Price range: ${recent_candles['l'].min():.4f} - ${recent_candles['h'].max():.4f}",
            f"- Average volume: {recent_candles['v'].mean():.0f}",
            f"- Current trend: {trend}",
            f"- Volatility (ATR): {atr_value:.4f}",
            ""
        ])

        return sections

    def _generate_core_indicators_summary(self, df: pd.DataFrame, timeframe: Timeframe) -> List[str]:
        """Generate focused crypto-specific indicators summary."""
        if df.empty:
            return [f"\n=== {timeframe.name} CORE INDICATORS ===", "No data available"]
            
        latest = df.iloc[-1]
        
        # Focus on 6 core crypto indicators that provide unique insights
        sections = [
            f"\n=== {timeframe.name} CORE INDICATORS ===",
            f"Price Action: ${latest.get('c', 0):.4f} (Range: ${df['l'].min():.4f}-${df['h'].max():.4f})",
            f"Volume: {latest.get('v', 0):.0f} (Ratio: {latest.get('v_ratio', 1):.2f}x avg)",
            f"Volatility: ATR {latest.get('ATR', 0):.4f} ({(latest.get('ATR', 0) / latest.get('c', 1) * 100):.2f}%)",
            f"Trend: SuperTrend {latest.get('SuperTrend', 0):.4f} ({'Bullish' if latest.get('c', 0) > latest.get('SuperTrend', 0) else 'Bearish'})",
            f"Momentum: RSI {latest.get('RSI', 50):.1f}, MACD Hist {latest.get('MACD_Hist', 0):.4f}",
            f"Volatility Squeeze: BB Width {latest.get('BB_width', 0):.4f} ({'Expanding' if latest.get('BB_width', 0) > df['BB_width'].iloc[-5:-1].mean() else 'Contracting'})",
            f"Key Levels: EMA {latest.get('EMA', 0):.4f}, VWAP {latest.get('VWAP', 0):.4f}",
            ""
        ]
        
        # Add recent pivot points if available
        if 'R1' in latest and 'S1' in latest:
            sections.extend([
                f"Pivot Levels: R1 {latest.get('R1', 0):.4f}, S1 {latest.get('S1', 0):.4f}",
                ""
            ])
        
        return sections
    
    def _generate_support_resistance_analysis(self, df: pd.DataFrame) -> List[str]:
        """Generate support and resistance analysis."""
        if df.empty or len(df) < 20:
            return ["\n--- Support/Resistance ---", "Insufficient data for S/R analysis"]
        
        # Calculate recent highs and lows
        recent_high = df['h'].rolling(window=10).max().iloc[-1]
        recent_low = df['l'].rolling(window=10).min().iloc[-1]
        current_price = df['c'].iloc[-1]
        
        # Calculate distance from key levels
        resistance_distance = ((recent_high - current_price) / current_price) * 100
        support_distance = ((current_price - recent_low) / current_price) * 100
        
        return [
            "\n--- Support/Resistance Analysis ---",
            f"Recent High: {recent_high:.4f} ({resistance_distance:+.2f}%)",
            f"Recent Low: {recent_low:.4f} ({support_distance:+.2f}%)",
            f"Range: {recent_high - recent_low:.4f} ({((recent_high - recent_low) / recent_low) * 100:.2f}%)",
        ]
    
    def _generate_price_action_analysis(self, df: pd.DataFrame) -> List[str]:
        """Generate price action analysis for Wyckoff methodology."""
        if df.empty or len(df) < 10:
            return ["\n--- Price Action ---", "Insufficient data for price action analysis"]
        
        # Calculate recent price movements
        last_5_change = ((df['c'].iloc[-1] - df['c'].iloc[-6]) / df['c'].iloc[-6]) * 100
        last_10_change = ((df['c'].iloc[-1] - df['c'].iloc[-11]) / df['c'].iloc[-11]) * 100
        
        # Volume-Price Relationship (key for Wyckoff)
        recent_volume_avg = df['v'].tail(5).mean()
        previous_volume_avg = df['v'].tail(10).head(5).mean()
        volume_change = ((recent_volume_avg - previous_volume_avg) / previous_volume_avg) * 100 if previous_volume_avg > 0 else 0
        
        # Price volatility
        volatility = df['c'].pct_change().tail(20).std() * 100
        
        # Determine potential Wyckoff phase
        wyckoff_phase = self._analyze_wyckoff_phase(df)
        
        return [
            "\n--- Price Action & Wyckoff Analysis ---",
            f"5-period change: {last_5_change:+.2f}%",
            f"10-period change: {last_10_change:+.2f}%",
            f"Volume change: {volume_change:+.2f}%",
            f"Volatility (20-period): {volatility:.2f}%",
            f"Potential Wyckoff Phase: {wyckoff_phase}",
        ]


    def _analyze_wyckoff_phase(self, df: pd.DataFrame) -> str:
        """Analyze potential Wyckoff market phase based on price and volume."""
        if df.empty or len(df) < 20:
            return "Insufficient data"
        
        # Get recent price and volume data
        prices = df['c'].tail(20)
        volumes = df['v'].tail(20)
        
        # Calculate price trend and volume trend
        price_trend = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
        volume_avg_recent = volumes.tail(10).mean()
        volume_avg_previous = volumes.head(10).mean()
        volume_trend = (volume_avg_recent - volume_avg_previous) / volume_avg_previous if volume_avg_previous > 0 else 0
        
        # Determine phase based on price and volume relationship
        return self._classify_wyckoff_phase(price_trend, volume_trend)
    
    def _classify_wyckoff_phase(self, price_trend: float, volume_trend: float) -> str:
        """Classify Wyckoff phase based on price and volume trends."""
        # Accumulation/Distribution phases
        if abs(price_trend) < 0.02 and volume_trend > 0.2:
            return "Accumulation/Distribution Phase" if volume_trend > 0.5 else "Consolidation with Volume"
        
        # Markup/Markdown phases
        if price_trend > 0.02 and volume_trend > 0:
            return "Markup Phase (Bullish)"
        if price_trend < -0.02 and volume_trend > 0:
            return "Markdown Phase (Bearish)"
        
        # Low volume scenarios
        if abs(price_trend) < 0.01 and volume_trend < -0.2:
            return "Low Volume Consolidation"
        
        # Divergence scenarios
        if price_trend > 0 and volume_trend < -0.2:
            return "Potential Distribution (Bearish Divergence)"
        if price_trend < 0 and volume_trend < -0.2:
            return "Potential Accumulation (Bullish Divergence)"
        
        return "Transition Phase"
    
    def _generate_funding_rates_section(self, funding_rates: List[FundingRateEntry]) -> List[str]:
        """Generate funding rates section for the prompt."""
        prompt_parts = [
            "=== FUNDING RATES ===",
            "Time | Rate | Premium | 8h_Avg | Trend"
        ]
        
        if not funding_rates:
            prompt_parts.append("No funding rate data available")
            return prompt_parts
            
        # Show last 48 hours of funding rates (48 entries since they're hourly)
        recent_funding = funding_rates[-48:] if len(funding_rates) >= 48 else funding_rates
        
        for i, rate in enumerate(recent_funding):
            # Calculate 8h moving average
            start_idx = max(0, i - 7)
            avg_8h = sum(r.funding_rate for r in recent_funding[start_idx:i+1]) / max(1, i + 1 - start_idx)
            
            # Determine trend (simplified)
            trend = "→"
            if i > 0:
                if rate.funding_rate > recent_funding[i-1].funding_rate:
                    trend = "↑"
                elif rate.funding_rate < recent_funding[i-1].funding_rate:
                    trend = "↓"
            
            # Format timestamp
            timestamp = datetime.fromtimestamp(rate.time / 1000).strftime("%m-%d %H:%M")
            
            prompt_parts.append(
                f"{timestamp} | {rate.funding_rate:.6f} | {rate.premium:.6f} | {avg_8h:.6f} | {trend}"
            )
        
        return prompt_parts
    
    def _generate_market_sentiment_section(self, funding_rates: List[FundingRateEntry]) -> List[str]:
        """Generate market sentiment section based on funding rates."""
        prompt_parts = ["", "=== MARKET SENTIMENT ==="]
        
        if not funding_rates:
            prompt_parts.extend([
                "Current funding sentiment: Neutral",
                "Funding rate magnitude: Low",
                "Current rate: 0.000000 (Neutral)",
                ""
            ])
            return prompt_parts
        
        # Add funding rate analysis
        funding_analysis = self.analyze_funding_rate_patterns(funding_rates)
        
        if funding_analysis:
            current_rate = funding_analysis.get('current_rate', 0.0)
            sentiment = funding_analysis.get('sentiment', 'neutral')
            trend = funding_analysis.get('trend', 'stable')
            extremes = funding_analysis.get('extremes', {})
            
            funding_trend = f"{sentiment.replace('_', ' ').title()} ({trend})"
            
            # Simplify magnitude calculation
            magnitude_value = extremes.get('magnitude', 0)
            if magnitude_value > 0.0005:
                magnitude = "High"
            elif magnitude_value > 0.0001:
                magnitude = "Moderate"
            else:
                magnitude = "Low"
            
            bias = "Neutral"
            if current_rate > 0.0001:
                bias = "Bullish bias"
            elif current_rate < -0.0001:
                bias = "Bearish bias"
            
            prompt_parts.extend([
                f"Current funding sentiment: {funding_trend}",
                f"Funding rate magnitude: {magnitude}",
                f"Current rate: {current_rate:.6f} ({bias})",
                ""
            ])
        
        return prompt_parts
    
    def _generate_funding_thresholds_section(self, funding_rates: List[FundingRateEntry]) -> List[str]:
        """Generate funding rate thresholds section."""
        if not funding_rates or len(funding_rates) < 24:
            return []
            
        return [
            "=== FUNDING RATE THRESHOLDS ===",
            "Rate Level | Threshold | Market Implication",
            "Extremely Bullish | >0.001000 | Strong short squeeze potential",
            "Very Bullish | >0.000500 | High premium, potential reversal",
            "Bullish | >0.000200 | Moderate bullish sentiment", 
            "Neutral High | >0.000100 | Slight bullish bias",
            "Neutral Low | <-0.000100 | Slight bearish bias",
            "Bearish | <-0.000200 | Moderate bearish sentiment",
            "Very Bearish | <-0.000500 | High discount, potential bounce",
            "Extremely Bearish | <-0.001000 | Strong long squeeze potential",
            ""
        ]


    def analyze_funding_rate_patterns(self, funding_rates: List[FundingRateEntry]) -> Dict[str, Any]:
        """Analyze funding rate patterns and provide thresholds and insights"""
        if not funding_rates:
            return {}
        
        rates = [r.funding_rate for r in funding_rates]
        
        # Calculate statistics
        current_rate = rates[-1] if rates else 0.0
        avg_24h = sum(rates[-24:]) / len(rates[-24:]) if len(rates) >= 24 else current_rate
        avg_7d = sum(rates[-168:]) / len(rates[-168:]) if len(rates) >= 168 else current_rate
        
        # Define funding rate thresholds
        thresholds = {
            'extremely_bullish': 0.001,      # 0.1% (very high positive funding)
            'very_bullish': 0.0005,          # 0.05%
            'bullish': 0.0002,               # 0.02%
            'neutral_high': 0.0001,          # 0.01%
            'neutral_low': -0.0001,          # -0.01%
            'bearish': -0.0002,              # -0.02%
            'very_bearish': -0.0005,         # -0.05%
            'extremely_bearish': -0.001      # -0.1% (very high negative funding)
        }
        
        # Determine current sentiment
        sentiment = 'neutral'
        if current_rate >= thresholds['extremely_bullish']:
            sentiment = 'extremely_bullish'
        elif current_rate >= thresholds['very_bullish']:
            sentiment = 'very_bullish'
        elif current_rate >= thresholds['bullish']:
            sentiment = 'bullish'
        elif current_rate >= thresholds['neutral_high']:
            sentiment = 'neutral_bullish'
        elif current_rate <= thresholds['extremely_bearish']:
            sentiment = 'extremely_bearish'
        elif current_rate <= thresholds['very_bearish']:
            sentiment = 'very_bearish'
        elif current_rate <= thresholds['bearish']:
            sentiment = 'bearish'
        elif current_rate <= thresholds['neutral_low']:
            sentiment = 'neutral_bearish'
        
        # Calculate trend
        trend = 'stable'
        if len(rates) >= 8:
            recent_avg = sum(rates[-8:]) / 8
            older_avg = sum(rates[-16:-8]) / 8 if len(rates) >= 16 else recent_avg
            if recent_avg > older_avg * 1.5:
                trend = 'increasing'
            elif recent_avg < older_avg * 0.5:
                trend = 'decreasing'
        
        # Calculate volatility
        if len(rates) >= 24:
            rate_changes = [abs(rates[i] - rates[i-1]) for i in range(1, min(25, len(rates)))]
            volatility = sum(rate_changes) / len(rate_changes)
        else:
            volatility = 0.0
        
        return {
            'current_rate': current_rate,
            'avg_24h': avg_24h,
            'avg_7d': avg_7d,
            'sentiment': sentiment,
            'trend': trend,
            'volatility': volatility,
            'thresholds': thresholds,
            'extremes': {
                'is_extreme': abs(current_rate) >= thresholds['very_bullish'],
                'direction': 'bullish' if current_rate > 0 else 'bearish' if current_rate < 0 else 'neutral',
                'magnitude': abs(current_rate)
            },
            'mean_reversion_signal': {
                'likely': abs(current_rate) > abs(avg_7d) * 2,
                'direction': 'down' if current_rate > avg_7d * 2 else 'up' if current_rate < avg_7d * 2 else 'none'
            }
        }
