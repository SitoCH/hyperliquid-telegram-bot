import os
from typing import Dict, List, Any
from datetime import datetime
import pandas as pd
from ..wyckoff.wyckoff_types import Timeframe
from ..funding_rates_cache import FundingRateEntry


class LLMPromptGenerator:
    """Handles generation of prompts for LLM analysis."""
    
    def __init__(self, timeframe_lookback_days: Dict[Timeframe, float]):
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
        prompt_parts = [
            f"URGENT TRADING ANALYSIS for {coin} - IMMEDIATE ACTION REQUIRED",
            f"Current price: ${mid:.4f}",
            f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "INSTRUCTION: Analyze ONLY the current market state and provide immediate trading action for the next 15-60 minutes.",
            "DO NOT provide general analysis or future predictions - focus on what can be traded RIGHT NOW.",
            "",
            "=== MARKET CONTEXT ===",
            f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"24h price change: {self._calculate_24h_change(dataframes.get(Timeframe.HOUR_1), mid):.2f}%",
            "",
        ]
        
        prompt_parts.extend(self._generate_candle_data_section(dataframes))
        prompt_parts.extend(self._generate_funding_rates_section(funding_rates))
        prompt_parts.extend(self._generate_market_sentiment_section(funding_rates))
        prompt_parts.extend(self._generate_funding_thresholds_section(funding_rates))
        
        # Streamlined analysis requirements
        prompt_parts.extend([
            "=== ANALYSIS REQUIREMENTS ===",
            "FOCUS: Provide actionable trading insights for the NEXT 15-60 MINUTES only.",
            "",
            "KEY ANALYSIS AREAS:",
            "1. CURRENT PRICE ACTION: Breakout, rejection, consolidation, or trend continuation?",
            "2. IMMEDIATE LEVELS: Next 2-3 key price levels (support/resistance, Fibonacci, pivot points)",
            "3. VOLUME CONFIRMATION: Does current volume support the price move?",
            "4. MOMENTUM STATUS: Building, fading, or neutral momentum right now?",
            "5. FUNDING PRESSURE: Current funding rate creating buy/sell pressure?",
            "6. WYCKOFF PHASE: Accumulation, Markup, Distribution, or Markdown?",
            "",
            "TRADING REQUIREMENTS:",
            "- Only recommend BUY/SELL if there's a setup actionable in the next 5-30 minutes",
            "- Prioritize scalping opportunities with 15-60 minute timeframes",
            "- Minimum 1:1.5 risk/reward ratio required",
            "- Clear invalidation levels must be provided",
            "",
            "=== RESPONSE FORMAT ===",
            "Provide your analysis in JSON format with complete trade setup details:",
            "",
            "{",
            '  "recap_heading": "Brief 1-line market state summary",',
            '  "trading_insight": "Your IMMEDIATE actionable trade - what to do RIGHT NOW in 4-5 lines. If no clear setup exists, say \'wait for [specific condition]\'. Be direct about current price action with some basic context",',
            '  "signal": "buy|sell|hold",',
            '  "confidence": 0.7,',
            '  "prediction": "bullish|bearish|sideways",',
            '  "risk_level": "low|medium|high",',
            '  "entry_price": 1234.56,',
            '  "stop_loss": 1200.00,',
            '  "target_price": 1300.00,',
            '  "time_horizon_hours": 1,',
            '  "key_drivers": ["primary_reason", "secondary_confirmation"],',
            '  "trading_setup": {',
            '    "action": "long|short|none",',
            '    "reason": "Concise rationale for the trade - REQUIRED for all signals",',
            '    "entry_strategy": "Market/limit order strategy - REQUIRED for buy/sell signals",',
            '    "risk_reward_ratio": 2.5,',
            '    "invalidation_level": "Price level where analysis becomes invalid - REQUIRED",',
            '    "session_timing": "Optimal entry timing - REQUIRED for buy/sell signals"',
            '  }',
            "}",
            "",
            "EXECUTION RULES:",
            "- IMMEDIATE ACTION: Only recommend BUY/SELL if setup is actionable within 5-30 minutes",
            "- CURRENT FOCUS: Signal based on what's happening NOW, not future possibilities",
            "- SCALPING PRIORITY: Quick trades (15-60 min timeframes) with clear exits",
            "- CONFLUENCE REQUIRED: Multiple indicators must confirm CURRENT setup",
            "- NO SPECULATION: Focus on current market reality, not 'could/might' scenarios",
            "- CONFIDENCE ≥0.7: Only for confirmed current price action setups",
            "",
            "MANDATORY REQUIREMENTS FOR BUY/SELL SIGNALS:",
            "1. ENTRY PRICE: Within 0.5% of current price",
            "2. STOP LOSS: Logical technical level (S/R, Fibonacci, pivot)",
            "3. TARGET PRICE: Clear reasoning (next S/R, measured move)",
            "4. RISK/REWARD: Minimum 1:1.5 ratio",
            "5. ACTION FIELD: Must match signal (buy=long, sell=short, hold=none)",
            "6. ENTRY STRATEGY: Specify order type and timing",
            "7. INVALIDATION LEVEL: Price where setup fails",
            "",
            "RISK ASSESSMENT (calculate score):",
            "POSITIVE FACTORS (+1 each): Multi-timeframe confluence, Volume confirmation, RSI 40-60, Normal funding rates, Clear trend, Price within BB middle 50%",
            "NEGATIVE FACTORS (-1 each): Conflicting signals (-2), RSI extremes (-2), Extreme funding (-2), High ATR >4% (-2), Price outside BB, Low volume",
            "RISK LEVELS: Score ≥3 = LOW, Score 0-2 = MEDIUM, Score <0 = HIGH",
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
        
        for timeframe in [Timeframe.MINUTES_15, Timeframe.MINUTES_30, Timeframe.HOUR_1, Timeframe.HOURS_4]:
            if timeframe not in dataframes:
                continue
                
            df = dataframes[timeframe]
            if df.empty:
                continue
                
            prompt_parts.extend(self._format_timeframe_data(df, timeframe))
        
        return prompt_parts
    
    def _format_timeframe_data(self, df: pd.DataFrame, timeframe: Timeframe) -> List[str]:
        """Format individual timeframe data."""
        # Calculate number of candles to show
        lookback_days = self.timeframe_lookback_days.get(timeframe, 3.0)
        candle_count = min(len(df), int((lookback_days * 24 * 60) / timeframe.minutes))
        recent_candles = df.tail(candle_count)
        
        sections = [
            f"\n{timeframe.name} Timeframe (last {len(recent_candles)} candles):",
            "Time | O | H | L | C | Vol | ATR | MACD | ST | BB_Up | BB_Low | Vol_Ratio"
        ]
        
        # Add candle data
        for idx, row in recent_candles.iterrows():
            timestamp = idx.strftime("%m-%d %H:%M") if hasattr(idx, 'strftime') else str(idx)
            sections.append(
                f"{timestamp} | {row.get('o', 0):.4f} | {row.get('h', 0):.4f} | "
                f"{row.get('l', 0):.4f} | {row.get('c', 0):.4f} | {row.get('v', 0):.0f} | "
                f"{row.get('ATR', 0):.4f} | {row.get('MACD', 0):.4f} | {row.get('SuperTrend', 0):.4f} | "
                f"{row.get('BB_upper', 0):.4f} | {row.get('BB_lower', 0):.4f} | {row.get('v_ratio', 1):.2f}"
            )
        
        # Add comprehensive technical indicators section
        sections.extend(self._generate_technical_indicators_section(recent_candles, timeframe))
        
        # Add timeframe summary
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

    def _generate_technical_indicators_section(self, df: pd.DataFrame, timeframe: Timeframe) -> List[str]:
        """Generate comprehensive technical indicators section for the prompt."""
        prompt_parts = [f"\n=== TECHNICAL INDICATORS - {timeframe.name} ==="]
        
        if df.empty:
            prompt_parts.append("No indicator data available")
            return prompt_parts
        
        # Get latest values for all indicators
        latest = df.iloc[-1]
        
        # Core Technical Indicators
        prompt_parts.extend([
            "\n--- Core Indicators ---",
            f"EMA: {latest.get('EMA', 0):.4f}",
            f"VWAP: {latest.get('VWAP', 0):.4f}",
            f"ATR: {latest.get('ATR', 0):.4f}",
            f"SuperTrend: {latest.get('SuperTrend', 0):.4f}",
            f"MACD: {latest.get('MACD', 0):.4f}",
            f"MACD Signal: {latest.get('MACD_Signal', 0):.4f}",
            f"MACD Histogram: {latest.get('MACD_Hist', 0):.4f}",
        ])
        
        # Oscillators
        prompt_parts.extend([
            "\n--- Oscillators ---",
            f"RSI: {latest.get('RSI', 50):.2f}",
            f"Stochastic %K: {latest.get('STOCH_K', 50):.2f}",
            f"Stochastic %D: {latest.get('STOCH_D', 50):.2f}",
            f"Williams %R: {latest.get('WILLR', -50):.2f}",
            f"CCI: {latest.get('CCI', 0):.2f}",
            f"ROC: {latest.get('ROC', 0):.2f}%",
        ])
        
        # Bollinger Bands
        prompt_parts.extend([
            "\n--- Bollinger Bands ---",
            f"BB Upper: {latest.get('BB_upper', 0):.4f}",
            f"BB Middle: {latest.get('BB_middle', 0):.4f}",
            f"BB Lower: {latest.get('BB_lower', 0):.4f}",
            f"BB Width: {latest.get('BB_width', 0):.4f}",
        ])
        
        # Fibonacci Levels
        prompt_parts.extend([
            "\n--- Fibonacci Retracements ---",
            f"Fib 78.6%: {latest.get('FIB_78', 0):.4f}",
            f"Fib 61.8%: {latest.get('FIB_61', 0):.4f}",
            f"Fib 50.0%: {latest.get('FIB_50', 0):.4f}",
            f"Fib 38.2%: {latest.get('FIB_38', 0):.4f}",
            f"Fib 23.6%: {latest.get('FIB_23', 0):.4f}",
        ])
        
        # Pivot Points
        prompt_parts.extend([
            "\n--- Pivot Points ---",
            f"Pivot Point: {latest.get('PIVOT', 0):.4f}",
            f"Resistance 2: {latest.get('R2', 0):.4f}",
            f"Resistance 1: {latest.get('R1', 0):.4f}",
            f"Support 1: {latest.get('S1', 0):.4f}",
            f"Support 2: {latest.get('S2', 0):.4f}",
        ])
        
        # Ichimoku Cloud
        prompt_parts.extend([
            "\n--- Ichimoku Cloud ---",
            f"Tenkan-sen: {latest.get('TENKAN', 0):.4f}",
            f"Kijun-sen: {latest.get('KIJUN', 0):.4f}",
            f"Senkou Span A: {latest.get('SENKOU_A', 0):.4f}",
            f"Senkou Span B: {latest.get('SENKOU_B', 0):.4f}",
            f"Chikou Span: {latest.get('CHIKOU', 0):.4f}",
        ])
        
        # Volume Analysis
        prompt_parts.extend([
            "\n--- Volume Analysis ---",
            f"Volume: {latest.get('v', 0):.0f}",
            f"Volume SMA: {latest.get('v_sma', 0):.0f}",
            f"Volume Ratio: {latest.get('v_ratio', 1):.2f}",
            f"Volume Normalized: {latest.get('v_normalized', 0):.2f}",
            f"Volume Trend: {latest.get('v_trend', 1):.2f}",
        ])
        
        # Support and Resistance Analysis
        prompt_parts.extend(self._generate_support_resistance_analysis(df))
        
        # Price Action Analysis
        prompt_parts.extend(self._generate_price_action_analysis(df))
        
        return prompt_parts
    
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
