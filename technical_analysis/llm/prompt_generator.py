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
            "INSTRUCTION: Analyze current market state and provide actionable trading recommendations for 3-4 hour position holding.",
            "Focus on immediate opportunities based on current technical setup and market conditions.",
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

        # High probability trading requirements with strict filters
        prompt_parts.extend([
            "=== HIGH PROBABILITY TRADING ANALYSIS ===",
            "",
            "CRITICAL MANDATE: Only recommend trades with 70%+ win probability. Default to HOLD unless exceptional setup exists.",
            "",            "=== ADAPTIVE TIMEFRAME ANALYSIS ===",
            "MOMENTUM-RESPONSIVE WEIGHTING:",
            "- 4H: 25% weight - Overall trend context and major levels",
            "- 1H: 35% weight - Primary trend direction and momentum",
            "- 30m: 25% weight - Immediate trend shifts and entry setups",
            "- 15m: 15% weight - Precise entry timing and micro-momentum",
            "",
            "FLEXIBLE ALIGNMENT RULES:",
            "1. STRONG SIGNALS: 1H and 30m aligned in same direction = Valid signal",
            "2. MOMENTUM OVERRIDE: If 30m+15m show strong momentum with volume, can override 4H if 1H neutral",
            "3. FAST MARKET MODE: In high volatility (ATR >3%), lower timeframes get 60% combined weight",
            "4. TREND CONTINUATION: Existing 1H trend + 30m/15m confirmation = Signal even if 4H neutral",
            "5. REVERSAL SETUPS: Strong divergence on 30m+15m can signal against 4H trend if 1H confirms",
            "",
            "TIER 1 SIGNALS (Required for LONG/SHORT recommendation):",
            "1. MULTI-TIMEFRAME ALIGNMENT: 1H and 30m must align OR strong momentum on 30m+15m with 1H neutral",
            "2. VOLUME CONFIRMATION: Recent volume >150% of 20-period average during setup formation",
            "3. KEY LEVEL CONFLUENCE: Price at major S/R (Pivot, Fibonacci, psychological levels) +/- 0.5%",
            "4. MOMENTUM CONFIRMATION: RSI 45-65 range with positive divergence OR breaking from oversold/overbought",
            "5. RISK/REWARD: Minimum 1:2 ratio with clear technical stop loss and target levels",
            "",
            "TIER 2 CONFIRMATIONS (Need 2+ for signal validation):",
            "- MACD histogram turning positive (bullish) or negative (bearish)",
            "- Bollinger Band squeeze release with directional momentum (BB_width expanding)",
            "- SuperTrend flip with volume confirmation",
            "- Stochastic oscillator crossing in favorable territory (%K crossing %D)",
            "- Williams %R reversal from extreme levels (-80 to -20 range)",
            "- CCI breaking above +100 (bullish) or below -100 (bearish)",
            "- ROC momentum acceleration (positive for bullish, negative for bearish)",
            "- Fibonacci level confluence (price within 0.5% of key levels)",
            "- Pivot point bounce/rejection with volume confirmation",
            "- Ichimoku cloud breakout (price above/below Senkou Span A/B)",
            "- Volume ratio >1.5 during setup formation (above average volume)",
            "- Funding rate showing mean reversion setup (extreme readings reversing)",
            "- EMA/VWAP reclaim with sustained follow-through",
            "",
            "AUTOMATIC DISQUALIFIERS (Force HOLD signal):",
            "- RSI >80 or <20 (extreme overbought/oversold without clear divergence)",
            "- Stochastic >90 or <10 in extreme territory without reversal setup",
            "- Williams %R >-10 or <-90 in extreme zones without momentum confirmation",
            "- CCI >+200 or <-200 extreme readings without volume confirmation",
            "- Conflicting signals across timeframes (3+ timeframes disagree)",
            "- Low volume (<80% of average) during setup formation",
            "- BB_width contracting (squeeze) without clear breakout direction",
            "- MACD and RSI showing opposing signals",
            "- Recent major news or event within 2 hours",
            "- Funding rate in neutral zone (-0.0001 to +0.0001) with no clear bias",
            "- Price action choppy (multiple false breakouts in last 6 hours)",
            "- Ichimoku cloud providing resistance in bullish setup or support in bearish setup",
            "",           
            "=== ENHANCED INDICATOR ANALYSIS FRAMEWORK ===",
            "",
            "OSCILLATOR ANALYSIS (All indicators must be evaluated):",
            "1. RSI (45-65 optimal range):",
            "   - Bullish: RSI 50-65 with positive divergence or breaking above 45 from oversold",
            "   - Bearish: RSI 35-50 with negative divergence or breaking below 55 from overbought",
            "   - Extreme caution: RSI >80 or <20 without clear reversal patterns",
            "",
            "2. Stochastic (%K and %D lines):",
            "   - Bullish signal: %K crossing above %D in 20-80 range with upward momentum",
            "   - Bearish signal: %K crossing below %D in 20-80 range with downward momentum",
            "   - Avoid: Stochastic >90 or <10 without confirmed reversal setup",
            "",
            "3. Williams %R:",
            "   - Bullish reversal: Williams %R rising from -80 to -20 range with volume",
            "   - Bearish reversal: Williams %R falling from -20 to -80 range with volume",
            "   - Extreme zones: >-10 (overbought) or <-90 (oversold) require caution",
            "",
            "4. CCI (Commodity Channel Index):",
            "   - Bullish breakout: CCI breaking above +100 with volume confirmation",
            "   - Bearish breakdown: CCI breaking below -100 with volume confirmation",
            "   - Extreme readings: CCI >+200 or <-200 often signal reversal without momentum",
            "",
            "5. ROC (Rate of Change):",
            "   - Bullish momentum: ROC positive and accelerating (increasing positive values)",
            "   - Bearish momentum: ROC negative and accelerating (decreasing negative values)",
            "   - Divergence signals: ROC direction opposing price movement",
            "",
            "TREND AND MOMENTUM INDICATORS:",
            "1. MACD System:",
            "   - Primary signal: MACD line crossing above/below Signal line",
            "   - Momentum confirmation: MACD Histogram turning positive (bullish) or negative (bearish)",
            "   - Strength indicator: Distance between MACD and Signal line",
            "",
            "2. SuperTrend:",
            "   - Trend direction: Price above SuperTrend = bullish, below = bearish",
            "   - Signal strength: Distance between price and SuperTrend line",
            "   - Flip signals: SuperTrend color change with volume confirmation",
            "",
            "3. Bollinger Bands:",
            "   - Squeeze setup: BB_width contracting (values decreasing) = potential breakout",
            "   - Breakout signal: BB_width expanding with directional price movement",
            "   - Reversal zones: Price touching BB_upper (resistance) or BB_lower (support)",
            "",
            "FIBONACCI AND PIVOT ANALYSIS:",
            "1. Fibonacci Retracements (FIB_78, FIB_61, FIB_38, FIB_23):",
            "   - Support levels: Price bouncing from Fib levels in uptrend",
            "   - Resistance levels: Price rejected at Fib levels in downtrend",
            "   - Confluence: Multiple Fib levels within 0.5% create strong zones",
            "",
            "2. Pivot Points (R2, R1, PIVOT, S1, S2):",
            "   - Breakout signals: Price breaking above R1/R2 or below S1/S2 with volume",
            "   - Reversal zones: Price bouncing from Pivot, S1, S2 (support) or R1, R2 (resistance)",
            "   - Target levels: Use next pivot level as profit target",
            "",
            "ICHIMOKU CLOUD SYSTEM:",
            "1. Cloud Analysis (SENKOU_A, SENKOU_B):",
            "   - Bullish setup: Price above cloud (above both Senkou Span A and B)",
            "   - Bearish setup: Price below cloud (below both Senkou Span A and B)",
            "   - Neutral zone: Price inside cloud = avoid trading",
            "",
            "2. Ichimoku Lines:",
            "   - TENKAN (Conversion): Fast signal line for short-term momentum",
            "   - KIJUN (Base): Medium-term trend direction",
            "   - CHIKOU (Lagging): Confirmation of trend strength",
            "",
            "VOLUME CONFIRMATION REQUIREMENTS:",
            "1. Volume Ratio Analysis:",
            "   - V_Ratio >1.5: Above average volume confirms breakout/breakdown",
            "   - V_Ratio 0.8-1.2: Normal volume, requires additional confirmation",
            "   - V_Ratio <0.8: Low volume, weakens signal strength",
            "",
            "2. Volume Trend:",
            "   - V_Trend >1.0: Increasing volume supports price direction",
            "   - V_Trend <1.0: Decreasing volume suggests weakening momentum",
            "",
            "ENHANCED SIGNAL SCORING SYSTEM:",
            "REQUIRED MINIMUM: 18 points for LONG/SHORT signal",
            "",
            "TIER 1 FACTORS (3 points each - need ALL 5):",
            "+ Multi-timeframe alignment (15 points total)",
            "",
            "TIER 2 FACTORS (2 points each):",
            "+ Clear momentum shift with volume confirmation",
            "+ Key level bounce/rejection with confirmation (Support/Resistance, Fibonacci, Pivot Points)",
            "+ Oscillator convergence (MACD histogram + RSI alignment)",
            "+ Bollinger Band squeeze release with directional momentum (BB_width expanding)", 
            "+ SuperTrend flip with volume confirmation",
            "+ Stochastic oscillator bullish/bearish crossover (%K crossing %D in favorable territory)",
            "+ Williams %R reversal from extreme levels (-80 to -20 bullish, -20 to -80 bearish)",
            "+ CCI momentum breakout (above +100 bullish, below -100 bearish)",
            "+ ROC momentum acceleration (positive for bullish, negative for bearish)",
            "+ Ichimoku cloud breakout with price above/below Senkou Span A/B",
            "+ Volume ratio confirmation (>1.5 during setup formation)",
            "+ Funding rate mean reversion setup",
            "+ EMA/VWAP reclaim with sustained follow-through",
            "",
            "PENALTY FACTORS:",
            "- Conflicting timeframe signals (-5 points)",
            "- Extreme RSI without divergence (-3 points)",
            "- Low confidence setup (-2 points)",
            "- Poor risk/reward <1:1.5 (-3 points)",
            "",
            "SIGNAL CONFIDENCE REQUIREMENTS:",
            "- confidence ≥0.8: Only for setups with 10+ points and perfect Tier 1 alignment",
            "- confidence 0.7-0.79: Setups with 8-9 points and strong confluence",
            "- confidence <0.7: Automatic HOLD signal",
            "",
            "MARKET STATE FILTERS:",
            "HIGH VOLATILITY PERIODS (ATR >4% of price):",
            "- Require 12+ points for signal (higher bar)",
            "- Wider stops (minimum 2.5% from entry)",
            "- Shorter time horizon (2-3 hours max)",
            "",
            "LOW VOLATILITY PERIODS (ATR <1.5% of price):",
            "- Accept 8+ points for signal",
            "- Tighter stops (1.5-2% from entry)",
            "- Extended time horizon (4-6 hours)",
            "",            "=== RESPONSE FORMAT ===",
            "Provide analysis in JSON format. CRITICAL: Default to HOLD unless exceptional setup exists.",
            "",
            "MANDATORY TRADING_SETUP REQUIREMENTS:",
            "- For LONG/SHORT signals: trading_setup MUST contain precise stop_loss and take_profit prices",
            "- stop_loss: Exact price level based on technical analysis (support/resistance/ATR/pivot)",
            "- take_profit: Exact price level at next key resistance/support or measured move target",
            "- NEVER use percentage-based stops - always use absolute price levels",
            "- For HOLD signals: trading_setup can contain conditional levels for future reference",
            "",
            "{",
            '  "recap_heading": "Brief market state - focus on why trading or waiting",',
            '  "trading_insight": "If HOLD: Explain what specific condition to wait for. If LONG/SHORT: Explain the high-probability setup with clear entry reasoning in 2-3 sentences.",',
            '  "signal": "long|short|hold",',
            '  "confidence": 0.8,',
            '  "prediction": "bullish|bearish|sideways",',
            '  "risk_level": "low|medium|high",',
            '  "time_horizon_hours": 4,',
            '  "key_drivers": ["primary_high_probability_reason", "secondary_confirmation"],',
            '  "signal_score": 12,',
            '  "tier1_checklist": {',
            '    "timeframe_alignment": true,',
            '    "volume_confirmation": true,',
            '    "key_level_confluence": true,',
            '    "momentum_confirmation": true,',
            '    "risk_reward_ratio": 2.1',
            '  },',
            '  "trading_setup": {',
            '    "stop_loss": 1200.00,  // REQUIRED: Exact price for stop loss based on technical level',
            '    "take_profit": 1300.00,  // REQUIRED: Exact price for profit target at key level',
            '  }',
            "}",            "",
            "EXECUTION MANDATE:",
            "- BE EXTREMELY SELECTIVE: Only 20-30% of analyses should result in LONG/SHORT signals",
            "- PREFER WAITING: Better to miss opportunities than take low-probability trades",
            "- TIER 1 REQUIREMENTS: All 5 must be met for actionable signal",
            "- CONFLUENCE OVER COMPLEXITY: Simple setups with multiple confirmations beat complex analysis",
            "- MARKET BIAS: In trending markets, only trade WITH the trend unless extreme reversal setup",
            "",
            "CRITICAL TRADING_SETUP REQUIREMENTS FOR LONG/SHORT SIGNALS:",
            "- stop_loss: MANDATORY exact price level (not percentage):",
            "  * Support/Resistance levels from pivot points, previous highs/lows",
            "  * Fibonacci retracement levels (23.6%, 38.2%, 50%, 61.8%)",
            "  * EMA/VWAP levels, Bollinger Band levels",
            "  * ATR-based stops: current_price ± (2 * ATR)",
            "  * Minimum 1.5% away from current price, maximum 4% away, starting by the nearest technical level or resistance/support level",            "- take_profit: MANDATORY exact price level:",
            "  * PRIMARY TARGETS (use these first):",
            "    - Fibonacci extension levels: 127.2%, 161.8%, 200%, 261.8%",
            "    - Previous swing highs (for LONG) / Previous swing lows (for SHORT)",
            "    - Psychological levels: round numbers ending in 00, 50 (e.g., 1200.00, 1250.00)",
            "    - Key resistance levels (LONG) / Key support levels (SHORT) from pivot analysis",
            "  * SECONDARY TARGETS:",
            "    - Bollinger Band outer bands (upper for LONG, lower for SHORT)",
            "    - Major EMA levels (50, 100, 200) acting as dynamic resistance/support",
            "    - VWAP extensions and standard deviation bands",
            "    - Ichimoku cloud boundaries (Senkou Span A/B levels)",
            "  * MEASURED MOVE TARGETS:",
            "    - Pattern-based targets: flag poles, triangles, channel projections",
            "    - ATR-based targets: current_price + (2-3 * ATR) for LONG, current_price - (2-3 * ATR) for SHORT",
            "  * RISK/REWARD VALIDATION:",
            "    - Target must be minimum 1.5x the stop loss distance (1:1.5 R/R minimum)",
            "    - Preferred 2:1 or better risk/reward ratio",
            "    - Distance constraints: minimum 1.5% from entry, maximum 6% from entry",
            "  * CONFLUENCE PRIORITY: Choose targets where 2+ methods align within 0.5% price range",
            "- WITHOUT these exact prices, DO NOT generate LONG/SHORT signals - default to HOLD",
            "",
            "🚨 CRITICAL REQUIREMENT: If your signal is LONG or SHORT, you MUST provide exact stop_loss and take_profit prices.",
            "If you cannot determine precise technical levels for both stop_loss and take_profit, then your signal MUST be HOLD.",
            "Do not generate LONG/SHORT signals without trading_setup containing exact prices - this is a system requirement for trade execution.",
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
        """Format individual timeframe data."""
        # Calculate number of candles to show using effective days (already limited by cutoff)
        candle_count = min(len(df), int((effective_days * 24 * 60) / timeframe.minutes))
        recent_candles = df.tail(candle_count)
        sections = [
            f"\n{timeframe.name} Timeframe (last {len(recent_candles)} candles):",
            "Time | O | H | L | C | Vol | ATR | MACD | MACD_Sig | MACD_Hist | ST | RSI | BB_Up | BB_Mid | BB_Low | BB_Width | EMA | VWAP"
        ]
        
        # Add candle data with all indicators
        for idx, row in recent_candles.iterrows():
            timestamp = idx.strftime("%m-%d %H:%M") if hasattr(idx, 'strftime') else str(idx)
            sections.append(
                f"{timestamp} | {row.get('o', 0):.4f} | {row.get('h', 0):.4f} | "
                f"{row.get('l', 0):.4f} | {row.get('c', 0):.4f} | {row.get('v', 0):.0f} | "
                f"{row.get('ATR', 0):.4f} | {row.get('MACD', 0):.4f} | {row.get('MACD_Signal', 0):.4f} | "
                f"{row.get('MACD_Hist', 0):.4f} | {row.get('SuperTrend', 0):.4f} | "
                f"{row.get('RSI', 50):.1f} | {row.get('BB_upper', 0):.4f} | {row.get('BB_middle', 0):.4f} | "
                f"{row.get('BB_lower', 0):.4f} | {row.get('BB_width', 0):.4f} | "
                f"{row.get('EMA', 0):.4f} | {row.get('VWAP', 0):.4f}"
            )
        
        # Add additional indicators table for last 15 candles
        if len(recent_candles) >= 15:
            sections.extend(self._generate_additional_indicators_table(recent_candles.tail(15), timeframe))
        
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

    def _generate_additional_indicators_table(self, df: pd.DataFrame, timeframe: Timeframe) -> List[str]:
        """Generate additional indicators table for recent candles."""
        sections = [
            f"\nAdditional Indicators (last {len(df)} {timeframe.name} candles):",
            "Time | STOCH_K | STOCH_D | WILLR | CCI | ROC | V_Ratio | V_Trend | FIB_78 | FIB_61 | FIB_38 | FIB_23 | R2 | R1 | S1 | S2"
        ]
        
        for idx, row in df.iterrows():
            timestamp = idx.strftime("%m-%d %H:%M") if hasattr(idx, 'strftime') else str(idx)
            sections.append(
                f"{timestamp} | {row.get('STOCH_K', 50):.1f} | {row.get('STOCH_D', 50):.1f} | "
                f"{row.get('WILLR', -50):.1f} | {row.get('CCI', 0):.1f} | {row.get('ROC', 0):.2f} | "
                f"{row.get('v_ratio', 1):.2f} | {row.get('v_trend', 1):.2f} | "
                f"{row.get('FIB_78', 0):.4f} | {row.get('FIB_61', 0):.4f} | {row.get('FIB_38', 0):.4f} | {row.get('FIB_23', 0):.4f} | "
                f"{row.get('R2', 0):.4f} | {row.get('R1', 0):.4f} | {row.get('S1', 0):.4f} | {row.get('S2', 0):.4f}"
            )
        
        # Add Ichimoku Cloud table for last 10 candles
        if len(df) >= 10:
            sections.extend([
                f"\nIchimoku Cloud (last 10 {timeframe.name} candles):",
                "Time | TENKAN | KIJUN | SENKOU_A | SENKOU_B | CHIKOU"
            ])
            
            for idx, row in df.tail(10).iterrows():
                timestamp = idx.strftime("%m-%d %H:%M") if hasattr(idx, 'strftime') else str(idx)
                sections.append(
                    f"{timestamp} | {row.get('TENKAN', 0):.4f} | {row.get('KIJUN', 0):.4f} | "
                    f"{row.get('SENKOU_A', 0):.4f} | {row.get('SENKOU_B', 0):.4f} | {row.get('CHIKOU', 0):.4f}"
                )
        
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
