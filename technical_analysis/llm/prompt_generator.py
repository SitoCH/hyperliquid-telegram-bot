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
            f"URGENT TRADING ANALYSIS for {coin} - IMMEDIATE ACTION REQUIRED",
            f"Current price: ${mid:.4f}",
            f"Analysis timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            "",
            "INSTRUCTION: Analyze current market state and provide actionable trading recommendations for 3-4 hour position holding.",
            "Focus on immediate opportunities based on current technical setup and market conditions.",
            "",
            "=== MARKET CONTEXT ===",
            f"Analysis timestamp: {current_time.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"24h price change: {self._calculate_24h_change(dataframes.get(Timeframe.HOUR_1), mid):.2f}%",
            "",
            "🚨 CRITICAL DATA FRESHNESS NOTICE:",
            "- 15m/30m candles: Last candle may be up to 14-29 minutes old",
            "- 1H candles: Last candle may be up to 59 minutes old", 
            "- 4H candles: Last candle may be up to 3 hours and 59 minutes old",
            "- Market conditions may have changed since the last complete candle",
            "- Weight more recent timeframes (15m/30m) for immediate entry decisions",
            "- Use 4H data for context and major trend direction only",
            "- Consider that price action after the last candle close is not visible in this data",
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
            "CRITICAL MANDATE: Only recommend LONG and SHORT trades with 70%+ win probability. Default to HOLD only when no clear directional bias exists.  Do not favor LONG over SHORT signals - treat bearish setups with equal weight and urgency.",
            "",
            "🚨 TIMEFRAME PRIORITY MANDATE:",
            "- FOCUS: 1H (40%) and 30m (35%) timeframes are your PRIMARY decision drivers",
            "- SUPPORT: 15m (15%) for entry timing confirmation",
            "- CONTEXT ONLY: 4H (10%) for broad trend awareness - DO NOT base trading decisions on 4H data",
            "- IGNORE 4H conflicts: If 1H+30m align strongly, 4H opposition should NOT prevent signals",
            "", 
            "=== ADAPTIVE TIMEFRAME ANALYSIS ===",
            "⚠️ DATA FRESHNESS WEIGHTING (adjust based on staleness):",
            "- 4H: 10% weight - Background trend context only (DATA MAY BE UP TO 4 HOURS OLD)",
            "- 1H: 40% weight - Primary trend direction (DATA MAY BE UP TO 1 HOUR OLD)",
            "- 30m: 35% weight - Recent trend shifts and momentum (DATA MAY BE UP TO 30 MIN OLD)",
            "- 15m: 15% weight - Most current price action (DATA MAY BE UP TO 15 MIN OLD)",
            "",
            "FLEXIBLE ALIGNMENT RULES:",
            "1. PRIMARY SIGNALS: 1H and 30m aligned in same direction = Strong signal",
            "2. MOMENTUM OVERRIDE: If 30m+15m show strong momentum with volume, can override 1H if neutral",
            "3. FAST MARKET MODE: In high volatility (ATR >3%), 30m+15m get 60% combined weight",
            "4. TREND CONTINUATION: Strong 1H trend + 30m/15m confirmation = Valid signal",
            "5. REVERSAL SETUPS: Strong divergence on 30m+15m can signal against trend if 1H shows weakness",
            "6. 4H CONTEXT ONLY: Use 4H solely for broad market context, NOT for entry/exit decisions",
            "",
            "TIER 1 SIGNALS (Required for LONG/SHORT recommendation):",
            "1. MULTI-TIMEFRAME ALIGNMENT: 1H and 30m must align OR strong momentum on 30m+15m with 1H neutral",
            "2. VOLUME CONFIRMATION: Recent volume >150% of 20-period average during setup formation",
            "3. KEY LEVEL CONFLUENCE: Price at major S/R (Pivot, Fibonacci, psychological levels) +/- 0.5%",
            "4. MOMENTUM CONFIRMATION: RSI 45-65 range with positive divergence OR breaking from oversold/overbought",
            "5. RISK/REWARD: Minimum 1:2 ratio with clear technical stop loss and target levels",
            "NOTE: 4H timeframe is for context only - do NOT require 4H alignment for signals",
            "",
            "TIER 2 CONFIRMATIONS (Need 2+ for signal validation - SIMPLIFIED):",
            "- MACD histogram momentum shift (positive for LONG, negative for SHORT)",
            "- Bollinger Band squeeze release with volume (BB_width expanding + V_Ratio >1.5)",
            "- SuperTrend alignment with price direction (LONG: price > SuperTrend, SHORT: price < SuperTrend)",
            "- Key level confluence at pivot points or psychological levels (within 0.5% tolerance)",
            "- EMA/VWAP reclaim or breakdown with sustained follow-through and volume",
            "- Funding rate mean reversion setup (extreme readings showing reversal tendency)",
            "",
            "AUTOMATIC DISQUALIFIERS (Force HOLD signal - STREAMLINED):",
            "- RSI extreme without reversal: RSI >75 or <25 without clear reversal patterns",
            "- Conflicting timeframe signals: 1H vs 30m disagree on direction",
            "- Low volume confirmation: V_Ratio <0.8 during setup formation", 
            "- Bollinger squeeze without direction: BB_width contracting without clear bias",
            "- MACD and RSI divergence: Indicators showing opposing momentum signals",
            "- Choppy price action: Multiple false breakouts in recent 6-hour period",
            "",           
            "=== ENHANCED INDICATOR ANALYSIS FRAMEWORK ===",
            "",
            "CORE CRYPTO INDICATOR ANALYSIS (Focus on 6 essential indicators):",
            "1. RSI (Primary Momentum):",
            "   - LONG bias: RSI 45-65 with positive divergence or breaking above 45 from oversold",
            "   - SHORT bias: RSI 35-55 with negative divergence or breaking below 55 from overbought",
            "   - Extreme caution: RSI >75 or <25 without clear reversal patterns",
            "",
            "2. Volume Confirmation (Critical for crypto):",
            "   - V_Ratio >1.5: Above average volume confirms breakout/breakdown",
            "   - V_Ratio <0.8: Low volume, weakens signal strength",
            "   - Volume must support price direction for valid signals",
            "",
            "3. ATR Volatility Context:",
            "   - High ATR (>3% of price): Requires wider stops, shorter time horizon",
            "   - Low ATR (<1.5% of price): Allows tighter stops, longer time horizon",
            "   - ATR determines position sizing and risk parameters",
            "",
            "4. MACD Histogram (Momentum Confirmation):",
            "   - LONG signal: MACD histogram turning positive with volume",
            "   - SHORT signal: MACD histogram turning negative with volume",
            "   - Divergence: Histogram direction opposing price movement (reversal setup)",
            "",
            "5. SuperTrend (Primary Trend Filter):",
            "   - LONG trend: Price above SuperTrend line (bullish bias)",
            "   - SHORT trend: Price below SuperTrend line (bearish bias)",
            "   - Flip signals: SuperTrend color change with volume confirmation",
            "",
            "6. Bollinger Band Squeeze (Breakout Anticipation):",
            "   - Squeeze setup: BB_width contracting = potential breakout imminent",
            "   - LONG breakout: BB_width expanding with price above BB_middle + volume",
            "   - SHORT breakdown: BB_width expanding with price below BB_middle + volume",
            "",
            "SIMPLIFIED SIGNAL SCORING SYSTEM:",
            "REQUIRED MINIMUM: 12 points for LONG/SHORT signal (reduced from 18)",
            "",
            "TIER 1 FACTORS (4 points each):",
            "+ Multi-timeframe alignment (1H and 30m agree)",
            "+ Volume confirmation (V_Ratio >1.5)",
            "",
            "TIER 2 FACTORS (2 points each - CORE CRYPTO INDICATORS):",
            "+ MACD histogram momentum alignment",
            "+ RSI in favorable range (45-65) with direction support",
            "+ SuperTrend alignment with price direction",
            "+ Bollinger Band context (squeeze release or directional bias)",
            "+ Key level confluence (EMA/VWAP/Pivot levels within 0.5%)",
            "+ Funding rate mean reversion setup",
            "+ ATR volatility supports trade setup and risk management",
            "",
            "PENALTY FACTORS:",
            "- Conflicting timeframe signals (-4 points)",
            "- Extreme RSI without reversal (-3 points)",
            "- Low volume during setup (-2 points)",
            "- Poor risk/reward <1:1.5 (-2 points)",
            "",
            "SIGNAL CONFIDENCE REQUIREMENTS:",
            "- confidence ≥0.8: Setups with 12+ points and both Tier 1 factors",
            "- confidence 0.7-0.79: Setups with 10-11 points and at least 1 Tier 1 factor",
            "- confidence <0.7: Automatic HOLD signal",
            "",
            "MARKET STATE FILTERS:",
            "HIGH VOLATILITY PERIODS (ATR >4% of price):",
            "- Require 12+ points for signal (higher bar)",
            "- Wider stops (minimum 2.5% from entry)",
            "- Shorter time horizon (2-3 hours max)",
            "",
            "LOW VOLATILITY PERIODS (ATR <1.5% of price):",
            "- Accept 8+ points for signal (lower bar in stable conditions)",
            "- Tighter stops (1.5-2% from entry)",
            "- Extended time horizon (4-6 hours)",
            "",            
            "=== RESPONSE FORMAT ===",
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
            '  "recap_heading": "Single sentence market state summary focusing on immediate trading opportunity or reason to wait, don''t start with the hold / long / short signal, just a plain sentence",',
            '  "trading_insight": "If HOLD: Explain what specific condition to wait for. If LONG/SHORT: Explain the high-probability setup with clear entry reasoning in 2-3 sentences.",',
            '  "signal": "long|short|hold",',
            '  "confidence": 0.8,',
            '  "risk_level": "low|medium|high",',
            '  "time_horizon_hours": 4,',
            '  "key_drivers": ["Clear bullish momentum with volume confirmation", "Multiple timeframe alignment at key support level"], // mininum 1, maximum 3 key drivers',
            '  "signal_score": 12,',
            '  "tier1_checklist": {',
            '    "timeframe_alignment": true,',
            '    "volume_confirmation": true,',
            '    "key_level_confluence": true,',
            '    "momentum_confirmation": true,',
            '    "risk_reward_ratio": 2.1',
            '  },',
            '  "trading_setup": {',
            f'    "stop_loss": 1200.00,  // REQUIRED for LONG/SHORT: Must be < {mid:.4f} for LONG, > {mid:.4f} for SHORT',
            f'    "take_profit": 1300.00,  // REQUIRED for LONG/SHORT: Must be > {mid:.4f} for LONG, < {mid:.4f} for SHORT',
            '  }',
            "}",            "",
            "EXECUTION MANDATE:",
            "- BE EXTREMELY SELECTIVE: Only 20-30% of analyses should result in LONG/SHORT signals",
            "- PREFER WAITING: Better to miss opportunities than take low-probability trades",
            "- EQUAL TREATMENT: Give SHORT setups equal weight and consideration as LONG setups",
            "- TIER 1 REQUIREMENTS: Need 4 out of 5 Tier 1 factors",
            "- CONFLUENCE OVER COMPLEXITY: Simple setups with multiple confirmations beat complex analysis",
            "- TREND AGNOSTIC: Trade both WITH and AGAINST trends when technical setup warrants",
            "",
            f"🚨 CRITICAL TRADING_SETUP REQUIREMENTS FOR LONG/SHORT SIGNALS:",
            f"CURRENT REFERENCE PRICE: ${mid:.4f} - ALL LEVELS MUST BE RELATIVE TO THIS PRICE",
            "",
            "MANDATORY LOGIC VALIDATION:",
            f"- For LONG signals: take_profit MUST be > ${mid:.4f} (current price)",
            f"- For LONG signals: stop_loss MUST be < ${mid:.4f} (current price)",
            f"- For SHORT signals: take_profit MUST be < ${mid:.4f} (current price)",
            f"- For SHORT signals: stop_loss MUST be > ${mid:.4f} (current price)",
            "- If these logical rules are violated, FORCE signal to HOLD",
            "",
            "- stop_loss: MANDATORY exact price level (not percentage):",
            "  * Support/Resistance levels from pivot points, previous highs/lows",
            "  * Key Fibonacci levels (38.2%, 50%, 61.8%)",
            "  * EMA/VWAP levels, Bollinger Band levels",
            "  * Previous swing highs/lows, psychological round numbers",
            f"  * ATR-based stops: LONG = ${mid:.4f} - (2 * ATR), SHORT = ${mid:.4f} + (2 * ATR)",
            f"  * Distance validation: minimum 1.5% from ${mid:.4f}, maximum 4% from ${mid:.4f}",
            "  * Choose the nearest valid technical level within these constraints",
            "",
            "- take_profit: MANDATORY exact price level:",
            "  * PRIMARY TARGETS (use these first):",
            "    - Key Fibonacci extensions: 127.2%, 161.8%",
            "    - Previous swing highs (for LONG) / Previous swing lows (for SHORT)",
            "    - Psychological levels: round numbers ending in 00, 50",
            "    - Key resistance/support levels from pivot analysis",
            "  * SECONDARY TARGETS:",
            "    - Bollinger Band outer bands (upper for LONG, lower for SHORT)",
            "    - Major EMA levels acting as dynamic resistance/support",
            "    - VWAP extensions and standard deviation bands",
            "  * MEASURED MOVE TARGETS:",
            "    - Pattern-based targets: flag poles, triangles, channel projections",
            f"    - ATR-based targets: LONG = ${mid:.4f} + (2-3 * ATR), SHORT = ${mid:.4f} - (2-3 * ATR)",
            "  * RISK/REWARD VALIDATION:",
            "    - Target must be minimum 1.5x the stop loss distance (1:1.5 R/R minimum)",
            "    - Preferred 2:1 or better risk/reward ratio",
            f"    - Distance constraints: minimum 1.5% from ${mid:.4f}, maximum 6% from ${mid:.4f}",
            "  * CONFLUENCE PRIORITY: Choose targets where 2+ methods align within 0.5% price range",
            "",
            "PRICE LOGIC DOUBLE-CHECK BEFORE FINALIZING:",
            f"1. LONG signal: Verify take_profit > ${mid:.4f} > stop_loss",
            f"2. SHORT signal: Verify stop_loss > ${mid:.4f} > take_profit",
            "3. If logic fails, immediately change signal to HOLD",
            "- WITHOUT these exact prices AND proper logic, DO NOT generate LONG/SHORT signals - default to HOLD",
            "",
            f"🚨 FINAL MANDATE: If your signal is LONG or SHORT, you MUST provide exact stop_loss and take_profit prices that follow proper logic:",
            f"- LONG: take_profit (${mid:.4f}+) > current_price (${mid:.4f}) > stop_loss (${mid:.4f}-)",
            f"- SHORT: stop_loss (${mid:.4f}+) > current_price (${mid:.4f}) > take_profit (${mid:.4f}-)",
            "If you cannot determine precise technical levels for both stop_loss and take_profit that follow this logic, then your signal MUST be HOLD.",
            "Do not generate LONG/SHORT signals without trading_setup containing logically correct exact prices - this is a system requirement for trade execution.",
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
