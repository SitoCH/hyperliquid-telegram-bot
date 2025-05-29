import pandas as pd
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime
from logging_utils import logger
from ..wyckoff.wyckoff_types import Timeframe


class AnalysisFilter:
    """Filter logic to determine when expensive AI analysis should be triggered."""

    
    def should_run_llm_analysis(self, dataframes: Dict[Timeframe, pd.DataFrame], coin: str, interactive: bool) -> Tuple[bool, str]:
        """Use a cheap LLM model to determine if expensive analysis is warranted."""
        if interactive:
            return True, "Interactive analysis requested"
        
        # Create a lightweight data summary for the cheap model
        market_summary = self._create_market_summary(dataframes)
        
        # Create prompt for cheap model filtering
        filter_prompt = self._create_filter_prompt(coin, market_summary)
        
        # Use cheap model to determine if detailed analysis is needed
        try:
            from .openrouter_client import OpenRouterClient
            
            # Initialize client with cheap model for filtering
            filter_client = OpenRouterClient()
            filter_client.model = "google/gemini-flash-8b"  # Cheap model for filtering
            
            response, cost = filter_client.call_api(filter_prompt)
            
            # Parse response to determine if analysis should proceed
            should_analyze, reason = self._parse_filter_response(response, coin)
            
            if should_analyze:
                logger.info(f"LLM filter triggered analysis for {coin}: {reason} (cost: ${cost:.4f})")
            else:
                logger.debug(f"LLM filter skipped analysis for {coin}: {reason} (cost: ${cost:.4f})")
            
            return should_analyze, reason
            
        except Exception as e:
            logger.error(f"LLM filter failed for {coin}: {str(e)}")
            # Fallback to basic analysis if LLM filtering fails
            return self._basic_activity_check(dataframes, coin)
    
    def _create_market_summary(self, dataframes: Dict[Timeframe, pd.DataFrame]) -> Dict[str, Any]:
        """Create a lightweight market summary for cheap LLM filtering."""
        summary: Dict[str, Dict[str, Any]] = {
            "timeframes": {}
        }
        
        for tf, df in dataframes.items():
            if df.empty or len(df) < 5:
                continue
                
            # Get basic price action data
            current_price = df['c'].iloc[-1]
            open_price = df['o'].iloc[-5] if len(df) >= 5 else df['o'].iloc[0]
            high_5 = df['h'].iloc[-5:].max()
            low_5 = df['l'].iloc[-5:].min()
            
            # Calculate basic metrics
            price_change_5 = (current_price - open_price) / open_price
            volatility = (high_5 - low_5) / current_price
            
            # Volume data if available
            volume_ratio = 1.0
            if 'v_ratio' in df.columns and not df['v_ratio'].empty:
                volume_ratio = df['v_ratio'].iloc[-1]
            
            # Basic indicator data
            indicators = {}
            if 'RSI' in df.columns and not df['RSI'].empty:
                indicators['rsi'] = df['RSI'].iloc[-1]
            if 'MACD' in df.columns and not df['MACD'].empty:
                indicators['macd'] = df['MACD'].iloc[-1]
            if 'SuperTrend' in df.columns and not df['SuperTrend'].empty:
                indicators['supertrend_bullish'] = current_price > df['SuperTrend'].iloc[-1]
            
            summary["timeframes"][str(tf)] = {
                "price_change_5p": round(price_change_5 * 100, 2),
                "volatility": round(volatility * 100, 2),
                "volume_ratio": round(volume_ratio, 2),
                "indicators": indicators
            }
        
        return summary
    
    def _create_filter_prompt(self, coin: str, market_summary: Dict[str, Any]) -> str:
        """Create prompt for cheap LLM model to determine if expensive analysis is needed."""
        return f"""You are a sophisticated trading analysis filter for {coin}. Your role is to evaluate market conditions and determine if they warrant expensive detailed technical analysis.

Current Market Data:
{json.dumps(market_summary, indent=2)}

Assessment Framework:
Analyze the market data holistically across all timeframes. Consider the interplay between:

• Price Action: Movement patterns, momentum, volatility characteristics
• Volume Profile: Activity levels, spikes, confirmations or divergences  
• Technical Indicators: Signal strength, extremes, convergence/divergence
• Multi-Timeframe Context: Alignment, conflicts, cascade effects
• Market Structure: Support/resistance levels, breakouts, trend changes

Market Conditions Assessment:
Consider various scenarios that could warrant detailed analysis:
- Significant directional moves with volume confirmation
- Technical breakouts from consolidation patterns
- Extreme indicator readings suggesting reversals or continuations
- Multi-timeframe confluence suggesting major moves
- Unusual volume activity indicating institutional interest
- Volatility expansion after compression periods
- Support/resistance level tests with momentum

Decision Criteria:
Use your analytical judgment to assess whether current conditions present:
1. Clear trading opportunities with favorable risk/reward
2. Market inflection points requiring detailed analysis
3. Sufficient signal strength to justify analysis costs
4. Potential for actionable insights

Your assessment should balance opportunity identification with cost efficiency. Consider both immediate trading signals and developing market structure changes.

Provide your analysis in JSON format:
{{
  "should_analyze": true/false,
  "reason": "detailed explanation of your assessment",
  "confidence": 0.0-1.0,
  "priority_level": "high/medium/low",
  "key_factors": ["list", "of", "key", "deciding", "factors"]
}}"""
    
    def _parse_filter_response(self, response: str, coin: str) -> Tuple[bool, str]:
        """Parse the cheap LLM response to determine if analysis should proceed."""
        try:
            import json
            data = json.loads(response)
            should_analyze = data.get("should_analyze", False)
            reason = data.get("reason", "LLM filter decision")
            confidence = data.get("confidence", 0.5)
            
            # Add confidence to reason for logging
            detailed_reason = f"{reason} (confidence: {confidence:.1%})"
            
            return should_analyze, detailed_reason
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM filter response for {coin}: {str(e)}")
            # Conservative fallback - if we can't parse, run analysis
            return True, "LLM filter parsing failed, proceeding with analysis"
    
    def _basic_activity_check(self, dataframes: Dict[Timeframe, pd.DataFrame], coin: str) -> Tuple[bool, str]:
        """Basic fallback activity check when LLM filtering fails."""
        for tf, df in dataframes.items():
            if df.empty or len(df) < 5:
                continue
                
            # Simple price movement check
            price_change = abs((df['c'].iloc[-1] - df['c'].iloc[-5]) / df['c'].iloc[-5])
            if price_change > 0.025:  # 2.5% movement
                return True, f"Fallback: {price_change:.1%} price movement in {tf}"
            
            # Simple volume check
            if 'v_ratio' in df.columns and not df['v_ratio'].empty:
                volume_ratio = df['v_ratio'].iloc[-1]
                if volume_ratio > 2.0:  # 2x volume spike
                    return True, f"Fallback: {volume_ratio:.1f}x volume spike in {tf}"
        
        return False, "Fallback: No significant activity detected"