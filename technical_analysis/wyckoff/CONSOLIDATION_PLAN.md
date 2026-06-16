# GRI-648: Wyckoff Consolidation Plan

## Status
**Finalized Strategy** - Ready for Implementation

## Audit Findings
- **Directory Split:** `wyckoff/` (single-TF) and `wyckoff/mtf/` (multi-TF) maintain parallel logic for interpreting market signals.
- **Type Redundancy:** `wyckoff_multi_timeframe_types.py` extends `wyckoff_types.py` but introduces new enums (`MultiTimeframeDirection`) and data classes (`TimeframeGroupAnalysis`) that could be unified.
- **Scattered Interpretation Logic:** The logic for whether a phase, sign, or action is "bullish" or "bearish" is present in `wyckoff_types.py` (helpers), `wyckoff.py` (phase detection), and `mtf/wyckoff_multi_timeframe.py` (group analysis).
- **Asymmetric Analysis:** `mtf` logic re-interprets `WyckoffState` using its own set of weights and heuristics that are sometimes redundant with the core detection logic.

---

## Unified Wyckoff Engine Strategy (Option 1)

### Phase 1: Type Unification
**Goal:** Single source of truth for all Wyckoff data structures.

- **Actions:**
    1.  Merge all types from `mtf/wyckoff_multi_timeframe_types.py` into `wyckoff_types.py`.
    2.  Move `MultiTimeframeDirection`, `TimeframeGroupAnalysis`, `MultiTimeframeContext`, and `AllTimeframesAnalysis` to `wyckoff_types.py`.
    3.  Relocate MTF-specific constants (momentum and volume thresholds) to `wyckoff_types.py`.
- **Target File:** `technical_analysis/wyckoff/wyckoff_types.py`

### Phase 2: Signal Interpretation Centralization
**Goal:** Decouple interpretation logic from both detection and aggregation.

- **Actions:**
    1.  Create `technical_analysis/wyckoff/signals.py`.
    2.  Migrate directional bias logic, weight adjustments, and momentum checks from:
        *   `mtf/wyckoff_multi_timeframe.py`
        *   `mtf/mtf_direction.py`
        *   `mtf/mtf_alignment.py`
    3.  Functions to consolidate:
        *   `is_phase_confirming_momentum`
        *   `get_volume_weight_adjustment`
        *   `get_base_volume_strength`
        *   `determine_overall_direction`
        *   `calculate_overall_alignment`
        *   `calculate_overall_confidence`
        *   `_calculate_momentum_intensity`
- **Target File:** `technical_analysis/wyckoff/signals.py`

### Phase 3: Core Logic Consolidation
**Goal:** Unified entry point for multi-timeframe analysis.

- **Actions:**
    1.  Move `mtf/wyckoff_multi_timeframe.py` to `technical_analysis/wyckoff/mtf_engine.py`.
    2.  Refactor to use unified types and centralized signals.
    3.  Update `WyckoffAnalyzer.analyze()` in `wyckoff_analyzer.py` to use the new `mtf_engine.py`.
- **Target File:** `technical_analysis/wyckoff/mtf_engine.py`

### Phase 4: Formatting & Suggestions
**Goal:** Unified output generation.

- **Actions:**
    1.  Create `technical_analysis/wyckoff/formatter.py`.
    2.  Consolidate `wyckoff_description.py` and `mtf/wyckoff_multi_timeframe_description.py`.
    3.  Standardize Wyckoff Sign explanations and Phase context logic.
    4.  Move `mtf/trade_suggestion.py` to `technical_analysis/wyckoff/trade_suggestion.py`.
- **Target Files:** `technical_analysis/wyckoff/formatter.py`, `technical_analysis/wyckoff/trade_suggestion.py`

### Phase 5: Cleanup & Verification
**Goal:** Remove legacy paths and ensure stability.

- **Actions:**
    1.  Update all remaining project imports.
    2.  Delete the `mtf/` subdirectory.
    3.  Run all Wyckoff-related tests and verify Telegram output.
- **Verification:** `pytest tests/technical_analysis/test_wyckoff.py`

---

## Implementation Plan

1.  **Harmonize Types:** Update `wyckoff_types.py`.
2.  **Extract Signals:** Create `signals.py` and migrate logic.
3.  **Refactor Engine:** Create `mtf_engine.py` and update `WyckoffAnalyzer`.
4.  **Refactor Formatter:** Create `formatter.py` and move `trade_suggestion.py`.
5.  **Final Cleanup:** Remove `mtf/` directory and update imports.
