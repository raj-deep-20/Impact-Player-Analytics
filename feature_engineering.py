"""
IPL Impact Player Analytics — Feature Engineering
Builds 25+ match-state features for the dual XGBoost models.
"""

import numpy as np
import pandas as pd
from data_generator import VENUES, PLAYER_POOL, get_phase, phase_to_int, player_type_to_int


VENUE_FACTOR_MAP = {v: d["pitch_factor"] for v, d in VENUES.items()}
VENUE_AVG_MAP    = {v: d["avg_score"]    for v, d in VENUES.items()}


def compute_win_probability(
    current_runs: int, current_wickets: int, overs_done: float,
    target: int, is_chasing: bool, venue_factor: float = 1.0
) -> float:
    """
    Simple Duckworth-Lewis-inspired win probability estimate.
    Returns P(batting team wins) ∈ [0, 1].
    """
    if not is_chasing:
        # Batting first: probability based on projected score vs historical avg
        balls_remaining = (20 - overs_done) * 6
        rpo = current_runs / max(overs_done, 0.1)
        projected = current_runs + rpo * (20 - overs_done) * venue_factor
        historical_avg = 175 * venue_factor
        p = 1 / (1 + np.exp(-(projected - historical_avg) / 20))
    else:
        # Chasing: probability based on required run rate vs current run rate
        runs_needed  = target - current_runs
        balls_left   = (20 - overs_done) * 6
        if balls_left <= 0:
            return 1.0 if current_runs >= target else 0.0
        rrr = (runs_needed / balls_left) * 6
        crr = (current_runs / max(overs_done, 0.1))
        
        # Resources remaining (simplified)
        wickets_remaining = 10 - current_wickets
        resource_factor   = wickets_remaining / 10 * (balls_left / 120)
        
        rate_diff = crr - rrr + (wickets_remaining - 5) * 0.3
        p = 1 / (1 + np.exp(-rate_diff * 0.8))
        
        # Penalise heavily for very few balls left
        if balls_left < 18:
            p *= (balls_left / 18) ** 0.5 if rrr > 14 else 1.0
    
    return float(np.clip(p, 0.02, 0.98))


def build_over_snapshot_features(deliveries_df: pd.DataFrame,
                                  matches_df:    pd.DataFrame) -> pd.DataFrame:
    """
    For each over boundary in a match, compute a state-snapshot row with 25+ features.
    This is the input to the Impact Player Classifier.
    """
    rows = []
    
    for match_id, mrow in matches_df.iterrows():
        mid        = mrow["match_id"]
        season     = mrow["season"]
        venue      = mrow["venue"]
        vf         = VENUE_FACTOR_MAP.get(venue, 1.0)
        va         = VENUE_AVG_MAP.get(venue, 175)
        has_impact = mrow["has_impact_rule"]
        
        match_dels = deliveries_df[deliveries_df["match_id"] == mid].copy()
        if match_dels.empty:
            continue
        
        for innings in [1, 2]:
            inn_dels = match_dels[match_dels["innings"] == innings]
            if inn_dels.empty:
                continue
            
            target = mrow["team1_score"] + 1 if innings == 2 else None
            is_chasing = innings == 2
            
            for over in range(1, 20):  # snapshot at end of each completed over
                up_to = inn_dels[inn_dels["over"] < over]
                if up_to.empty or up_to["over"].max() < over - 1:
                    continue
                
                cum_runs     = up_to["cumulative_runs"].iloc[-1]
                cum_wickets  = up_to["cumulative_wickets"].iloc[-1]
                overs_done   = over
                
                # ── Core rate features ──────────────────────────────────────
                crr = cum_runs / max(overs_done, 0.1)
                rrr = ((target - cum_runs) / max((20 - overs_done) * 6, 1) * 6
                       if is_chasing else crr)
                rate_diff = crr - rrr
                
                balls_remaining  = (20 - overs_done) * 6
                required_per_ball = (target - cum_runs) / max(balls_remaining, 1) if is_chasing else 0
                
                # ── Phase ────────────────────────────────────────────────────
                phase     = get_phase(over)
                phase_int = phase_to_int(phase)
                
                # ── Last 3 overs run rate ─────────────────────────────────
                last3 = inn_dels[inn_dels["over"].isin(range(max(0, over - 3), over))]
                last3_rpo = last3["total_runs"].sum() / max(len(last3) / 6, 0.1)
                last3_wkts = last3["is_wicket"].sum()
                
                # ── Powerplay metrics ────────────────────────────────────
                pp_dels = inn_dels[inn_dels["over"] < 6]
                pp_runs = pp_dels["total_runs"].sum() if not pp_dels.empty else 0
                pp_wkts = pp_dels["is_wicket"].sum()  if not pp_dels.empty else 0
                
                # ── Win probability (pre-substitution) ──────────────────
                win_prob_pre = compute_win_probability(
                    cum_runs, cum_wickets, overs_done,
                    target or va, is_chasing, vf
                )
                
                # ── Impact Player substitution label ──────────────────────
                sub_over = mrow.get("impact_sub_over_inn1") if innings == 1 else None
                optimal_sub = 0
                
                if has_impact and sub_over is not None and over == sub_over:
                    # Calculate post-sub win prob (3 overs later)
                    post_over = min(over + 3, 19)
                    post_dels = inn_dels[inn_dels["over"] < post_over + 1]
                    if not post_dels.empty:
                        post_runs = post_dels["cumulative_runs"].iloc[-1]
                        post_wkts = post_dels["cumulative_wickets"].iloc[-1]
                        win_prob_post = compute_win_probability(
                            post_runs, post_wkts, post_over,
                            target or va, is_chasing, vf
                        )
                        if win_prob_post - win_prob_pre > 0.05:
                            optimal_sub = 1
                
                # ── Impact player type feature ────────────────────────────
                player_in = mrow.get("impact_player_in")
                sub_player_type = 0
                sub_auction_val = 0.0
                sub_form        = 0.5
                if player_in and player_in in PLAYER_POOL:
                    p = PLAYER_POOL[player_in]
                    sub_player_type = player_type_to_int(p["type"])
                    sub_auction_val = p["price"]
                    sub_form        = p["form_2025"]
                
                # ── Venue features ────────────────────────────────────────
                is_chinnaswamy  = int("Chinnaswamy" in venue)
                is_wankhede     = int("Wankhede"    in venue)
                
                # ── Momentum (boundary rate last 3 overs) ────────────────
                boundary_deliveries = last3[last3["total_runs"].isin([4, 6])]
                boundary_rate = len(boundary_deliveries) / max(len(last3), 1)
                
                rows.append({
                    # IDs
                    "match_id":         mid,
                    "season":           season,
                    "innings":          innings,
                    "over":             over,
                    
                    # ── 25 Features ──────────────────────────────────────
                    "phase_int":        phase_int,          # 1
                    "over_num":         over,               # 2
                    "cum_runs":         cum_runs,           # 3
                    "cum_wickets":      cum_wickets,        # 4
                    "current_rpo":      round(crr, 3),      # 5
                    "required_rpo":     round(rrr, 3),      # 6
                    "rate_differential":round(rate_diff, 3),# 7
                    "balls_remaining":  balls_remaining,    # 8
                    "is_chasing":       int(is_chasing),    # 9
                    "pp_runs":          pp_runs,            # 10
                    "pp_wickets":       pp_wkts,            # 11
                    "last3_rpo":        round(last3_rpo, 3),# 12
                    "last3_wickets":    last3_wkts,         # 13
                    "boundary_rate":    round(boundary_rate,3),# 14
                    "venue_factor":     vf,                 # 15
                    "venue_avg_score":  va,                 # 16
                    "is_chinnaswamy":   is_chinnaswamy,     # 17
                    "is_wankhede":      is_wankhede,        # 18
                    "win_prob_pre":     round(win_prob_pre, 4),# 19
                    "sub_player_type":  sub_player_type,    # 20
                    "sub_auction_val":  sub_auction_val,    # 21
                    "sub_form":         sub_form,           # 22
                    "has_impact_rule":  int(has_impact),    # 23
                    "season_year":      season,             # 24
                    "wickets_remaining":10 - cum_wickets,   # 25
                    
                    # ── Label ──────────────────────────────────────────────
                    "optimal_sub":      optimal_sub,
                    "win_prob":         win_prob_pre,
                })
    
    return pd.DataFrame(rows)


def build_player_prediction_features(player_stats_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build match-level player features for the fantasy points regressor.
    Adds head-to-head, form trajectory, and auction ROI signals.
    """
    df = player_stats_df.copy()
    
    # Season-over-season form trajectory
    df_sorted = df.sort_values(["player", "season"])
    df["prev_season_pts"] = df_sorted.groupby("player")["fantasy_pts_avg"].shift(1)
    df["form_trajectory"] = (df["fantasy_pts_avg"] - df["prev_season_pts"]) / \
                             df["prev_season_pts"].clip(lower=1)
    df["form_trajectory"]  = df["form_trajectory"].fillna(0.0)
    
    # Auction value vs output ROI
    df["value_per_cr"]     = df["fantasy_pts_avg"] / df["auction_price_cr"].clip(lower=0.5)
    
    # Encode player type
    df["player_type_int"]  = df["player_type"].map(
        lambda t: ["opener","top_order","middle_order","finisher","allrounder",
                   "spinner","pace_powerplay","pace_death","wicketkeeper"].index(t)
        if t in ["opener","top_order","middle_order","finisher","allrounder",
                 "spinner","pace_powerplay","pace_death","wicketkeeper"] else 0
    )
    
    # Strike rate / economy normalised
    df["batting_contribution"] = (
        df["runs"] / df["matches"].clip(lower=1) * (df["strike_rate"] / 100)
    ).fillna(0)
    
    df["bowling_contribution"] = (
        df["wickets"] / df["matches"].clip(lower=1) * (12 - df["economy"].clip(upper=12))
    ).fillna(0)
    
    # Sixes rate (entertainment/fantasy premium)
    df["sixes_per_match"]  = df["sixes"] / df["matches"].clip(lower=1)
    
    # Composite performance index
    df["perf_index"] = (
        0.35 * df["form_score"] +
        0.25 * (df["fantasy_pts_avg"] / 80).clip(upper=1) +
        0.20 * df["roi_score"].clip(upper=5) / 5 +
        0.20 * (df["form_trajectory"].clip(-0.5, 0.5) + 0.5)
    )
    
    return df


FEATURE_COLS_CLASSIFIER = [
    "phase_int", "over_num", "cum_runs", "cum_wickets",
    "current_rpo", "required_rpo", "rate_differential",
    "balls_remaining", "is_chasing", "pp_runs", "pp_wickets",
    "last3_rpo", "last3_wickets", "boundary_rate",
    "venue_factor", "venue_avg_score", "is_chinnaswamy", "is_wankhede",
    "win_prob_pre", "sub_player_type", "sub_auction_val", "sub_form",
    "has_impact_rule", "season_year", "wickets_remaining",
]

FEATURE_COLS_REGRESSOR = [
    "player_type_int", "form_score", "form_trajectory",
    "batting_avg", "strike_rate", "economy",
    "batting_contribution", "bowling_contribution",
    "sixes_per_match", "auction_price_cr", "value_per_cr",
    "matches", "perf_index", "season",
]
