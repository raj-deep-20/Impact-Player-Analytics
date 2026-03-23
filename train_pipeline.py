"""
IPL Impact Player Analytics — Main Training Pipeline
Run: python train_pipeline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import json
import numpy as np
import pandas as pd
from pathlib import Path

from data_generator import (
    generate_ball_by_ball, generate_player_stats,
    PLAYER_POOL, VENUES, PLAYER_NAMES
)
from feature_engineering import (
    build_over_snapshot_features,
    build_player_prediction_features,
    compute_win_probability,
    FEATURE_COLS_CLASSIFIER, FEATURE_COLS_REGRESSOR,
)
from models import ImpactPlayerClassifier, FantasyPointsRegressor, train_all

Path("data").mkdir(exist_ok=True)
Path("models").mkdir(exist_ok=True)
Path("outputs").mkdir(exist_ok=True)


def main():
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  IPL IMPACT PLAYER ANALYTICS — Training Pipeline    ║")
    print("║  Dual-Model: Classifier + Regressor + Explainability║")
    print("╚══════════════════════════════════════════════════════╝\n")

    # ── Step 1: Data Generation ──────────────────────────────────────────────
    print("Step 1/4 │ Generating IPL ball-by-ball data (2020-2025)...")
    deliveries, matches = generate_ball_by_ball(n_matches=300)
    player_stats        = generate_player_stats(deliveries, matches)

    deliveries.to_csv("data/deliveries.csv", index=False)
    matches.to_csv("data/matches.csv",       index=False)
    player_stats.to_csv("data/player_stats.csv", index=False)

    print(f"  ✓ {len(deliveries):,} deliveries | {len(matches):,} matches | "
          f"{len(player_stats):,} player-season rows")

    # ── Step 2: Feature Engineering ─────────────────────────────────────────
    print("\nStep 2/4 │ Engineering 25+ match-state features...")
    snapshot_df = build_over_snapshot_features(deliveries, matches)
    player_df   = build_player_prediction_features(player_stats)

    snapshot_df.to_csv("data/snapshot_features.csv", index=False)
    player_df.to_csv("data/player_features.csv",     index=False)

    print(f"  ✓ Snapshot rows : {len(snapshot_df):,}")
    print(f"  ✓ Player rows   : {len(player_df):,}")
    print(f"  ✓ Positive subs : {snapshot_df['optimal_sub'].sum():,} "
          f"({snapshot_df['optimal_sub'].mean()*100:.1f}%)")

    # ── Step 3: Model Training ───────────────────────────────────────────────
    print("\nStep 3/4 │ Training Gradient Boosting models...")
    metrics, clf, reg = train_all(snapshot_df, player_df)

    # ── Step 4: Generate Inference Examples ──────────────────────────────────
    print("\nStep 4/4 │ Running inference examples & saving outputs...")

    # Example A: Live match state → substitution recommendation
    example_state = {
        "phase_int":         2,      # Death overs
        "over_num":          16,
        "cum_runs":          148,
        "cum_wickets":       5,
        "current_rpo":       9.25,
        "required_rpo":      13.5,
        "rate_differential": -4.25,
        "balls_remaining":   24,
        "is_chasing":        1,
        "pp_runs":           52,
        "pp_wickets":        1,
        "last3_rpo":         11.3,
        "last3_wickets":     2,
        "boundary_rate":     0.28,
        "venue_factor":      1.25,   # Chinnaswamy
        "venue_avg_score":   195,
        "is_chinnaswamy":    1,
        "is_wankhede":       0,
        "win_prob_pre":      0.38,
        "sub_player_type":   7,      # pace_death
        "sub_auction_val":   18.0,   # Bumrah-tier
        "sub_form":          0.94,
        "has_impact_rule":   1,
        "season_year":       2025,
        "wickets_remaining": 5,
    }

    sub_pred   = clf.predict_proba(example_state)
    sub_shap   = clf.explain(example_state, top_n=6)

    # Example B: Fantasy picks for RCB vs MI
    rcb_mi_players = player_df[
        player_df["team"].isin(["Royal Challengers Bengaluru", "Mumbai Indians"]) &
        (player_df["season"] == 2025)
    ].copy()

    top5_picks = reg.rank_players(rcb_mi_players, top_n=5) if len(rcb_mi_players) >= 5 else pd.DataFrame()

    # ── Save inference output ─────────────────────────────────────────────
    output = {
        "substitution_recommendation": {
            "match_context": "IPL 2025 | Chinnaswamy | Over 16 | Chasing 185",
            "prediction":    sub_pred,
            "shap_explanation": sub_shap,
            "win_prob_lift_if_sub": round(sub_pred["sub_probability"] * 0.12, 3),
        },
        "fantasy_top5": top5_picks[[
            "player", "team", "player_type", "auction_price_cr",
            "predicted_pts", "confidence", "fantasy_pts_avg", "roi_score"
        ]].to_dict("records") if not top5_picks.empty else [],
        "model_metrics": metrics,
    }

    with open("outputs/inference_examples.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    # ── Print Summary ─────────────────────────────────────────────────────
    print("\n" + "═"*55)
    print("  INFERENCE RESULTS")
    print("═"*55)

    print(f"\n  [Impact Player Decision @ Over 16, Chinnaswamy]")
    print(f"  Recommend Sub  : {'✅ YES' if sub_pred['recommend_sub'] else '❌ NO'}")
    print(f"  Sub Prob       : {sub_pred['sub_probability']*100:.1f}%")
    print(f"  Win Prob Lift  : +{output['substitution_recommendation']['win_prob_lift_if_sub']*100:.1f}%")

    print(f"\n  [SHAP Explanation — Top Drivers]")
    for s in sub_shap[:4]:
        bar = "█" * int(abs(s["shap_value"]) * 200)
        print(f"  {s['direction']} {s['feature']:22s} {s['shap_value']:+.4f}  {bar}")

    if not top5_picks.empty:
        print(f"\n  [Fantasy Top-5: RCB vs MI @ 2025]")
        for i, row in top5_picks.iterrows():
            conf_bar = "●" * int(row["confidence"] * 10)
            print(f"  {row['player']:25s} {row['predicted_pts']:5.1f}pts "
                  f"[conf: {row['confidence']:.2f}] {conf_bar}")

    print(f"\n  [Model Performance]")
    print(f"  Classifier ROC-AUC : {metrics['classifier']['cv_roc_auc_mean']:.4f} "
          f"± {metrics['classifier']['cv_roc_auc_std']:.4f}")
    print(f"  Regressor  MAE     : {metrics['regressor']['cv_mae_mean']:.1f} pts")

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║  ✓ Pipeline complete. All outputs saved.            ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    return metrics, clf, reg, output


if __name__ == "__main__":
    main()
