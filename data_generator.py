"""
IPL Impact Player Analytics — Synthetic Data Generator
Simulates IPL 2020-2025 ball-by-ball data with Impact Player substitutions (2023+)
Mirrors structure of Kaggle 'IPL Ball-By-Ball Dataset' (deliveries.csv + matches.csv)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

# ── Constants ────────────────────────────────────────────────────────────────

TEAMS = [
    "Royal Challengers Bengaluru", "Mumbai Indians", "Chennai Super Kings",
    "Kolkata Knight Riders", "Rajasthan Royals", "Delhi Capitals",
    "Sunrisers Hyderabad", "Punjab Kings", "Lucknow Super Giants",
    "Gujarat Titans"
]

VENUES = {
    "M Chinnaswamy Stadium": {"pitch_factor": 1.25, "avg_score": 195, "city": "Bengaluru"},
    "Wankhede Stadium":      {"pitch_factor": 1.18, "avg_score": 185, "city": "Mumbai"},
    "MA Chidambaram Stadium":{"pitch_factor": 0.88, "avg_score": 165, "city": "Chennai"},
    "Eden Gardens":          {"pitch_factor": 1.05, "avg_score": 175, "city": "Kolkata"},
    "Sawai Mansingh Stadium":{"pitch_factor": 1.10, "avg_score": 178, "city": "Jaipur"},
    "Arun Jaitley Stadium":  {"pitch_factor": 1.08, "avg_score": 176, "city": "Delhi"},
    "Rajiv Gandhi Intl":     {"pitch_factor": 1.12, "avg_score": 180, "city": "Hyderabad"},
    "BRSABV Ekana Stadium":  {"pitch_factor": 0.95, "avg_score": 168, "city": "Lucknow"},
    "Narendra Modi Stadium": {"pitch_factor": 1.02, "avg_score": 172, "city": "Ahmedabad"},
}

PLAYER_TYPES = ["opener", "top_order", "middle_order", "finisher", "allrounder",
                "spinner", "pace_powerplay", "pace_death", "wicketkeeper"]

# 2025 Auction prices (₹cr) — realistic figures
PLAYER_POOL = {
    # Openers / Batters
    "Virat Kohli":        {"type": "top_order",    "team": "Royal Challengers Bengaluru", "price": 21.0, "form_2025": 0.88},
    "Rohit Sharma":       {"type": "opener",        "team": "Mumbai Indians",              "price": 16.3, "form_2025": 0.74},
    "Shubman Gill":       {"type": "opener",        "team": "Gujarat Titans",              "price": 14.9, "form_2025": 0.82},
    "KL Rahul":           {"type": "wicketkeeper",  "team": "Lucknow Super Giants",        "price": 18.0, "form_2025": 0.79},
    "Ruturaj Gaikwad":    {"type": "opener",        "team": "Chennai Super Kings",         "price": 18.0, "form_2025": 0.81},
    "Yashasvi Jaiswal":   {"type": "opener",        "team": "Rajasthan Royals",            "price": 18.0, "form_2025": 0.90},
    "Travis Head":        {"type": "opener",        "team": "Sunrisers Hyderabad",         "price": 6.8,  "form_2025": 0.85},
    "Faf du Plessis":     {"type": "opener",        "team": "Royal Challengers Bengaluru", "price": 7.0,  "form_2025": 0.72},
    "Quinton de Kock":    {"type": "wicketkeeper",  "team": "Kolkata Knight Riders",       "price": 3.6,  "form_2025": 0.78},
    # Middle order / Finishers
    "MS Dhoni":           {"type": "wicketkeeper",  "team": "Chennai Super Kings",         "price": 4.0,  "form_2025": 0.71},
    "Hardik Pandya":      {"type": "allrounder",    "team": "Mumbai Indians",              "price": 15.0, "form_2025": 0.76},
    "Rishabh Pant":       {"type": "wicketkeeper",  "team": "Delhi Capitals",              "price": 27.0, "form_2025": 0.83},
    "Heinrich Klaasen":   {"type": "finisher",      "team": "Sunrisers Hyderabad",         "price": 5.25, "form_2025": 0.88},
    "Rinku Singh":        {"type": "finisher",      "team": "Kolkata Knight Riders",       "price": 13.0, "form_2025": 0.86},
    "Glenn Maxwell":      {"type": "allrounder",    "team": "Royal Challengers Bengaluru", "price": 11.0, "form_2025": 0.77},
    "Suryakumar Yadav":   {"type": "finisher",      "team": "Mumbai Indians",              "price": 16.35,"form_2025": 0.92},
    "Tilak Varma":        {"type": "middle_order",  "team": "Mumbai Indians",              "price": 8.5,  "form_2025": 0.84},
    "Rajat Patidar":      {"type": "middle_order",  "team": "Royal Challengers Bengaluru", "price": 11.0, "form_2025": 0.80},
    "Abhishek Sharma":    {"type": "opener",        "team": "Sunrisers Hyderabad",         "price": 14.0, "form_2025": 0.87},
    "Nitish Kumar Reddy": {"type": "allrounder",    "team": "Sunrisers Hyderabad",         "price": 6.0,  "form_2025": 0.82},
    # Bowlers
    "Jasprit Bumrah":     {"type": "pace_death",    "team": "Mumbai Indians",              "price": 18.0, "form_2025": 0.94},
    "Mohammed Shami":     {"type": "pace_powerplay", "team": "Gujarat Titans",             "price": 10.0, "form_2025": 0.88},
    "Rashid Khan":        {"type": "spinner",       "team": "Gujarat Titans",              "price": 18.0, "form_2025": 0.91},
    "Yuzvendra Chahal":   {"type": "spinner",       "team": "Rajasthan Royals",            "price": 18.0, "form_2025": 0.86},
    "Arshdeep Singh":     {"type": "pace_death",    "team": "Punjab Kings",                "price": 18.0, "form_2025": 0.85},
    "T Natarajan":        {"type": "pace_death",    "team": "Sunrisers Hyderabad",         "price": 10.75,"form_2025": 0.83},
    "Varun Chakaravarthy":{"type": "spinner",       "team": "Kolkata Knight Riders",       "price": 12.0, "form_2025": 0.84},
    "Axar Patel":         {"type": "allrounder",    "team": "Delhi Capitals",              "price": 16.5, "form_2025": 0.81},
    "Ravindra Jadeja":    {"type": "allrounder",    "team": "Chennai Super Kings",         "price": 18.0, "form_2025": 0.80},
    "Pat Cummins":        {"type": "allrounder",    "team": "Kolkata Knight Riders",       "price": 20.5, "form_2025": 0.86},
    "Harshal Patel":      {"type": "pace_death",    "team": "Royal Challengers Bengaluru", "price": 11.75,"form_2025": 0.78},
    "Mayank Yadav":       {"type": "pace_powerplay", "team": "Lucknow Super Giants",       "price": 11.0, "form_2025": 0.83},
    "Yash Dayal":         {"type": "pace_death",    "team": "Royal Challengers Bengaluru", "price": 5.0,  "form_2025": 0.76},
    "Avesh Khan":         {"type": "pace_death",    "team": "Lucknow Super Giants",        "price": 10.0, "form_2025": 0.74},
}

PLAYER_NAMES = list(PLAYER_POOL.keys())

def get_phase(over: int) -> str:
    if over < 6:   return "powerplay"
    if over < 15:  return "middle"
    return "death"

def phase_to_int(phase: str) -> int:
    return {"powerplay": 0, "middle": 1, "death": 2}[phase]

def player_type_to_int(ptype: str) -> int:
    return PLAYER_TYPES.index(ptype) if ptype in PLAYER_TYPES else 0

def simulate_ball(over, ball, batting_team_strength, bowling_team_strength,
                  wickets_fallen, phase, venue_factor):
    """Simulate a single delivery outcome."""
    base_scoring_rate = 8.5 * venue_factor
    
    # Phase multipliers
    phase_mult = {"powerplay": 1.1, "middle": 0.9, "death": 1.2}[phase]
    
    # Pressure from wickets
    wicket_pressure = max(0.6, 1.0 - wickets_fallen * 0.05)
    
    # Scoring probability
    score_prob = min(0.95, base_scoring_rate / 30 * phase_mult * batting_team_strength * wicket_pressure)
    
    runs = 0
    is_wicket = False
    is_wide = False
    is_no_ball = False
    
    r = random.random()
    
    # Extra (wide / no-ball)
    extra_prob = 0.04
    if r < extra_prob:
        is_wide = random.random() > 0.5
        is_no_ball = not is_wide
        runs = 1
        return runs, is_wicket, is_wide, is_no_ball
    
    # Wicket probability
    wicket_base = 0.055 / bowling_team_strength
    if phase == "powerplay": wicket_base *= 1.1
    if phase == "death":     wicket_base *= 0.85
    
    if random.random() < wicket_base and wickets_fallen < 9:
        is_wicket = True
        return 0, True, False, False
    
    # Runs scored
    run_probs = {0: 0.30, 1: 0.28, 2: 0.10, 3: 0.02, 4: 0.18, 6: 0.12}
    if phase == "death":
        run_probs = {0: 0.25, 1: 0.22, 2: 0.08, 3: 0.01, 4: 0.22, 6: 0.22}
    
    keys   = list(run_probs.keys())
    values = list(run_probs.values())
    total  = sum(values)
    values = [v / total for v in values]
    
    runs = np.random.choice(keys, p=values)
    
    # Venue boost for high-scoring grounds
    if venue_factor > 1.15 and random.random() < 0.1:
        runs = min(runs + 2, 6)
    
    return int(runs), False, False, False


def generate_ball_by_ball(n_matches: int = 300) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic ball-by-ball IPL data (2020–2025)."""
    
    deliveries_rows = []
    matches_rows    = []
    
    seasons = [2020, 2021, 2022, 2023, 2024, 2025]
    match_id = 1
    
    for season in seasons:
        n_season = n_matches // len(seasons)
        teams_this_season = TEAMS[:8] if season <= 2021 else TEAMS
        
        for _ in range(n_season):
            # Pick two teams
            team1, team2 = random.sample(teams_this_season, 2)
            venue_name   = random.choice(list(VENUES.keys()))
            venue        = VENUES[venue_name]
            
            match_date = datetime(season, 3, 25) + timedelta(days=random.randint(0, 45))
            toss_winner = random.choice([team1, team2])
            toss_decision = random.choice(["bat", "field"])
            batting_first = toss_winner if toss_decision == "bat" else (team2 if toss_winner == team1 else team1)
            bowling_first = team2 if batting_first == team1 else team1
            
            t1_strength = random.uniform(0.85, 1.15)
            t2_strength = random.uniform(0.85, 1.15)
            
            innings_scores = []
            
            # Impact Player flag (2023+)
            has_impact_rule = season >= 2023
            impact_subs = {}  # {innings: (over, player_in, player_out)}
            
            for innings in range(1, 3):
                batting_team  = batting_first  if innings == 1 else bowling_first
                bowling_team  = bowling_first  if innings == 1 else batting_first
                bat_strength  = t1_strength    if batting_team == team1 else t2_strength
                bowl_strength = t2_strength    if bowling_team == team2 else t1_strength
                
                total_runs    = 0
                total_wickets = 0
                balls_faced   = 0
                
                # Impact Player substitution window (overs 7–15 typically)
                impact_over   = random.randint(7, 15) if has_impact_rule else None
                impact_done   = False
                
                for over in range(20):
                    if total_wickets >= 10:
                        break
                    
                    phase       = get_phase(over)
                    venue_factor = venue["pitch_factor"]
                    
                    # Impact substitution happens at start of over
                    if has_impact_rule and impact_over == over and not impact_done:
                        player_in  = random.choice(PLAYER_NAMES)
                        player_out = random.choice(PLAYER_NAMES)
                        if player_in != player_out:
                            impact_subs[innings] = (over, player_in, player_out)
                            # Boost effectiveness post-substitution
                            if PLAYER_POOL[player_in]["type"] in ["pace_death", "spinner", "finisher"]:
                                bowl_strength *= 1.08
                        impact_done = True
                    
                    legal_deliveries = 0
                    while legal_deliveries < 6:
                        runs, is_wicket, is_wide, is_no_ball = simulate_ball(
                            over, legal_deliveries, bat_strength, bowl_strength,
                            total_wickets, phase, venue_factor
                        )
                        
                        total_runs += runs
                        if is_wicket:
                            total_wickets += 1
                        
                        if not is_wide and not is_no_ball:
                            legal_deliveries += 1
                        balls_faced += 1
                        
                        # Target in 2nd innings
                        target_check = innings_scores[0] + 1 if innings == 2 and innings_scores else None
                        
                        row = {
                            "match_id":          match_id,
                            "season":            season,
                            "innings":           innings,
                            "over":              over,
                            "ball":              legal_deliveries,
                            "batting_team":      batting_team,
                            "bowling_team":      bowling_team,
                            "venue":             venue_name,
                            "city":              venue["city"],
                            "total_runs":        runs,
                            "wide_runs":         1 if is_wide else 0,
                            "noball_runs":       1 if is_no_ball else 0,
                            "is_wicket":         1 if is_wicket else 0,
                            "cumulative_runs":   total_runs,
                            "cumulative_wickets":total_wickets,
                            "phase":             phase,
                            "has_impact_rule":   int(has_impact_rule),
                        }
                        deliveries_rows.append(row)
                        
                        if total_wickets >= 10:
                            break
                        
                        if target_check and total_runs >= target_check:
                            break
                    
                    if innings == 2 and innings_scores and total_runs >= innings_scores[0] + 1:
                        break
                
                innings_scores.append(total_runs)
            
            winner = batting_first if innings_scores[0] > innings_scores[1] else bowling_first
            if innings_scores[0] == innings_scores[1]:
                winner = "Tie"
            
            sub_over_inn1 = impact_subs.get(1, (None,))[0]
            sub_over_inn2 = impact_subs.get(2, (None,))[0]
            player_in_inn1 = impact_subs.get(1, (None, None))[1] if 1 in impact_subs else None
            
            matches_rows.append({
                "match_id":          match_id,
                "season":            season,
                "date":              match_date.strftime("%Y-%m-%d"),
                "venue":             venue_name,
                "city":              venue["city"],
                "team1":             team1,
                "team2":             team2,
                "toss_winner":       toss_winner,
                "toss_decision":     toss_decision,
                "batting_first":     batting_first,
                "team1_score":       innings_scores[0],
                "team2_score":       innings_scores[1] if len(innings_scores) > 1 else 0,
                "winner":            winner,
                "has_impact_rule":   int(has_impact_rule),
                "impact_sub_over_inn1": sub_over_inn1,
                "impact_player_in":  player_in_inn1,
                "venue_pitch_factor":venue["pitch_factor"],
            })
            
            match_id += 1
    
    deliveries_df = pd.DataFrame(deliveries_rows)
    matches_df    = pd.DataFrame(matches_rows)
    
    return deliveries_df, matches_df


def generate_player_stats(deliveries_df: pd.DataFrame, matches_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate player-level stats per season from ball-by-ball data + overlay 2025 auction data."""
    
    rows = []
    for name, info in PLAYER_POOL.items():
        for season in [2023, 2024, 2025]:
            # Simulate seasonal stats based on player type and form
            form = info["form_2025"] * random.uniform(0.88, 1.05)
            
            if info["type"] in ["opener", "top_order", "middle_order", "finisher", "wicketkeeper"]:
                matches_played = random.randint(10, 16)
                runs           = int(form * random.uniform(280, 620))
                avg            = runs / max(matches_played - random.randint(0, 3), 1)
                sr             = random.uniform(130, 185) * form
                fours          = int(runs * random.uniform(0.06, 0.10))
                sixes          = int(runs * random.uniform(0.04, 0.09))
                wickets        = 0
                economy        = 0.0
            elif info["type"] == "allrounder":
                matches_played = random.randint(12, 16)
                runs           = int(form * random.uniform(150, 380))
                avg            = runs / max(matches_played - 2, 1)
                sr             = random.uniform(120, 160) * form
                fours          = int(runs * 0.07)
                sixes          = int(runs * 0.06)
                wickets        = int(form * random.uniform(8, 20))
                economy        = random.uniform(7.5, 9.5) / form
            else:  # Bowlers
                matches_played = random.randint(10, 16)
                runs           = int(form * random.uniform(10, 80))
                avg            = runs / max(1, matches_played - 10)
                sr             = random.uniform(100, 140) * form
                fours          = 0; sixes = 0
                wickets        = int(form * random.uniform(12, 28))
                economy        = random.uniform(6.8, 9.2) / form
            
            # Dream11 Fantasy points proxy
            fantasy_pts = (
                runs * 1.0 +
                fours * 1.0 +
                sixes * 2.0 +
                (1 if runs >= 30 else 0) * 4 +
                (1 if runs >= 50 else 0) * 8 +
                wickets * 25 +
                (10 if wickets >= 3 else 0) +
                (15 if wickets >= 5 else 0)
            ) / max(matches_played, 1)
            
            rows.append({
                "player":          name,
                "season":          season,
                "team":            info["team"],
                "player_type":     info["type"],
                "auction_price_cr":info["price"],
                "form_score":      round(info["form_2025"], 3),
                "matches":         matches_played,
                "runs":            runs,
                "batting_avg":     round(avg, 1),
                "strike_rate":     round(sr, 1),
                "fours":           fours,
                "sixes":           sixes,
                "wickets":         wickets,
                "economy":         round(economy, 2) if economy > 0 else 0.0,
                "fantasy_pts_avg": round(fantasy_pts, 1),
                "roi_score":       round(fantasy_pts / max(info["price"], 0.5), 2),
            })
    
    return pd.DataFrame(rows)


if __name__ == "__main__":
    print("Generating IPL ball-by-ball data...")
    deliveries, matches = generate_ball_by_ball(n_matches=300)
    player_stats = generate_player_stats(deliveries, matches)
    
    deliveries.to_csv("data/deliveries.csv", index=False)
    matches.to_csv("data/matches.csv", index=False)
    player_stats.to_csv("data/player_stats.csv", index=False)
    
    print(f"✓ deliveries.csv  → {len(deliveries):,} rows")
    print(f"✓ matches.csv     → {len(matches):,} rows")
    print(f"✓ player_stats.csv→ {len(player_stats):,} rows")
