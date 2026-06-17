#!/usr/bin/env python3
"""
ABPA – Autonomous Betting Pick Agent (Cloud‑Ready)
Deploy with Railway cron: 0 8 * * *
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Tuple

import requests
import yaml
from dotenv import load_dotenv
from google import generativeai as genai
from tavily import TavilyClient

# ------------------------- 0. Setup -------------------------
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"abpa_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("ABPA")

DEFAULT_CONFIG = {
    "sports": ["soccer_epl", "basketball_nba", "tennis_atp"],
    "odds_regions": "eu,us",
    "odds_market": "h2h",
    "max_events_for_llm": 25,
    "min_future_hours": 1,
    "llm_model": "gemini-2.0-flash",
    "tavily_max_results": 3,
    "sure_prob": 0.80,
    "sure_edge": 0.02,
    "odds_prob": 0.70,
    "odds_edge": 0.03,
    "acca_prob_low": 0.35,
    "acca_prob_high": 0.55,
    "acca_edge": 0.04,
    "fallback_acca_low": 0.30,
    "fallback_acca_high": 0.60,
    "fallback_acca_edge": 0.02,
    "max_sure": 2,
    "max_odds": 5,
    "max_acca": 20,
    "results_file": "data/results.json",  # persists inside Railway volume
}

# ------------------------- 1. Dependencies -------------------------
def ensure_dependencies():
    try:
        import google.generativeai, tavily, yaml
    except ImportError:
        logger.info("Installing missing packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                               "requests", "google-generativeai", "tavily-python",
                               "python-dotenv", "pyyaml"])
        logger.info("Dependencies installed.")

def load_config() -> dict:
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path) as f:
            custom = yaml.safe_load(f)
        return {**DEFAULT_CONFIG, **custom}
    return DEFAULT_CONFIG

def setup_api_keys():
    """Check env vars – no interactive prompts in cloud."""
    required = ["ODDS_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY"]
    missing = [k for k in required if not os.getenv(k)]
    if missing:
        logger.error(f"Missing API keys: {', '.join(missing)}")
        sys.exit(1)
    logger.info("API keys verified.")

# ------------------------- 2. Odds Collector -------------------------
class OddsCollector:
    def __init__(self, config):
        self.api_key = os.getenv("ODDS_API_KEY")
        self.base = "https://api.odds-api.io/v4"
        self.sport_keys = config["sports"]
        self.regions = config["odds_regions"]
        self.market = config["odds_market"]
        self.min_future = config["min_future_hours"]
        self.retries = 3

    def _get(self, endpoint, params):
        params["apiKey"] = self.api_key
        for attempt in range(1, self.retries+1):
            try:
                r = requests.get(f"{self.base}/{endpoint}", params=params, timeout=20)
                r.raise_for_status()
                return r.json()
            except Exception as e:
                logger.warning(f"Attempt {attempt}: {e}")
                time.sleep(2**attempt)
        return {}

    def fetch_odds(self, sport_key):
        raw = self._get(f"sports/{sport_key}/odds/",
                        {"regions": self.regions, "markets": self.market, "oddsFormat": "decimal"})
        events = raw if isinstance(raw, list) else raw.get("data", [])
        now = datetime.now(timezone.utc)
        future_time = now + timedelta(hours=self.min_future)
        filtered = []
        for e in events:
            ct = e.get("commence_time")
            if not ct: continue
            try:
                start = datetime.fromisoformat(ct.replace("Z", "+00:00"))
                if start >= future_time:
                    filtered.append(e)
            except: pass
        return filtered

    def fetch_all_odds(self):
        all_events = []
        for key in self.sport_keys:
            events = self.fetch_odds(key)
            label = key.replace("soccer_","Football (").replace("basketball_","").replace("tennis_","Tennis (") + ")"
            for e in events: e["sport_label"] = label
            all_events.extend(events)
        logger.info(f"Future events: {len(all_events)}")
        return all_events

    def fetch_scores(self, sport_key, days):
        data = self._get(f"sports/{sport_key}/scores/", {"daysFrom": days})
        return data if isinstance(data, list) else data.get("data", [])

# ------------------------- 3. LLM Adjuster -------------------------
class LLMAdjuster:
    def __init__(self, config):
        self.gemini_key = os.getenv("GEMINI_API_KEY")
        self.tavily_key = os.getenv("TAVILY_API_KEY")
        self.model_name = config["llm_model"]
        self.max_news = config["tavily_max_results"]
        if not self.gemini_key or not self.tavily_key:
            self.enabled = False
            return
        genai.configure(api_key=self.gemini_key)
        self.model = genai.GenerativeModel(self.model_name)
        self.tavily = TavilyClient(api_key=self.tavily_key)
        self.enabled = True

    def adjust(self, home, away, base_prob):
        if not self.enabled: return base_prob, "disabled"
        try:
            query = f"{home} {away} injury report team news"
            search = self.tavily.search(query, max_results=self.max_news)
            context = "\n".join([r["content"] for r in search.get("results", [])]) or "No news"
            prompt = f"Match: {home} vs {away}. Base home win prob: {base_prob:.2%}. News: {context}\nReturn JSON: {{\"adjusted_prob\": float, \"reasoning\": str}}"
            resp = self.model.generate_content(prompt)
            text = resp.text.strip().replace("```json","").replace("```","")
            data = json.loads(text)
            adj = float(data.get("adjusted_prob", base_prob))
            adj = max(0.05, min(0.95, adj))
            return adj, data.get("reasoning", "")
        except Exception as e:
            return base_prob, f"Error: {e}"

# ------------------------- 4. Pick Generator -------------------------
class PickGenerator:
    @staticmethod
    def teams(e): return e.get("home_team","?"), e.get("away_team","?")
    @staticmethod
    def outcomes(e):
        prices = {"home":2.0,"away":2.0,"draw":3.0}
        for bk in e.get("bookmakers",[]):
            if bk.get("key")=="pinnacle":
                for m in bk.get("markets",[]):
                    if m.get("key")=="h2h":
                        for o in m.get("outcomes",[]):
                            n = o.get("name","").lower()
                            if "home" in n: prices["home"]=float(o["price"])
                            elif "away" in n: prices["away"]=float(o["price"])
                            elif "draw" in n: prices["draw"]=float(o["price"])
                return prices
        if e.get("bookmakers"):
            for m in e["bookmakers"][0].get("markets",[]):
                if m.get("key")=="h2h":
                    for o in m.get("outcomes",[]):
                        n = o.get("name","").lower()
                        if "home" in n: prices["home"]=float(o["price"])
                        elif "away" in n: prices["away"]=float(o["price"])
                        elif "draw" in n: prices["draw"]=float(o["price"])
        return prices

    @staticmethod
    def select_uncorrelated(candidates, count, min_prob):
        used = set()
        sel = []
        for c in sorted(candidates, key=lambda x: -x["prob"]):
            if c["prob"] < min_prob: continue
            if not (c["teams"] & used):
                sel.append(c)
                used.update(c["teams"])
                if len(sel)>=count: break
        return sel

    @staticmethod
    def generate(events, adj_probs, config):
        enriched = []
        for e in events:
            h,a = PickGenerator.teams(e)
            outs = PickGenerator.outcomes(e)
            for outcome in ["home","away","draw"]:
                price = outs[outcome]
                if price<=1.0: continue
                implied = 1/price
                prob = adj_probs.get(f"{h} vs {a}", implied) if outcome=="home" else implied
                edge = prob-implied
                enriched.append({"event":e,"home":h,"away":a,"outcome":outcome,"prob":prob,"odds":price,"edge":edge,"teams":{h,a}})
        all_sorted = sorted(enriched, key=lambda x: -x["prob"])
        sure = PickGenerator.select_uncorrelated(
            [x for x in all_sorted if x["prob"]>=config["sure_prob"] and x["edge"]>config["sure_edge"]],
            config["max_sure"], config["sure_prob"])
        odds_p = PickGenerator.select_uncorrelated(
            [x for x in all_sorted if x["prob"]>=config["odds_prob"] and x["edge"]>config["odds_edge"]],
            config["max_odds"], config["odds_prob"])
        acca_cand = [x for x in all_sorted if config["acca_prob_low"]<=x["prob"]<=config["acca_prob_high"] and x["edge"]>config["acca_edge"]]
        if len(acca_cand)<config["max_acca"]:
            acca_cand = [x for x in all_sorted if config["fallback_acca_low"]<=x["prob"]<=config["fallback_acca_high"] and x["edge"]>config["fallback_acca_edge"]]
        acca = PickGenerator.select_uncorrelated(acca_cand, config["max_acca"], 0.0)
        return {"sure_bets":sure,"odds_bets":odds_p,"acca_bets":acca}

# ------------------------- 5. Results & Tracking -------------------------
def compute_yesterday_results(config):
    results_file = config["results_file"]
    if not Path(results_file).exists():
        return
    with open(results_file) as f:
        history = json.load(f)
    if not history: return
    last = None
    for entry in reversed(history):
        if not entry.get("results_computed"):
            last = entry
            break
    if not last: return
    ts = datetime.fromisoformat(last["timestamp"])
    days_ago = (datetime.now(timezone.utc)-ts).days
    if days_ago < 1:
        logger.info("Picks too recent to have scores.")
        return
    logger.info(f"Checking results for {ts.date()}")
    collector = OddsCollector(config)
    all_scores = {}
    for sport in config["sports"]:
        all_scores[sport] = collector.fetch_scores(sport, days_ago)
    total_profit = 0.0
    results_list = []
    for cat, picks in last["picks"].items():
        for pick in picks:
            home, away, outcome, odds = pick["home"], pick["away"], pick["outcome"], pick["odds"]
            matched = None
            for sport, score_events in all_scores.items():
                for se in score_events:
                    if se.get("home_team")==home and se.get("away_team")==away:
                        matched = se; break
                if matched: break
            if not matched:
                res, profit, score = "pending", 0.0, "N/A"
            else:
                h_score = matched.get("home_score") or matched.get("score",{}).get("home")
                a_score = matched.get("away_score") or matched.get("score",{}).get("away")
                if h_score is None or a_score is None:
                    res, profit, score = "pending", 0.0, "N/A"
                else:
                    score = f"{h_score}-{a_score}"
                    if outcome=="home" and h_score>a_score: res, profit = "win", odds-1
                    elif outcome=="away" and a_score>h_score: res, profit = "win", odds-1
                    elif outcome=="draw" and h_score==a_score: res, profit = "win", odds-1
                    else: res, profit = "lose", -1.0
            results_list.append({**pick, "result":res, "profit":profit, "actual_score":score})
            total_profit += profit
    print("\n📊 YESTERDAY'S RESULTS")
    print("-"*60)
    for p in results_list:
        emoji = "✅" if p["result"]=="win" else ("❌" if p["result"]=="lose" else "⏳")
        print(f"  {emoji} {p['home']} vs {p['away']} | {p['outcome'].upper()} | {p['odds']} | {p['result']} | {p['actual_score']} | {p['profit']:+.2f}")
    print(f"\n💰 Day profit: {total_profit:+.2f} units")
    last["results"] = results_list
    last["results_computed"] = True
    with open(results_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("Yesterday's results saved.")

def display_picks(title, picks, emoji):
    print(f"\n{emoji} {title}")
    print("-"*60)
    if not picks:
        print("  No picks")
        return
    for i,p in enumerate(picks,1):
        print(f"  {i:2}. {p['home']} vs {p['away']} | {p['outcome'].upper()} | Prob {p['prob']:.1%} | Odds {p['odds']:.2f} | Edge {p['edge']:.2%}")

def record_new_picks(picks, config):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results_computed": False,
        "picks": {
            cat: [{"home":p["home"],"away":p["away"],"outcome":p["outcome"],
                   "prob":p["prob"],"odds":p["odds"],"edge":p["edge"]} for p in plist]
            for cat, plist in picks.items()
        }
    }
    results_file = config["results_file"]
    history = []
    if Path(results_file).exists():
        with open(results_file) as f:
            history = json.load(f)
    history.append(record)
    with open(results_file, "w") as f:
        json.dump(history, f, indent=2)
    logger.info("New picks saved.")

# ------------------------- 6. Main -------------------------
def main():
    # Ensure volume/data directory exists
    Path("data").mkdir(exist_ok=True)
    ensure_dependencies()
    load_dotenv()
    setup_api_keys()
    config = load_config()
    logger.info("ABPA Agent start")

    # 0. Show yesterday
    compute_yesterday_results(config)

    # 1. Odds
    collector = OddsCollector(config)
    events = collector.fetch_all_odds()
    if not events:
        logger.error("No events.")
        return

    # 2. LLM
    top_events = sorted(events, key=lambda e: len(e.get("bookmakers",[])), reverse=True)[:config["max_events_for_llm"]]
    adjuster = LLMAdjuster(config)
    adj_probs = {}
    for idx,e in enumerate(top_events,1):
        h,a = PickGenerator.teams(e)
        base_odds = PickGenerator.outcomes(e)["home"]
        base_prob = 1/base_odds if base_odds>0 else 0.5
        adj, reason = adjuster.adjust(h,a,base_prob)
        adj_probs[f"{h} vs {a}"] = adj
        logger.info(f"[{idx}/{len(top_events)}] {h} vs {a}: {base_prob:.1%} → {adj:.1%} ({reason[:50]})")

    # 3. Picks
    picks = PickGenerator.generate(events, adj_probs, config)

    # 4. Display
    print("\n"+"="*70)
    print("🏆 TODAY'S PICKS")
    print("="*70)
    display_picks("SURE BETS", picks["sure_bets"], "🔥")
    display_picks("RECOMMENDED", picks["odds_bets"], "📈")
    display_picks("ACCUMULATOR", picks["acca_bets"], "🎰")
    acca = picks["acca_bets"]
    if acca:
        total_odds = 1.0
        for p in acca: total_odds *= p["odds"]
        print(f"\n🎯 Acca odds: {total_odds:.2f} | Est win: {1/total_odds:.2%}")

    # 5. Store
    record_new_picks(picks, config)
    logger.info("Done.")

if __name__ == "__main__":
    main()
