#!/usr/bin/env python3
"""
Backtest Ligue 1 - Compare plusieurs modeles de pronostic sur des saisons passees.

But : mesurer OBJECTIVEMENT si une version de l'algo est meilleure qu'une autre,
en simulant ce que chaque modele aurait predit match par match, sans jamais
utiliser d'informations du futur (walk-forward).

Modeles compares par defaut :
  - "algo_actuel"   : reproduit exactement la formule utilisee dans ton bot MPP
                      (moyenne simple attaque/defense, arrondie)
  - "poisson"       : modele de Poisson attaque/defense (plus robuste, donne
                      aussi des probabilites 1N2 exploitables)
  - "baseline_1-1"  : predit toujours 1-1 (score le plus frequent en general)
  - "baseline_prior": ne predit pas de score, seulement des probas 1N2 fixes
                      (~45% domicile / 27% nul / 28% exterieur), sert de
                      reference minimale pour le Brier score / log-loss

IMPORTANT :
  - Ce backtest ne teste QUE la partie "algo" (stats historiques). Il ne peut
    pas inclure le consensus MPP (%), car cette donnee n'est pas archivee
    historiquement et n'est disponible qu'en temps reel sur le site.
  - Necessite une cle API gratuite sur https://www.football-data.org
    (variable d'environnement FOOTBALL_API_TOKEN).
  - Le palier gratuit limite les requetes (~10/min) et l'historique de saisons
    disponibles. Ce script ne fait qu'UN appel API par saison (et met en cache
    le resultat localement), donc tu ne devrais quasiment jamais etre bloque.

Usage :
  export FOOTBALL_API_TOKEN="ta_cle"
  python3 backtest_ligue1.py --seasons 2022 2023 2024

  # Pour ne tester qu'un modele en particulier, ou changer la fenetre d'historique :
  python3 backtest_ligue1.py --seasons 2023 --n-history 8 --min-history 4
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import requests

API_BASE = "https://api.football-data.org/v4"
DEFAULT_COMPETITION = "FL1"  # Ligue 1

# Probabilites 1N2 "a priori" typiques d'un championnat europeen avec avantage
# du terrain (utilisees uniquement comme baseline de reference, pas une verite
# absolue - a ajuster si tu as des chiffres plus precis pour la Ligue 1).
PRIOR_PROBS = {"H": 0.45, "D": 0.27, "A": 0.28}


# --------------------------------------------------------------------------- #
# Recuperation des donnees (avec cache local pour ne pas re-interroger l'API)
# --------------------------------------------------------------------------- #

def fetch_season_matches(
    season: int,
    token: str,
    competition: str = DEFAULT_COMPETITION,
    cache_dir: Path = Path("cache"),
    force_refresh: bool = False,
) -> List[dict]:
    """Recupere tous les matchs d'une saison (1 seul appel API), avec cache JSON local."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{competition}_{season}.json"

    if cache_file.exists() and not force_refresh:
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{API_BASE}/competitions/{competition}/matches"
    headers = {"X-Auth-Token": token.strip()}
    params = {"season": season}

    for attempt in range(3):
        resp = requests.get(url, headers=headers, params=params, timeout=15)
        if resp.status_code == 429:
            wait = 30 * (attempt + 1)
            print(f"   ⏳ Rate limit atteint, attente {wait}s...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        break
    else:
        raise RuntimeError(f"Impossible de recuperer la saison {season} apres 3 tentatives")

    data = resp.json().get("matches", [])

    matches = []
    for m in data:
        if m.get("status") != "FINISHED":
            continue
        home_goals = m["score"]["fullTime"]["home"]
        away_goals = m["score"]["fullTime"]["away"]
        if home_goals is None or away_goals is None:
            continue
        matches.append(
            {
                "date": m["utcDate"],
                "matchday": m.get("matchday"),
                "home": m["homeTeam"]["name"],
                "away": m["awayTeam"]["name"],
                "home_goals": home_goals,
                "away_goals": away_goals,
            }
        )

    matches.sort(key=lambda x: x["date"])

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(matches, f, ensure_ascii=False, indent=2)

    return matches


# --------------------------------------------------------------------------- #
# Suivi de la forme des equipes (walk-forward, aucune fuite de donnees)
# --------------------------------------------------------------------------- #

@dataclass
class TeamForm:
    avg_scored: float
    avg_conceded: float
    n_matches: int


def compute_form(history: List[Tuple[int, int]], n_history: int) -> Optional[TeamForm]:
    """history = liste chronologique de (buts_marques, buts_encaisses) AVANT le match courant."""
    window = history[-n_history:]
    if not window:
        return None
    scored = [s for s, c in window]
    conceded = [c for s, c in window]
    return TeamForm(
        avg_scored=sum(scored) / len(scored),
        avg_conceded=sum(conceded) / len(conceded),
        n_matches=len(window),
    )


# --------------------------------------------------------------------------- #
# Modeles de prediction
# --------------------------------------------------------------------------- #

def predict_current_algo(form_h: TeamForm, form_a: TeamForm, league_avg: float) -> dict:
    """Reproduit exactement la formule de get_algo_prediction() du bot MPP."""
    buts_dom = round((form_h.avg_scored + form_a.avg_conceded) / 2)
    buts_ext = round((form_a.avg_scored + form_h.avg_conceded) / 2)
    return {"home_goals": buts_dom, "away_goals": buts_ext, "probs": None}


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        lam = 1e-6
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def predict_poisson(
    form_h: TeamForm,
    form_a: TeamForm,
    league_avg: float,
    home_advantage: float = 1.15,
    max_goals: int = 6,
) -> dict:
    """Modele de Poisson attaque/defense.

    lambda_home = force d'attaque domicile x force de defense exterieur x avantage terrain
    lambda_away = force d'attaque exterieur x force de defense domicile
    """
    league_avg = max(league_avg, 0.1)  # garde-fou

    attack_h = form_h.avg_scored / league_avg
    defense_a = form_a.avg_conceded / league_avg
    lambda_h = attack_h * defense_a * league_avg * home_advantage

    attack_a = form_a.avg_scored / league_avg
    defense_h = form_h.avg_conceded / league_avg
    lambda_a = attack_a * defense_h * league_avg

    # Matrice de probabilite de chaque scoreline possible
    score_probs = {}
    for h in range(max_goals + 1):
        for a in range(max_goals + 1):
            score_probs[(h, a)] = _poisson_pmf(h, lambda_h) * _poisson_pmf(a, lambda_a)

    best_score = max(score_probs, key=score_probs.get)

    prob_h = sum(p for (h, a), p in score_probs.items() if h > a)
    prob_d = sum(p for (h, a), p in score_probs.items() if h == a)
    prob_a = sum(p for (h, a), p in score_probs.items() if h < a)
    total = prob_h + prob_d + prob_a
    outcome_probs = {"H": prob_h / total, "D": prob_d / total, "A": prob_a / total}

    return {
        "home_goals": best_score[0],
        "away_goals": best_score[1],
        "probs": outcome_probs,
        "lambda_home": round(lambda_h, 2),
        "lambda_away": round(lambda_a, 2),
    }


def predict_baseline_11(form_h: TeamForm, form_a: TeamForm, league_avg: float) -> dict:
    return {"home_goals": 1, "away_goals": 1, "probs": None}


def predict_baseline_prior(form_h: TeamForm, form_a: TeamForm, league_avg: float) -> dict:
    """Ne propose pas de score (ce n'est pas son but), seulement des probas 1N2 fixes."""
    return {"home_goals": None, "away_goals": None, "probs": dict(PRIOR_PROBS)}


MODELS: Dict[str, Callable] = {
    "algo_actuel": predict_current_algo,
    "poisson": predict_poisson,
    "baseline_1-1": predict_baseline_11,
    "baseline_prior": predict_baseline_prior,
}


# --------------------------------------------------------------------------- #
# Metriques
# --------------------------------------------------------------------------- #

def result_from_score(h: int, a: int) -> str:
    if h > a:
        return "H"
    if h < a:
        return "A"
    return "D"


def brier_score_1x2(probs: Dict[str, float], actual: str) -> float:
    return sum((probs.get(r, 0.0) - (1.0 if r == actual else 0.0)) ** 2 for r in ("H", "D", "A"))


def log_loss_1x2(probs: Dict[str, float], actual: str, eps: float = 1e-10) -> float:
    p = max(probs.get(actual, 0.0), eps)
    return -math.log(p)


# --------------------------------------------------------------------------- #
# Boucle principale de backtest
# --------------------------------------------------------------------------- #

def run_backtest(
    seasons: List[int],
    token: str,
    n_history: int,
    min_history: int,
    home_advantage: float,
    cache_dir: Path,
    verbose: bool = True,
) -> List[dict]:
    rows = []

    for season in seasons:
        print(f"\n📥 Saison {season}...")
        matches = fetch_season_matches(season, token, cache_dir=cache_dir)
        print(f"   {len(matches)} matchs termines recuperes")

        team_history: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        goals_running_total = 0
        matches_running_total = 0

        n_predicted = 0
        for m in matches:
            home, away = m["home"], m["away"]
            actual_h, actual_a = m["home_goals"], m["away_goals"]

            h_hist = team_history[home]
            a_hist = team_history[away]

            if len(h_hist) >= min_history and len(a_hist) >= min_history and matches_running_total >= 20:
                form_h = compute_form(h_hist, n_history)
                form_a = compute_form(a_hist, n_history)
                league_avg = goals_running_total / (2 * matches_running_total)

                actual_result = result_from_score(actual_h, actual_a)

                for model_name, model_fn in MODELS.items():
                    pred = model_fn(form_h, form_a, league_avg)
                    row = {
                        "season": season,
                        "date": m["date"],
                        "match": f"{home} vs {away}",
                        "model": model_name,
                        "pred_home": pred["home_goals"],
                        "pred_away": pred["away_goals"],
                        "actual_home": actual_h,
                        "actual_away": actual_a,
                        "actual_result": actual_result,
                    }
                    if pred["home_goals"] is not None:
                        row["pred_result"] = result_from_score(pred["home_goals"], pred["away_goals"])
                        row["exact_match"] = int(
                            (pred["home_goals"], pred["away_goals"]) == (actual_h, actual_a)
                        )
                    else:
                        row["pred_result"] = None
                        row["exact_match"] = None
                    if pred.get("probs"):
                        row["brier"] = brier_score_1x2(pred["probs"], actual_result)
                        row["log_loss"] = log_loss_1x2(pred["probs"], actual_result)
                    else:
                        row["brier"] = None
                        row["log_loss"] = None
                    rows.append(row)
                n_predicted += 1

            # Mise a jour de l'historique APRES la prediction (pas de fuite)
            team_history[home].append((actual_h, actual_a))
            team_history[away].append((actual_a, actual_h))
            goals_running_total += actual_h + actual_a
            matches_running_total += 1

        print(f"   ✅ {n_predicted} matchs utilises pour le backtest (apres periode de chauffe)")

    return rows


# --------------------------------------------------------------------------- #
# Agregation des resultats + rapport
# --------------------------------------------------------------------------- #

def summarize(rows: List[dict]) -> Dict[str, dict]:
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)

    summary = {}
    for model, rs in by_model.items():
        n = len(rs)
        exact = [r["exact_match"] for r in rs if r["exact_match"] is not None]
        result_hits = [
            int(r["pred_result"] == r["actual_result"]) for r in rs if r["pred_result"] is not None
        ]
        mae_h = [
            abs(r["pred_home"] - r["actual_home"]) for r in rs if r["pred_home"] is not None
        ]
        mae_a = [
            abs(r["pred_away"] - r["actual_away"]) for r in rs if r["pred_away"] is not None
        ]
        briers = [r["brier"] for r in rs if r["brier"] is not None]
        loglosses = [r["log_loss"] for r in rs if r["log_loss"] is not None]

        summary[model] = {
            "n_matches": n,
            "exact_score_acc": sum(exact) / len(exact) if exact else None,
            "result_1x2_acc": sum(result_hits) / len(result_hits) if result_hits else None,
            "mae_home": sum(mae_h) / len(mae_h) if mae_h else None,
            "mae_away": sum(mae_a) / len(mae_a) if mae_a else None,
            "brier_score": sum(briers) / len(briers) if briers else None,
            "log_loss": sum(loglosses) / len(loglosses) if loglosses else None,
        }
    return summary


def format_summary_table(summary: Dict[str, dict]) -> str:
    headers = [
        "Modele", "N", "Score exact", "Resultat 1N2", "MAE dom.", "MAE ext.", "Brier", "LogLoss",
    ]
    col_w = [16, 6, 12, 13, 9, 9, 8, 8]

    def fmt_pct(v):
        return f"{v*100:.1f}%" if v is not None else "-"

    def fmt_num(v):
        return f"{v:.3f}" if v is not None else "-"

    lines = []
    lines.append(" | ".join(h.ljust(w) for h, w in zip(headers, col_w)))
    lines.append("-+-".join("-" * w for w in col_w))
    for model, s in sorted(summary.items(), key=lambda kv: (kv[1]["result_1x2_acc"] or 0), reverse=True):
        row = [
            model.ljust(col_w[0]),
            str(s["n_matches"]).ljust(col_w[1]),
            fmt_pct(s["exact_score_acc"]).ljust(col_w[2]),
            fmt_pct(s["result_1x2_acc"]).ljust(col_w[3]),
            fmt_num(s["mae_home"]).ljust(col_w[4]),
            fmt_num(s["mae_away"]).ljust(col_w[5]),
            fmt_num(s["brier_score"]).ljust(col_w[6]),
            fmt_num(s["log_loss"]).ljust(col_w[7]),
        ]
        lines.append(" | ".join(row))
    return "\n".join(lines)


def save_outputs(rows: List[dict], summary: Dict[str, dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "predictions_detail.csv"
    if rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    report_path = out_dir / "rapport_backtest.txt"
    table = format_summary_table(summary)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BACKTEST LIGUE 1 - RAPPORT DE COMPARAISON DES MODELES\n")
        f.write("=" * 60 + "\n\n")
        f.write(table + "\n\n")
        f.write(
            "Lecture :\n"
            "  - Score exact   : % de matchs ou le score predit est exactement juste\n"
            "  - Resultat 1N2  : % de matchs ou domicile/nul/exterieur est juste\n"
            "  - MAE dom./ext. : erreur moyenne (en buts) sur le score domicile/exterieur\n"
            "  - Brier / LogLoss : qualite des probabilites 1N2 (plus bas = meilleur,\n"
            "                      seulement calcule pour les modeles qui donnent des\n"
            "                      probabilites : poisson et baseline_prior)\n"
        )

    print(f"\n💾 Details : {csv_path}")
    print(f"💾 Rapport : {report_path}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Backtest de modeles de pronostic Ligue 1")
    parser.add_argument("--seasons", type=int, nargs="+", required=True, help="Ex: --seasons 2022 2023 2024")
    parser.add_argument("--token", type=str, default=os.environ.get("FOOTBALL_API_TOKEN"))
    parser.add_argument("--n-history", type=int, default=5, help="Fenetre de matchs recents utilisee (defaut: 5)")
    parser.add_argument("--min-history", type=int, default=3, help="Nb minimum de matchs avant de predire (defaut: 3)")
    parser.add_argument("--home-advantage", type=float, default=1.15, help="Facteur avantage domicile pour Poisson")
    parser.add_argument("--out-dir", type=str, default="backtest_output")
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--force-refresh", action="store_true", help="Ignore le cache et re-interroge l'API")
    args = parser.parse_args()

    if args.token:
        args.token = args.token.strip()

    if not args.token:
        print("❌ Aucun token API. Definis FOOTBALL_API_TOKEN ou passe --token TON_TOKEN")
        print("   (cle gratuite sur https://www.football-data.org/client/register)")
        sys.exit(1)

    if args.force_refresh:
        cache_dir = Path(args.cache_dir)
        for season in args.seasons:
            f = cache_dir / f"{DEFAULT_COMPETITION}_{season}.json"
            if f.exists():
                f.unlink()

    rows = run_backtest(
        seasons=args.seasons,
        token=args.token,
        n_history=args.n_history,
        min_history=args.min_history,
        home_advantage=args.home_advantage,
        cache_dir=Path(args.cache_dir),
    )

    if not rows:
        print("⚠️ Aucun match n'a pu etre utilise pour le backtest (historique insuffisant ?)")
        sys.exit(0)

    summary = summarize(rows)
    print("\n" + format_summary_table(summary))

    save_outputs(rows, summary, Path(args.out_dir))


if __name__ == "__main__":
    main()
