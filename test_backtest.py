"""Tests avec donnees simulees (pas d'appel API reel)."""
import random
from pathlib import Path
from unittest.mock import patch

import backtest_ligue1 as bt


def make_fake_season(n_teams=10, n_rounds=6, seed=1):
    """Genere un mini-championnat rond-robin simple avec des scores aleatoires plausibles."""
    random.seed(seed)
    teams = [f"Team{i}" for i in range(n_teams)]
    matches = []
    date_counter = 0
    for rnd in range(n_rounds):
        random.shuffle(teams)
        for i in range(0, len(teams), 2):
            if i + 1 >= len(teams):
                continue
            home, away = teams[i], teams[i + 1]
            hg = random.choices([0, 1, 2, 3, 4], weights=[20, 35, 25, 15, 5])[0]
            ag = random.choices([0, 1, 2, 3, 4], weights=[25, 35, 22, 13, 5])[0]
            date_counter += 1
            matches.append(
                {
                    "date": f"2023-08-{date_counter:02d}T18:00:00Z" if date_counter <= 28 else f"2023-09-{date_counter-28:02d}T18:00:00Z",
                    "matchday": rnd + 1,
                    "home": home,
                    "away": away,
                    "home_goals": hg,
                    "away_goals": ag,
                }
            )
    matches.sort(key=lambda m: m["date"])
    return matches


def test_compute_form():
    hist = [(2, 1), (1, 1), (0, 3), (3, 0), (1, 2)]
    form = bt.compute_form(hist, n_history=3)
    assert form is not None
    # last 3 = (0,3),(3,0),(1,2) -> avg_scored=(0+3+1)/3, avg_conceded=(3+0+2)/3
    assert abs(form.avg_scored - (0 + 3 + 1) / 3) < 1e-9
    assert abs(form.avg_conceded - (3 + 0 + 2) / 3) < 1e-9
    assert form.n_matches == 3
    print("✅ compute_form OK")


def test_compute_form_empty():
    assert bt.compute_form([], n_history=5) is None
    print("✅ compute_form (vide) OK")


def test_predict_current_algo_matches_original_formula():
    form_h = bt.TeamForm(avg_scored=2.0, avg_conceded=1.0, n_matches=5)
    form_a = bt.TeamForm(avg_scored=1.0, avg_conceded=2.0, n_matches=5)
    pred = bt.predict_current_algo(form_h, form_a, league_avg=1.3)
    # buts_dom = round((2.0 + 2.0)/2) = 2 ; buts_ext = round((1.0 + 1.0)/2) = 1
    assert pred["home_goals"] == 2
    assert pred["away_goals"] == 1
    print("✅ predict_current_algo reproduit bien la formule originale")


def test_shrink_form_pulls_toward_league_average_when_small_sample():
    # Petit historique (2 matchs) tres au-dessus de la moyenne de la ligue
    raw = bt.TeamForm(avg_scored=4.0, avg_conceded=0.0, n_matches=2)
    league_avg = 1.3
    shrunk = bt.shrink_form(raw, league_avg, k=8.0)
    # Le resultat retrecis doit etre entre la valeur brute et la moyenne de la ligue,
    # et plus proche de la moyenne de la ligue vu le faible echantillon (n=2 << k=8)
    assert league_avg < shrunk.avg_scored < raw.avg_scored
    assert raw.avg_conceded < shrunk.avg_conceded < league_avg
    print(f"✅ shrink_form OK (avg_scored brut={raw.avg_scored} -> retreci={shrunk.avg_scored:.2f}, moyenne ligue={league_avg})")


def test_shrink_form_has_little_effect_with_large_sample():
    # Avec beaucoup de matchs (n=100 >> k=8), le retrecissement doit etre quasi nul
    raw = bt.TeamForm(avg_scored=2.0, avg_conceded=0.8, n_matches=100)
    shrunk = bt.shrink_form(raw, league_avg=1.3, k=8.0)
    assert abs(shrunk.avg_scored - raw.avg_scored) < 0.1
    print("✅ shrink_form a bien un effet negligeable avec un grand echantillon")


def test_poisson_shrink_is_less_extreme_than_poisson_on_small_sample():
    # Meme scenario "petit historique extreme" pour les deux equipes
    form_h = bt.TeamForm(avg_scored=4.0, avg_conceded=0.0, n_matches=2)
    form_a = bt.TeamForm(avg_scored=0.0, avg_conceded=4.0, n_matches=2)
    league_avg = 1.3

    raw_pred = bt.predict_poisson(form_h, form_a, league_avg)

    fh_shrunk = bt.shrink_form(form_h, league_avg, k=8.0)
    fa_shrunk = bt.shrink_form(form_a, league_avg, k=8.0)
    shrunk_pred = bt.predict_poisson(fh_shrunk, fa_shrunk, league_avg)

    # Le modele retreci doit etre moins categorique (probabilite de victoire domicile
    # plus proche de 0.5 / moins extreme) que le modele brut sur-confiant
    assert shrunk_pred["probs"]["H"] < raw_pred["probs"]["H"]
    print(
        f"✅ poisson_shrink moins extreme que poisson brut "
        f"(P(H) brut={raw_pred['probs']['H']:.3f} vs retreci={shrunk_pred['probs']['H']:.3f})"
    )


def test_predict_poisson_shapes():
    form_h = bt.TeamForm(avg_scored=2.0, avg_conceded=0.8, n_matches=5)
    form_a = bt.TeamForm(avg_scored=0.9, avg_conceded=1.5, n_matches=5)
    pred = bt.predict_poisson(form_h, form_a, league_avg=1.3)
    assert pred["home_goals"] >= 0 and pred["away_goals"] >= 0
    probs = pred["probs"]
    assert set(probs.keys()) == {"H", "D", "A"}
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-6, f"Les probas 1N2 doivent sommer a 1, obtenu {total}"
    # Une equipe qui marque beaucoup / encaisse peu à domicile doit être favorite
    assert probs["H"] > probs["A"]
    print(f"✅ predict_poisson OK (lambda_home={pred['lambda_home']}, lambda_away={pred['lambda_away']}, probs={probs})")


def test_metrics():
    assert bt.result_from_score(2, 1) == "H"
    assert bt.result_from_score(1, 2) == "A"
    assert bt.result_from_score(1, 1) == "D"

    probs = {"H": 0.6, "D": 0.25, "A": 0.15}
    brier = bt.brier_score_1x2(probs, "H")
    expected = (0.6 - 1) ** 2 + (0.25 - 0) ** 2 + (0.15 - 0) ** 2
    assert abs(brier - expected) < 1e-9

    ll = bt.log_loss_1x2(probs, "H")
    import math
    assert abs(ll - (-math.log(0.6))) < 1e-9

    # Un modele parfaitement sur de la bonne reponse -> brier = 0, log_loss ~ 0
    perfect = {"H": 1.0, "D": 0.0, "A": 0.0}
    assert bt.brier_score_1x2(perfect, "H") == 0.0
    print("✅ metriques (result_from_score, brier, log_loss) OK")


def test_full_backtest_loop_with_mocked_api():
    fake_matches = make_fake_season(n_teams=10, n_rounds=8)

    with patch.object(bt, "fetch_season_matches", return_value=fake_matches):
        rows = bt.run_backtest(
            seasons=[2023],
            token="fake-token",
            n_history=5,
            min_history=2,
            home_advantage=1.15,
            cache_dir=Path("/tmp/fake_cache_wont_be_used"),
        )

    assert len(rows) > 0, "Le backtest devrait produire des lignes de prediction"

    models_seen = {r["model"] for r in rows}
    assert models_seen == set(bt.MODELS.keys())

    summary = bt.summarize(rows)
    for model, s in summary.items():
        assert s["n_matches"] > 0
        if model in ("algo_actuel", "baseline_1-1"):
            assert s["exact_score_acc"] is not None
            assert s["brier_score"] is None  # ces modeles ne donnent pas de probas
        if model in ("poisson", "poisson_shrink", "baseline_prior"):
            assert s["brier_score"] is not None
            assert 0 <= s["brier_score"] <= 2  # brier max possible = 2

    table = bt.format_summary_table(summary)
    assert "algo_actuel" in table and "poisson" in table and "poisson_shrink" in table
    print("\n" + table)
    print("\n✅ Boucle complete de backtest (avec API mockee) OK")


if __name__ == "__main__":
    test_compute_form()
    test_compute_form_empty()
    test_predict_current_algo_matches_original_formula()
    test_shrink_form_pulls_toward_league_average_when_small_sample()
    test_shrink_form_has_little_effect_with_large_sample()
    test_poisson_shrink_is_less_extreme_than_poisson_on_small_sample()
    test_predict_poisson_shapes()
    test_metrics()
    test_full_backtest_loop_with_mocked_api()
    print("\n🎉 TOUS LES TESTS PASSENT")
