# Backtest Ligue 1

Outil de backtest pour comparer objectivement plusieurs modèles de pronostic
sur des matchs passés de Ligue 1, sans fuite de données (walk-forward strict).

## Pourquoi

Avant de faire confiance à un algo de pronostic, il faut le mesurer sur des
matchs déjà joués. Ce script simule ce que chaque modèle aurait prédit,
match par match, en n'utilisant que les données disponibles **avant** ce
match, puis compare aux résultats réels.

## Modèles comparés

| Modèle | Description |
|---|---|
| `algo_actuel` | Formule simple attaque/défense (moyenne des 5 derniers matchs, arrondie) |
| `poisson` | Modèle de Poisson attaque/défense, calibré sur la moyenne de buts en cours de saison, avec avantage du terrain |
| `baseline_1-1` | Prédit toujours 1-1 (référence triviale) |
| `baseline_prior` | Probabilités 1N2 fixes (~45 % dom. / 27 % nul / 28 % ext.), sert de référence pour le Brier score / log-loss |

## Installation

```bash
git clone <url-de-ton-repo>
cd backtest-ligue1
pip install -r requirements.txt
```

## Clé API

Le script utilise l'API gratuite [football-data.org](https://www.football-data.org/client/register).

```bash
export FOOTBALL_API_TOKEN="ta_cle"        # macOS / Linux
$env:FOOTBALL_API_TOKEN="ta_cle"          # Windows PowerShell
```

Le palier gratuit limite l'historique disponible : commence avec des saisons
récentes (`2023`, `2024`, `2025`) si une saison plus ancienne renvoie une erreur.

## Utilisation

```bash
python3 backtest_ligue1.py --seasons 2023 2024 2025
```

Options utiles :

| Option | Défaut | Description |
|---|---|---|
| `--n-history` | 5 | Nombre de matchs récents utilisés pour la forme |
| `--min-history` | 3 | Nombre minimum de matchs avant de commencer à prédire |
| `--home-advantage` | 1.15 | Facteur d'avantage du terrain (modèle Poisson) |
| `--out-dir` | `backtest_output` | Dossier de sortie des résultats |
| `--cache-dir` | `cache` | Dossier de cache des données API (JSON par saison) |
| `--force-refresh` | — | Ignore le cache et réinterroge l'API |

## Résultats

- `backtest_output/rapport_backtest.txt` — tableau comparatif des modèles
  (score exact, résultat 1N2, erreur moyenne sur les buts, Brier score, log-loss)
- `backtest_output/predictions_detail.csv` — le détail match par match

## Limites connues

- Ne teste **que** la partie "algo" : le consensus du site MPP n'est pas
  disponible historiquement, donc pas inclus dans ce backtest.
- La forme des équipes est réinitialisée à chaque saison (pas de report
  d'une saison à l'autre), pour éviter les effets de transferts/mercato.
- Le facteur d'avantage du terrain (1.15 par défaut) est une estimation de
  départ, pas calibré spécifiquement sur la Ligue 1 — à ajuster une fois que
  tu as assez de données de backtest.

## Tests

Un jeu de tests avec données simulées (pas d'appel API) est disponible pour
vérifier la logique interne (formes, modèle de Poisson, métriques, absence
de fuite de données) :

```bash
python3 test_backtest.py
```
