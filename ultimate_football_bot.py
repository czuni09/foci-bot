import numpy as np
import pandas as pd
from scipy.stats import poisson
# Importáld az új modellt
from dixon_coles_model import fit_dixon_coles_model, predict_match

# ... (a meglévő importok maradnak: understat scraping, stb.)

class FootballPredictor:
    def __init__(self):
        self.params = None
        self.teams = None
        self.xi = 0.0018  # time decay, ~3-4 hónap félértékidő

    def prepare_data(self, matches_df):
        # Understat xG adatokkal bővítve, home_goals -> home_xg, away_goals -> away_xg
        df = matches_df[['date', 'home_team', 'away_team', 'home_xg', 'away_xg']].copy()
        df.rename(columns={'home_xg': 'home_goals', 'away_xg': 'away_goals'}, inplace=True)
        return df

    def train(self, historical_df):
        df = self.prepare_data(historical_df)
        self.params, self.teams = fit_dixon_coles_model(df, xi=self.xi)

    def predict(self, home_team, away_team):
        if self.params is None:
            raise ValueError("Model not trained yet!")

        pred = predict_match(self.params, self.teams, home_team, away_team)

        # Value calc oddsokkal (marad a régi logika)
        # BTTS, Over/Under számítás a score_matrix-ből
        btts_prob = np.sum(pred['score_matrix'][1:, 1:])
        over25_prob = 1 - np.sum(pred['score_matrix'][:3, :3])

        return {
            '1X2': (pred['home_win'], pred['draw'], pred['away_win']),
            'BTTS': btts_prob,
            'Over 2.5': over25_prob,
            'Expected Goals': (pred['expected_home_goals'], pred['expected_away_goals'])
        }

# A többi rész (scraping, modifiers hírek/időjárás/odds) marad változatlan
# Csak a predikciós hívást cseréld: predictor = FootballPredictor() -> train -> predict
