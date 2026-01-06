import pandas as pd
import joblib
from .predictor.ensemble_predictor import predict_game_ensemble_weighted
from pathlib import Path

# --- Set paths relative to this file ---
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "rolling_df.csv"
MODELS_FOLDER = ROOT / "models"

# --- Load rolling data ---
rolling_df = pd.read_csv(DATA_PATH)

# --- Load trained models and predictor columns ---
ridge_model = joblib.load(MODELS_FOLDER / "ridge_classifier_final.pkl")
ridge_predictors = joblib.load(MODELS_FOLDER / "selected_predictors_ridge.pkl")

logistic_model = joblib.load(MODELS_FOLDER / "logistic_model_final.pkl")
logistic_predictors = joblib.load(MODELS_FOLDER / "selected_predictors_logistic.pkl")


# --- Function to predict a single game ---
def predict_nba_game(home_team: str, away_team: str, ridge_weight: float = 0.02):
    """
    Predicts a single NBA game outcome using the ensemble predictor.

    Args:
        home_team (str): Home team abbreviation (e.g., 'LAL')
        away_team (str): Away team abbreviation (e.g., 'BOS')
        ridge_weight (float): Weight for the Ridge classifier in ensemble

    Returns:
        dict: {
            "home_team": str,
            "away_team": str,
            "predicted_winner": str,
            "predicted_loser": str,
            "home_win_prob": float,
            "away_win_prob": float
        }
    """
    home_team = home_team.upper()
    away_team = away_team.upper()

    result = predict_game_ensemble_weighted(
        rolling_df=rolling_df,
        ridge_model=ridge_model,
        ridge_predictors=ridge_predictors,
        logistic_model=logistic_model,
        logistic_predictors=logistic_predictors,
        home_team=home_team,
        away_team=away_team,
        ridge_weight=ridge_weight
    )

    return result
