# NBA Game Predictor

This project predicts NBA game outcomes using historical game data and rolling team statistics. The prediction pipeline uses machine learning models trained on six NBA seasons (2020–2026) and evaluated through walk-forward season backtesting.

A **React frontend** communicates with a **Flask backend API**, which serves predictions generated from trained machine learning models.

### Model Performance

| Model               | Walk-Forward Backtest Accuracy |
| ------------------- | ------------------------------ |
| Logistic Regression | 62.85%                         |
| Ridge Classifier    | 61.91%                         |

The Logistic Regression model achieved the strongest historical performance and serves as the primary benchmark for prediction quality.

---

## Project Structure

```text
.
├── backend/
│   ├── app.py                    # Flask API for predictions
│   ├── ml/
│   │   ├── data/
│   │   │   └── rolling_df.csv    # Rolling features for model input
│   │   ├── predictor/
│   │   │   └── ensemble_predictor.py # Prediction utilities and experimentation
│   │   └── predict_game.py       # Wrapper function for API use
│   ├── scrape/                   # Optional, only for data updating
│   │   ├── fetch_nba_seasons.py
│   │   ├── read_nba_seasons.py
│   │   ├── parse_nba_data.py
│   │   ├── preprocess_nba_data.py
│   │   └── data/nba_games.csv
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx               # Main React component
│   │   └── components/
│   │       └── TeamSelector.jsx  # Team selection component
│   └── package.json
```

---

## Data and Model Training Notes

* Models were trained on NBA games from the **start of the 2020 season through 01/02/2026**.
* All rolling statistics are computed from this dataset.
* Scraping scripts (`backend/scrape/`) are **optional** and provided only to show how the dataset was generated.
* You do **not need to run the scraping pipeline** to use the application or prediction API.

> ⚠️ Running the scraping scripts can take several hours to a full day depending on the season range.

> ⚠️ **Important:** If you want to predict games using data newer than 01/02/2026, you must:
>
> 1. Run `fetch_nba_seasons.py` to download updated season schedules.
> 2. Run `read_nba_seasons.py` to download game box scores.
> 3. Run `parse_nba_data.py` to generate an updated `nba_games.csv`.
> 4. Retrain the models to generate updated feature data and model files.

---

## Backend API (Flask)

The Flask backend exposes endpoints for NBA game predictions.

### Endpoints

### GET `/`

Returns a simple status message:

```json
{
  "message": "NBA Predictor API is running"
}
```

### POST `/predict`

Returns a prediction for a single game.

#### Request

```json
{
  "home": "LAL",
  "away": "BOS"
}
```

#### Response

```json
{
  "home": "LAL",
  "away": "BOS",
  "predicted_winner": "BOS",
  "predicted_loser": "LAL",
  "home_win_prob": 45.3,
  "away_win_prob": 54.7
}
```

---

## Frontend (React)

The frontend provides a simple interface for selecting home and away teams and viewing predictions.

### Features

* Team selection dropdowns with NBA team abbreviations and names.
* Validation preventing the same team from being selected twice.
* Displays predicted winner, loser, and win probabilities.
* Shows model training information directly in the UI.

---

## Example App Screenshots

### Team Selection

![Team Selection Placeholder](./screenshots/team_selection.png)

*Dropdown menu to select home and away teams.*

### Prediction Result

![Prediction Result Placeholder](./screenshots/prediction_result.png)

*Displays predicted winner and win probabilities.*

---

## Usage

### Run Flask Backend

```bash
# Activate virtual environment
.\venv\Scripts\activate      # Windows
source venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start Flask API
python backend/app.py
```

### Run React Frontend

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:3000
```

in your browser.

---

## Predict a Single Game in Python

```python
import pandas as pd
import joblib
from backend.ml.predictor.ensemble_predictor import predict_game_ensemble_weighted

rolling_df = pd.read_csv("backend/ml/data/rolling_df.csv")

ridge_model = joblib.load("backend/ml/models/ridge_classifier_final.pkl")
ridge_predictors = joblib.load("backend/ml/models/selected_predictors_ridge.pkl")

logistic_model = joblib.load("backend/ml/models/logistic_model_final.pkl")
logistic_predictors = joblib.load("backend/ml/models/selected_predictors_logistic.pkl")

result = predict_game_ensemble_weighted(
    rolling_df,
    ridge_model,
    ridge_predictors,
    logistic_model,
    logistic_predictors,
    home_team="LAL",
    away_team="BOS"
)

print(result)
```

---

## Machine Learning Pipeline

### Data Preparation

* Load and clean historical NBA game data.
* Generate rolling team statistics using a 10-game window.
* Construct matchup-level features.
* Prevent future data leakage through rolling feature alignment.

### Model Training

* Logistic Regression predicts home-team win probabilities.
* Ridge Classifier predicts game winners.
* Sequential Feature Selection identifies the most predictive features.
* Models are trained using walk-forward season backtesting.

### Evaluation

* Logistic Regression achieved **62.85% backtest accuracy**.
* Ridge Classifier achieved **61.91% backtest accuracy**.
* Walk-forward validation was used to simulate real-world forecasting conditions.

---

## Notes

* Rolling features are computed using a **10-game window** by default.
* Predictions outside the 2020–2026 training range may be less reliable.
* Scraping scripts are included for reproducibility but are not required to use the application.
* Frontend styling is responsive and provides validation for invalid team selections.
