# NBA Game Predictor

This project predicts NBA game outcomes using historical game data and rolling team statistics. The prediction pipeline uses machine learning models trained on six NBA seasons (2020–2026) and evaluated through walk-forward season backtesting.

A **React frontend** communicates with a **Flask backend API**, which serves predictions generated from trained machine learning models. Both parts of the application can be run locally or containerized using **Docker** and **Docker Compose**.

## Model Performance

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
│   ├── Dockerfile                # Docker configuration for Flask
│   ├── app.py                    # Flask API for predictions
│   ├── ml/
│   │   ├── data/
│   │   │   └── rolling_df.csv    # Rolling features for model input
│   │   ├── models/               # Trained models and selected features
│   │   ├── predictor/
│   │   │   └── ensemble_predictor.py
│   │   └── predict_game.py       # Wrapper function for API use
│   ├── scrape/                   # Optional scripts for data updating
│   │   ├── fetch_nba_seasons.py
│   │   ├── read_nba_seasons.py
│   │   ├── parse_nba_data.py
│   │   ├── preprocess_nba_data.py
│   │   └── data/nba_games.csv
│
├── frontend/
│   ├── Dockerfile                # Docker configuration for React
│   ├── src/
│   │   ├── App.jsx               # Main React component
│   │   └── components/
│   │       └── TeamSelector.jsx  # Team selection component
│   ├── package.json
│   └── vite.config.js
│
├── screenshots/
│   ├── team_selection.png
│   └── prediction_result.png
│
├── compose.yaml                  # Runs frontend and backend together
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Data and Model Training Notes

- Models were trained on NBA games from the **start of the 2020 season through 01/02/2026**.
- All rolling statistics are computed from this dataset.
- Scraping scripts in `backend/scrape/` are optional and are provided to show how the dataset was generated.
- You do **not** need to run the scraping pipeline to use the application or prediction API.

> ⚠️ Running the scraping scripts can take several hours to a full day depending on the season range.

> ⚠️ **Important:** To predict games using data newer than 01/02/2026, you must:
>
> 1. Run `fetch_nba_seasons.py` to download updated season schedules.
> 2. Run `read_nba_seasons.py` to download game box scores.
> 3. Run `parse_nba_data.py` to generate an updated `nba_games.csv`.
> 4. Retrain the models to generate updated feature data and model files.

---

## Backend API

The Flask backend exposes endpoints for NBA game predictions.

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

## Frontend

The React frontend provides an interface for selecting home and away teams and viewing predictions.

### Features

- Team selection dropdowns with NBA team abbreviations and names.
- Validation preventing the same team from being selected twice.
- Displays the predicted winner, loser, and win probabilities.
- Shows model training information directly in the UI.
- Responsive styling for different screen sizes.

---

## Example App Screenshots

### Team Selection

![Team Selection](./screenshots/team_selection.png)

*Dropdown menus for selecting the home and away teams.*

### Prediction Result

![Prediction Result](./screenshots/prediction_result.png)

*Displays the predicted winner and each team's win probability.*

---

## Running with Docker Compose

Docker Compose is the easiest way to run the Flask backend and React frontend together.

### Prerequisites

Make sure Docker Desktop is installed and running.

### Start the Application

From the main project directory, run:

```bash
docker compose up --build
```

Docker Compose will:

- Build the Flask backend image.
- Build the React frontend image.
- Start both containers.
- Map the backend to port `5000`.
- Map the frontend to port `3000`.

Open the application at:

```text
http://localhost:3000
```

The backend API is available at:

```text
http://localhost:5000
```

### Stop the Application

Press `Ctrl + C` in the terminal, then run:

```bash
docker compose down
```

After the images have already been built, the application can be started again with:

```bash
docker compose up
```

Use `--build` again after changing a Dockerfile, dependency file, or other build configuration:

```bash
docker compose up --build
```

---

## Running Each Docker Container Separately

The frontend and backend can also be built and run without Docker Compose.

Run all commands from the main project directory.

### Build the Backend Image

```bash
docker build -f backend/Dockerfile -t nba-backend .
```

### Run the Backend Container

```bash
docker run --name nba-backend-container -p 5000:5000 nba-backend
```

The backend will be available at:

```text
http://localhost:5000
```

### Build the Frontend Image

Open another terminal and run:

```bash
docker build -f frontend/Dockerfile -t nba-frontend .
```

### Run the Frontend Container

```bash
docker run --name nba-frontend-container -p 3000:3000 nba-frontend
```

The frontend will be available at:

```text
http://localhost:3000
```

### Stop the Containers

```bash
docker stop nba-backend-container nba-frontend-container
```

### Remove the Containers

```bash
docker rm nba-backend-container nba-frontend-container
```

---

## Running Locally Without Docker

The application can also be run directly with Python and Node.js.

### Run the Flask Backend

From the main project directory:

```bash
# Create a virtual environment
python -m venv venv
```

Activate the virtual environment:

```bash
# Windows
.\venv\Scripts\activate
```

```bash
# macOS/Linux
source venv/bin/activate
```

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Start the Flask API:

```bash
python backend/app.py
```

The backend will run at:

```text
http://localhost:5000
```

### Run the React Frontend

Open another terminal:

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
from backend.ml.predictor.ensemble_predictor import (
    predict_game_ensemble_weighted
)

rolling_df = pd.read_csv("backend/ml/data/rolling_df.csv")

ridge_model = joblib.load(
    "backend/ml/models/ridge_classifier_final.pkl"
)
ridge_predictors = joblib.load(
    "backend/ml/models/selected_predictors_ridge.pkl"
)

logistic_model = joblib.load(
    "backend/ml/models/logistic_model_final.pkl"
)
logistic_predictors = joblib.load(
    "backend/ml/models/selected_predictors_logistic.pkl"
)

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

- Load and clean historical NBA game data.
- Generate rolling team statistics using a 10-game window.
- Construct matchup-level features.
- Prevent future data leakage through rolling feature alignment.

### Model Training

- Logistic Regression predicts home-team win probabilities.
- Ridge Classifier predicts game winners.
- Sequential Feature Selection identifies the most predictive features.
- Models are trained using walk-forward season backtesting.

### Evaluation

- Logistic Regression achieved **62.85% backtest accuracy**.
- Ridge Classifier achieved **61.91% backtest accuracy**.
- Walk-forward validation was used to simulate real-world forecasting conditions.

---

## Notes

- Rolling features are computed using a **10-game window** by default.
- Predictions outside the 2020–2026 training range may be less reliable.
- Scraping scripts are included for reproducibility but are not required to use the application.
- The frontend communicates with the Flask API running on port `5000`.
- The frontend runs on port `3000`.
- Docker Compose allows both parts of the application to be started with one command.
