from flask import Flask, request, jsonify
from flask_cors import CORS
from ml.predict_game import predict_nba_game

# Initialize Flask app
app = Flask(__name__)
# Enable CORS so the frontend can make requests from a different port
CORS(app)

@app.route("/")
def home():
    """
    Root endpoint of the API.
    
    Returns
    -------
    JSON
        A simple message indicating that the NBA Predictor API is running.
    """
    return jsonify({"message": "NBA Predictor API is running"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict the outcome of a single NBA game using the ensemble model.
    
    Expects a JSON payload with 'home' and 'away' team abbreviations.
    Returns the predicted winner, loser, and win probabilities for both teams.

    Request JSON
    ------------
    {
        "home": "LAL",
        "away": "BOS"
    }

    Returns JSON
    -----------
    {
        "home": "LAL",
        "away": "BOS",
        "predicted_winner": "BOS",
        "predicted_loser": "LAL",
        "home_win_prob": 45.3,
        "away_win_prob": 54.7
    }
    """
    # Get JSON data from the POST request
    data = request.get_json()
    home = data.get("home")
    away = data.get("away")

    # Validate that both teams are provided
    if not home or not away:
        return jsonify({"error": "Both home and away teams are required"}), 499
    
    # Call the prediction function from ml.predict_game
    prediction_res = predict_nba_game(home, away)

    # Convert probabilities to percentages and round to 1 decimal place
    home_win_pct = round(prediction_res["home_win_prob"] * 100, 1)
    away_win_pct = round(prediction_res["away_win_prob"] * 100, 1)

    # Return JSON response with prediction results
    return jsonify({
        "home": home,
        "away": away,
        "predicted_winner": prediction_res["predicted_winner"],
        "predicted_loser": prediction_res["predicted_loser"],
        "home_win_prob": home_win_pct,
        "away_win_prob": away_win_pct
    })    

# Run the app in debug mode if this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)
