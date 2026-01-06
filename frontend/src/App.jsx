import { useState } from "react";
import TeamSelector from "./components/TeamSelector";
import "./App.css";

function App() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(null);

  const handlePredict = async ({ home, away }) => {
    setLoading(true);

    try {
      const res = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ home, away })
      });

      if (!res.ok) 
        throw new Error("Failed to fetch prediction");
      
      const data = await res.json();

      setResult(data);
    }
    catch(err) {
      console.error(err);
      setResult({ prediction: "Error fetching prediction "});
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <p className="note">
        Note: The current prediction model has been trained on NBA games from the beginning of the 2020 season through 01/02/2026.
      </p>

      <h1 className="header">NBA Predictor</h1>

      <TeamSelector onSubmit={handlePredict}/>

      {loading && <p className="loading">Loading...</p>}

      {result && (
        <div className="result-card">
          <p><strong>Home:</strong> {result.home}</p>
          <p><strong>Away:</strong> {result.away}</p>
          <p><strong>Predicted Winner:</strong> {result.predicted_winner}</p>
          <p><strong>Predicted Loser:</strong> {result.predicted_loser}</p>
          <p><strong>Home win probability:</strong> {result.home_win_prob.toFixed(1)}%</p>
          <p><strong>Away win probability:</strong> {result.away_win_prob.toFixed(1)}%</p>
        </div>
      )}
    </div>
  )
}

export default App;