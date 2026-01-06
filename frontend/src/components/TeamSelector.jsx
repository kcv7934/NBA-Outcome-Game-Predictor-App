import { useState } from "react";

const teams = [
  { abbr: "BOS", name: "Celtics" },
  { abbr: "BRK", name: "Nets" },
  { abbr: "NYK", name: "Knicks" },
  { abbr: "PHI", name: "76ers" },
  { abbr: "TOR", name: "Raptors" },
  { abbr: "CHI", name: "Bulls" },
  { abbr: "CLE", name: "Cavaliers" },
  { abbr: "DET", name: "Pistons" },
  { abbr: "IND", name: "Pacers" },
  { abbr: "MIL", name: "Bucks" },
  { abbr: "ATL", name: "Hawks" },
  { abbr: "CHO", name: "Hornets" },
  { abbr: "MIA", name: "Heat" },
  { abbr: "ORL", name: "Magic" },
  { abbr: "WAS", name: "Wizards" },
  { abbr: "DEN", name: "Nuggets" },
  { abbr: "MIN", name: "Timberwolves" },
  { abbr: "OKC", name: "Thunder" },
  { abbr: "POR", name: "Blazers" },
  { abbr: "UTA", name: "Jazz" },
  { abbr: "GSW", name: "Warriors" },
  { abbr: "LAC", name: "Clippers" },
  { abbr: "LAL", name: "Lakers" },
  { abbr: "PHX", name: "Suns" },
  { abbr: "SAC", name: "Kings" },
  { abbr: "DAL", name: "Mavericks" },
  { abbr: "HOU", name: "Rockets" },
  { abbr: "MEM", name: "Grizzlies" },
  { abbr: "NOP", name: "Pelicans" },
  { abbr: "SAS", name: "Spurs" }
];

export default function TeamSelector({ onSubmit }) {
    const [home, setHome] = useState("");
    const [away, setAway] = useState("");
    const [error, setError] = useState("");

    const handleSubmit = (e) => {
        e.preventDefault();

        if (!home || !away) {
            setError("Please select both home and away teams.");
            return;
        }

        if (home === away) {
            setError("Home and away teams cannot be the same!");
            return;
        }

        setError(""); 
        onSubmit({ home, away });
    };

    return (
        <form onSubmit={handleSubmit} className="team-selector-card">
            <h2>NBA Game Predictor</h2>

            {error && <p className="error">{error}</p>}

            <label>Home Team</label>
            <select value={home} onChange={e => setHome(e.target.value)}>
                <option value="">Select team</option>
                {teams.map(t => (
                    <option key={t.abbr} value={t.abbr}>
                        {t.abbr} - {t.name}
                    </option>
                ))}
            </select>

            <label>Away Team</label>
            <select value={away} onChange={e => setAway(e.target.value)}>
                <option value="">Select team</option>
                {teams.map(t => (
                    <option key={t.abbr} value={t.abbr}>
                        {t.abbr} - {t.name}
                    </option>
                ))}
            </select>

            <button type="submit">Predict Game</button>
        </form>
    );
}
