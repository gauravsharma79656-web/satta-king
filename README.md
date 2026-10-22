# AI Number Predictor & Game Relation Analyzer

This application analyzes results for **Faridabad (FRBD)**, **Ghaziabad (GBD)**, **Gali (GALI)**, and **Desawar (DSWR)** from [satta-king-fast.com](https://satta-king-fast.com), integrates your historical dataset (`history.csv`), extracts cross-game mathematical relations, and runs trained Machine Learning models (Random Forest, Markov Chain, and Haruf Transfer Priors) to provide real-time predictions with a single click.

---

## 🚀 Key Features

1. **⚡ 1-Click Sync & Predict All**:
   - Automatically checks and scrapes the latest draw results from `satta-king-fast.com`.
   - Merges and cleans missing values (`XX` vs declared numbers).
   - Retrains the Machine Learning models on over **1,025+ historical days** (2024 to 2026).
   - Produces predictions simultaneously for all 4 games.

2. **🎯 Game Prediction Cards**:
   - **Top Recommended Jodi**: High confidence synthesized number (00-99).
   - **Top 5 Primary Jodis**: Ranked by joint probability $P(\text{Tens}=i) \times P(\text{Ones}=j) \times \text{RelationBoost}$.
   - **Top 5 Support Jodis**: Alternative mirror and Rashi pairs.
   - **Solid Andar Haruf**: Dominant tens digit with confidence %.
   - **Solid Bahar Haruf**: Dominant ones digit with confidence %.
   - **Crossing Digits (Chauka/Panja)**: 4-5 high probability digits for permutation combinations.
   - **Backtest Verification**: Historical hit rates across the last 30 draws.

3. **🔗 Cross-Game Relational Intelligence**:
   - Analyzes sequential draw ordering:
     $$\text{DSWR (05:00 AM)} \longrightarrow \text{FRBD (18:15 PM)} \longrightarrow \text{GBD (20:30 PM)} \longrightarrow \text{GALI (23:30 PM)} \longrightarrow \text{DSWR (Next Day)}$$
   - Computes empirical **Haruf Transfer Rates**:
     - Direct digit passing rate
     - Rashi (mirror) passing rate (hits **59% - 61%** across sequential games)
   - Pairwise correlation and difference deltas.

4. **📊 Hot / Cold Frequency & Digit Chart**:
   - Most frequent (hot) numbers over the last 100 draws.
   - Due (cold) numbers that haven't appeared in the longest cycle.
   - Interactive bar chart for Andar vs Bahar digit distributions.

5. **📜 Master Historical Record**:
   - Complete record table with date filter and search.

---

## 💻 How to Run

### Method 1: Double-Click Launcher
Double-click `run_dashboard.bat` in `d:\S K`.

### Method 2: Command Line
```powershell
cd "d:\S K"
python app.py
```
Then open your browser at:
👉 **http://127.0.0.1:5000**

---

## 📁 File Structure

- `app.py`: Flask Web Server & REST API.
- `ai_engine.py`: Random Forest Classifiers, Markov transition priors, backtesting engine.
- `relation_analyzer.py`: Cross-game digit passing, correlations, and Rashi logic.
- `data_manager.py`: Scraper and synchronizer with `satta-king-fast.com`.
- `templates/index.html`: Responsive UI dashboard with Tailwind CSS & Chart.js.
- `data/all_games_dataset.csv`: Compiled master history dataset (1,025+ draws).
- `data/history_frbd_original.csv`: Preserved copy of your original FRBD `history.csv`.
- `run_dashboard.bat`: Windows launch batch script.
