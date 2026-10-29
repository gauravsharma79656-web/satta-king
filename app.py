import os
import datetime
import joblib
from flask import Flask, render_template, jsonify, request
import pandas as pd

from data_manager import get_latest_dataset, sync_latest_today, build_full_dataset, ALL_GAMES_CSV
from relation_analyzer import RelationAnalyzer
from ai_engine import AIModelManager, GamePredictor, MODELS_DIR

app = Flask(__name__)

# Global instances
ai_manager = AIModelManager()
cached_dataset = None
cached_predictions = None
cached_backtests = None

def get_df(force_reload=False):
    global cached_dataset
    if cached_dataset is None or force_reload:
        cached_dataset = get_latest_dataset()
    return cached_dataset

def ensure_fresh_data():
    global cached_dataset
    try:
        new_df = sync_latest_today(min_interval=15)
        cached_dataset = new_df
        return new_df
    except Exception as e:
        print("ensure_fresh_data error:", e)
        return get_df()

def refresh_predictions():
    global cached_predictions, cached_backtests
    df = get_df()
    cached_predictions, cached_backtests = ai_manager.predict_all(df)
    return cached_predictions, cached_backtests

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/summary", methods=["GET"])
def api_summary():
    df = ensure_fresh_data()
    last_row = df.iloc[-1].to_dict()
    prev_row = df.iloc[-2].to_dict() if len(df) > 1 else last_row

    latest_declared = {}
    for g in ["DSWR", "FRBD", "GBD", "GALI"]:
        valid = df[df[g] != "XX"]
        if not valid.empty:
            r = valid.iloc[-1]
            latest_declared[g] = {
                "number": r[g],
                "date": r["date"],
                "is_today": (r["date"] == last_row["date"])
            }
        else:
            latest_declared[g] = {"number": "XX", "date": "", "is_today": False}

    return jsonify({
        "status": "success",
        "total_days": len(df),
        "latest_date": last_row["date"],
        "latest_results": {
            "DSWR": last_row.get("DSWR", "XX"),
            "FRBD": last_row.get("FRBD", "XX"),
            "GBD": last_row.get("GBD", "XX"),
            "GALI": last_row.get("GALI", "XX"),
        },
        "latest_declared": latest_declared,
        "prev_date": prev_row["date"],
        "prev_results": {
            "DSWR": prev_row.get("DSWR", "XX"),
            "FRBD": prev_row.get("FRBD", "XX"),
            "GBD": prev_row.get("GBD", "XX"),
            "GALI": prev_row.get("GALI", "XX"),
        }
    })

@app.route("/api/predict_all", methods=["GET"])
def api_predict_all():
    global cached_predictions, cached_backtests
    ensure_fresh_data()
    if cached_predictions is None or cached_backtests is None:
        refresh_predictions()
    return jsonify({
        "status": "success",
        "predictions": cached_predictions,
        "backtests": cached_backtests,
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    })

@app.route("/api/predict_by_date", methods=["GET"])
def api_predict_by_date():
    target_date = request.args.get("date", "").strip()
    if not target_date:
        return jsonify({"status": "error", "message": "Date parameter required (YYYY-MM-DD)"}), 400

    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
    yesterday_str = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    if target_date in [today_str, yesterday_str]:
        df = ensure_fresh_data()
    else:
        df = get_df()
    date_matches = df[df["date"] == target_date]
    has_actual = False
    actual_results = {}

    if not date_matches.empty:
        target_idx = date_matches.index[0]
        actual_row = df.iloc[target_idx].to_dict()
        actual_results = {
            "DSWR": str(actual_row.get("DSWR", "XX")).strip(),
            "FRBD": str(actual_row.get("FRBD", "XX")).strip(),
            "GBD": str(actual_row.get("GBD", "XX")).strip(),
            "GALI": str(actual_row.get("GALI", "XX")).strip(),
        }
        has_actual = any(v not in ["XX", ""] for v in actual_results.values())

    latest_date = df.iloc[-1]["date"]
    predictions = {}
    verdicts = {}

    # If target_date is today / latest date, guarantee 100% exact alignment with predict_all
    if target_date == latest_date:
        global cached_predictions
        if cached_predictions is None:
            refresh_predictions()
        predictions = cached_predictions
    else:
        for game in ["DSWR", "FRBD", "GBD", "GALI"]:
            if not date_matches.empty:
                df_eval = df.iloc[:target_idx+1].copy()
                # Follow real-world sequential order within the day:
                if game == "DSWR":
                    df_eval.loc[target_idx, ["DSWR", "FRBD", "GBD", "GALI"]] = ["XX", "XX", "XX", "XX"]
                elif game == "FRBD":
                    df_eval.loc[target_idx, ["FRBD", "GBD", "GALI"]] = ["XX", "XX", "XX"]
                elif game == "GBD":
                    df_eval.loc[target_idx, ["GBD", "GALI"]] = ["XX", "XX"]
                elif game == "GALI":
                    df_eval.loc[target_idx, ["GALI"]] = ["XX"]
            else:
                df_eval = df.copy()
                df_eval = pd.concat([df_eval, pd.DataFrame([{
                    "date": target_date, "DSWR": "XX", "FRBD": "XX", "GBD": "XX", "GALI": "XX"
                }])], ignore_index=True)

            pkl_path = os.path.join(MODELS_DIR, f"{game}_predictor.pkl")
            if os.path.exists(pkl_path):
                p_saved = joblib.load(pkl_path)
                eval_pred = GamePredictor(game, df_eval)
                eval_pred.model_andar_rf = p_saved.model_andar_rf
                eval_pred.model_bahar_rf = p_saved.model_bahar_rf
                eval_pred.model_andar_hgb = p_saved.model_andar_hgb
                eval_pred.model_bahar_hgb = p_saved.model_bahar_hgb
                eval_pred.feature_names = p_saved.feature_names
            else:
                predictor = ai_manager.get_predictor(game, df)
                eval_pred = GamePredictor(game, df_eval)
                eval_pred.model_andar_rf = predictor.model_andar_rf
                eval_pred.model_bahar_rf = predictor.model_bahar_rf
                eval_pred.model_andar_hgb = predictor.model_andar_hgb
                eval_pred.model_bahar_hgb = predictor.model_bahar_hgb
                eval_pred.feature_names = predictor.feature_names

            res = eval_pred.predict_next()
            predictions[game] = res

    # Compute verdicts against actual results
    for game in ["DSWR", "FRBD", "GBD", "GALI"]:
        res = predictions[game]
        if has_actual and actual_results.get(game) not in ["XX", None, ""]:
            actual_val = actual_results[game]
            try:
                act_num = int(actual_val)
                actual_t = act_num // 10
                actual_o = act_num % 10
                top5 = [j["jodi"] for j in res["top_primary_jodis"]]
                crossing = res["crossing_digits"]

                is_exact = (actual_val == res["top_primary_jodis"][0]["jodi"])
                is_top5 = (actual_val in top5)
                is_crossing = (str(actual_t) in crossing and str(actual_o) in crossing)
                andar_hit = (actual_t == res["solid_andar"]["primary"])
                bahar_hit = (actual_o == res["solid_bahar"]["primary"])

                verdicts[game] = {
                    "actual": actual_val,
                    "is_exact_hit": is_exact,
                    "is_top5_hit": is_top5,
                    "is_crossing_hit": is_crossing,
                    "andar_hit": andar_hit,
                    "bahar_hit": bahar_hit,
                }
            except Exception:
                verdicts[game] = None
        else:
            verdicts[game] = None

    return jsonify({
        "status": "success",
        "target_date": target_date,
        "has_actual": has_actual,
        "actual_results": actual_results,
        "predictions": predictions,
        "verdicts": verdicts
    })

@app.route("/api/predict/<game>", methods=["GET"])
def api_predict_game(game):
    game = game.upper()
    if game not in ["DSWR", "FRBD", "GBD", "GALI"]:
        return jsonify({"status": "error", "message": "Invalid game"}), 400
    df = get_df()
    predictor = ai_manager.get_predictor(game, df)
    pred = predictor.predict_next()
    backtest = predictor.evaluate_backtest(test_draws=30)
    return jsonify({
        "status": "success",
        "game": game,
        "prediction": pred,
        "backtest": backtest
    })

@app.route("/api/relations", methods=["GET"])
def api_relations():
    df = get_df()
    analyzer = RelationAnalyzer(df)
    correlations = analyzer.compute_game_correlations()
    transfers = analyzer.compute_haruf_transfer_rates()
    deltas = {
        "DSWR_to_FRBD": analyzer.compute_digit_difference_distribution("DSWR", "FRBD"),
        "FRBD_to_GBD": analyzer.compute_digit_difference_distribution("FRBD", "GBD"),
        "GBD_to_GALI": analyzer.compute_digit_difference_distribution("GBD", "GALI"),
    }
    return jsonify({
        "status": "success",
        "correlations": correlations,
        "transfer_rates": transfers,
        "digit_deltas": deltas
    })

@app.route("/api/hotcold/<game>", methods=["GET"])
def api_hotcold(game):
    game = game.upper()
    if game not in ["DSWR", "FRBD", "GBD", "GALI"]:
        game = "FRBD"
    df = get_df()
    analyzer = RelationAnalyzer(df)
    stats = analyzer.compute_hot_cold_frequencies(game, last_n=100)
    return jsonify({"status": "success", "stats": stats})

@app.route("/api/sync", methods=["POST"])
def api_sync():
    """1-Click Sync from satta-king-fast.com"""
    global cached_dataset
    try:
        new_df = sync_latest_today(min_interval=0)
        cached_dataset = new_df
        predictions, backtests = refresh_predictions()
        return jsonify({
            "status": "success",
            "message": "Results synchronized with satta-king-fast.com and models refreshed!",
            "total_days": len(new_df),
            "predictions": predictions,
            "backtests": backtests
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/history", methods=["GET"])
def api_history():
    limit = int(request.args.get("limit", 60))
    df = ensure_fresh_data()
    records = df.tail(limit).to_dict(orient="records")
    records.reverse()
    return jsonify({
        "status": "success",
        "total": len(df),
        "history": records
    })

if __name__ == "__main__":
    print("Starting Satta King AI Prediction & Relation Application...")
    print("Pre-training AI models on historical dataset...")
    refresh_predictions()
    print("AI models ready! Server running on http://127.0.0.1:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
