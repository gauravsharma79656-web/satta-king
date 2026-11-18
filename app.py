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
cached_dataset_sig = None
cached_predictions = None
cached_backtests = None

def _is_declared(val):
    if val is None:
        return False
    s = str(val).strip().upper()
    return s not in ["XX", "", "NONE", "NAN", "--", "-"]

def _dataset_sig(df):
    if df is None or df.empty:
        return ""
    last = df.iloc[-1]
    return f"{len(df)}|{last.get('date')}|{last.get('DSWR')}|{last.get('FRBD')}|{last.get('GBD')}|{last.get('GALI')}"

def _run_game_prediction(game, full_df, df_eval):
    """Runs the exact untouched GamePredictor.predict_next() logic on df_eval."""
    predictor = ai_manager.get_predictor(game, full_df)
    eval_pred = GamePredictor(game, df_eval)
    eval_pred.model_andar_rf = predictor.model_andar_rf
    eval_pred.model_bahar_rf = predictor.model_bahar_rf
    eval_pred.model_andar_hgb = predictor.model_andar_hgb
    eval_pred.model_bahar_hgb = predictor.model_bahar_hgb
    eval_pred.feature_names = predictor.feature_names
    return eval_pred.predict_next()

def _make_locked_prediction(game, pred_date, waiting_for, target_date):
    if game == "DSWR":
        msg = f"Next-day DSWR ({pred_date}) prediction will generate once {target_date}'s actual GALI result is declared."
        status = f"Waiting for GALI ({pred_date})"
    else:
        msg = f"{game} prediction will generate once {target_date}'s actual {waiting_for} result is declared."
        status = f"Waiting for {waiting_for}"

    return {
        "game": game,
        "prediction_for_date": pred_date,
        "current_status": status,
        "is_locked": True,
        "waiting_for": waiting_for,
        "waiting_message": msg,
        "top_primary_jodis": [],
        "top_support_jodis": [],
        "solid_andar": None,
        "solid_bahar": None,
        "crossing_digits": [],
        "reasoning": msg
    }

def build_sequential_predictions(df, target_date):
    """
    Sequential prediction pipeline (without changing any prediction math):
      1. FRBD (target_date) -> Always generates first
      2. GBD  (target_date) -> Generates only after FRBD actual result comes
      3. GALI (target_date) -> Generates only after GBD actual result comes
      4. DSWR (next_date)   -> Generates for NEXT DAY only after GALI actual result comes
    """
    try:
        dt_obj = datetime.datetime.strptime(target_date, "%Y-%m-%d")
        next_date = (dt_obj + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    except Exception:
        next_date = "Next Day"

    date_matches = df[df["date"] == target_date]
    if not date_matches.empty:
        target_idx = date_matches.index[0]
        actual_row = df.iloc[target_idx].to_dict()
        frbd_actual = str(actual_row.get("FRBD", "XX")).strip()
        gbd_actual = str(actual_row.get("GBD", "XX")).strip()
        gali_actual = str(actual_row.get("GALI", "XX")).strip()
        df_base = df.iloc[:target_idx + 1].copy()
    else:
        frbd_actual = "XX"
        gbd_actual = "XX"
        gali_actual = "XX"
        df_base = pd.concat([
            df.copy(),
            pd.DataFrame([{
                "date": target_date, "DSWR": "XX", "FRBD": "XX", "GBD": "XX", "GALI": "XX"
            }])
        ], ignore_index=True)
        target_idx = len(df_base) - 1

    # Check next day's DSWR actual result (for verification/status)
    next_matches = df[df["date"] == next_date]
    if not next_matches.empty:
        next_dswr_actual = str(df.iloc[next_matches.index[0]].get("DSWR", "XX")).strip()
    else:
        next_dswr_actual = "XX"

    actual_results = {
        "FRBD": frbd_actual,
        "GBD": gbd_actual,
        "GALI": gali_actual,
        "DSWR": next_dswr_actual,
    }
    has_actual = any(_is_declared(v) for v in actual_results.values())

    predictions = {}

    # 1. FRBD (target_date) - Always unlocked first
    df_frbd = df_base.copy()
    df_frbd.loc[target_idx, ["FRBD", "GBD", "GALI"]] = ["XX", "XX", "XX"]
    res_frbd = _run_game_prediction("FRBD", df, df_frbd)
    res_frbd["prediction_for_date"] = target_date
    res_frbd["is_locked"] = False
    res_frbd["current_status"] = f"Result Out: {frbd_actual}" if _is_declared(frbd_actual) else "Draw Pending"
    predictions["FRBD"] = res_frbd

    # 2. GBD (target_date) - Unlocks only after FRBD actual result comes
    if _is_declared(frbd_actual):
        df_gbd = df_base.copy()
        df_gbd.loc[target_idx, ["GBD", "GALI"]] = ["XX", "XX"]
        res_gbd = _run_game_prediction("GBD", df, df_gbd)
        res_gbd["prediction_for_date"] = target_date
        res_gbd["is_locked"] = False
        res_gbd["current_status"] = f"Result Out: {gbd_actual}" if _is_declared(gbd_actual) else "Draw Pending"
        predictions["GBD"] = res_gbd
    else:
        predictions["GBD"] = _make_locked_prediction("GBD", target_date, "FRBD", target_date)

    # 3. GALI (target_date) - Unlocks only after GBD actual result comes
    if _is_declared(gbd_actual):
        df_gali = df_base.copy()
        df_gali.loc[target_idx, ["GALI"]] = ["XX"]
        res_gali = _run_game_prediction("GALI", df, df_gali)
        res_gali["prediction_for_date"] = target_date
        res_gali["is_locked"] = False
        res_gali["current_status"] = f"Result Out: {gali_actual}" if _is_declared(gali_actual) else "Draw Pending"
        predictions["GALI"] = res_gali
    else:
        predictions["GALI"] = _make_locked_prediction("GALI", target_date, "GBD", target_date)

    # 4. DSWR (next_date, 5:00 AM) - Unlocks only after GALI actual result comes
    if _is_declared(gali_actual):
        df_dswr_next = pd.concat([
            df_base.copy(),
            pd.DataFrame([{
                "date": next_date, "DSWR": "XX", "FRBD": "XX", "GBD": "XX", "GALI": "XX"
            }])
        ], ignore_index=True)
        res_dswr = _run_game_prediction("DSWR", df, df_dswr_next)
        res_dswr["prediction_for_date"] = next_date
        res_dswr["is_locked"] = False
        res_dswr["current_status"] = (
            f"Result Out: {next_dswr_actual}" if _is_declared(next_dswr_actual) else f"Next Day ({next_date})"
        )
        predictions["DSWR"] = res_dswr
    else:
        predictions["DSWR"] = _make_locked_prediction("DSWR", next_date, "GALI", target_date)

    return predictions, actual_results, has_actual, next_date

def get_df(force_reload=False):
    global cached_dataset, cached_dataset_sig
    if cached_dataset is None or force_reload:
        cached_dataset = get_latest_dataset()
        cached_dataset_sig = _dataset_sig(cached_dataset)
    return cached_dataset

def ensure_fresh_data():
    global cached_dataset, cached_dataset_sig, cached_predictions
    try:
        new_df = sync_latest_today(min_interval=15)
        new_sig = _dataset_sig(new_df)
        if new_sig != cached_dataset_sig:
            cached_dataset = new_df
            cached_dataset_sig = new_sig
            cached_predictions = None  # Invalidate cached predictions when new live result arrives
        return cached_dataset
    except Exception as e:
        print("ensure_fresh_data error:", e)
        return get_df()

def refresh_predictions():
    global cached_predictions, cached_backtests
    df = get_df()
    latest_date = df.iloc[-1]["date"]
    cached_predictions, _, _, _ = build_sequential_predictions(df, latest_date)
    if cached_backtests is None:
        backtests = {}
        for game in ["DSWR", "FRBD", "GBD", "GALI"]:
            p = ai_manager.get_predictor(game, df)
            backtests[game] = p.evaluate_backtest(test_draws=30)
        cached_backtests = backtests
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

    predictions, actual_results, has_actual, next_date = build_sequential_predictions(df, target_date)
    verdicts = {}

    # Compute verdicts against actual results for unlocked games
    for game in ["FRBD", "GBD", "GALI", "DSWR"]:
        res = predictions[game]
        if (
            not res.get("is_locked")
            and _is_declared(actual_results.get(game))
            and res.get("top_primary_jodis")
        ):
            actual_val = actual_results[game]
            try:
                act_num = int(actual_val)
                actual_t = act_num // 10
                actual_o = act_num % 10
                top5 = [j["jodi"] for j in res["top_primary_jodis"]]
                crossing = res["crossing_digits"]

                is_exact = (f"{act_num:02d}" == res["top_primary_jodis"][0]["jodi"])
                is_top5 = (f"{act_num:02d}" in top5)
                is_crossing = (str(actual_t) in crossing and str(actual_o) in crossing)
                andar_hit = (actual_t == res["solid_andar"]["primary"])
                bahar_hit = (actual_o == res["solid_bahar"]["primary"])

                verdicts[game] = {
                    "actual": f"{act_num:02d}",
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
        "next_date": next_date,
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
    global cached_dataset, cached_dataset_sig, cached_predictions
    try:
        new_df = sync_latest_today(min_interval=0)
        cached_dataset = new_df
        cached_dataset_sig = _dataset_sig(new_df)
        cached_predictions = None
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
