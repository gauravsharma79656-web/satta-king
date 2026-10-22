import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import joblib
import os
from relation_analyzer import RASHI_MAP, RelationAnalyzer

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)

class GamePredictor:
    def __init__(self, target_game, df):
        self.target_game = target_game
        self.df = df.copy()
        self.analyzer = RelationAnalyzer(df)
        self.model_andar_rf = None
        self.model_bahar_rf = None
        self.model_andar_hgb = None
        self.model_bahar_hgb = None
        self.cached_backtest = None
        self.feature_names = []
        self._prepare_dataset()

    def _prepare_dataset(self):
        for col in ["DSWR", "FRBD", "GBD", "GALI"]:
            self.df[col + "_num"] = pd.to_numeric(self.df[col], errors="coerce").fillna(-1).astype(int)
            self.df[col + "_tens"] = self.df[col + "_num"].apply(lambda v: v // 10 if v >= 0 else -1)
            self.df[col + "_ones"] = self.df[col + "_num"].apply(lambda v: v % 10 if v >= 0 else -1)
            self.df[col + "_sum"] = self.df[col + "_num"].apply(lambda v: (v // 10 + v % 10) % 10 if v >= 0 else -1)
            self.df[col + "_rashi_t"] = self.df[col + "_tens"].apply(lambda v: RASHI_MAP.get(v, -1) if v >= 0 else -1)
            self.df[col + "_rashi_o"] = self.df[col + "_ones"].apply(lambda v: RASHI_MAP.get(v, -1) if v >= 0 else -1)
            self.df[col + "_jora"] = self.df[col + "_num"].apply(lambda v: int((v // 10) == (v % 10)) if v >= 0 else -1)

        self.df["dt"] = pd.to_datetime(self.df["date"], errors="coerce")
        self.df["dayofweek"] = self.df["dt"].dt.dayofweek.fillna(0).astype(int)
        self.df["day"] = self.df["dt"].dt.day.fillna(1).astype(int)
        self.df["month"] = self.df["dt"].dt.month.fillna(1).astype(int)

        # Precompute numpy arrays for nanosecond access
        self.N = len(self.df)
        self.dow_arr = self.df["dayofweek"].values
        self.day_arr = self.df["day"].values
        self.month_arr = self.df["month"].values

        self.game_data = {}
        for g in ["DSWR", "FRBD", "GBD", "GALI"]:
            self.game_data[g] = {
                "num": self.df[g + "_num"].values,
                "tens": self.df[g + "_tens"].values,
                "ones": self.df[g + "_ones"].values,
                "sum": self.df[g + "_sum"].values,
                "rashi_t": self.df[g + "_rashi_t"].values,
                "rashi_o": self.df[g + "_rashi_o"].values,
                "jora": self.df[g + "_jora"].values,
            }

        # Precompute rolling 14-day counts for target game tens and ones
        t_arr = self.game_data[self.target_game]["tens"]
        o_arr = self.game_data[self.target_game]["ones"]
        self.roll_tens = np.zeros((self.N, 10), dtype=np.int32)
        self.roll_ones = np.zeros((self.N, 10), dtype=np.int32)

        for i in range(1, self.N):
            start = max(0, i - 14)
            for d in range(10):
                self.roll_tens[i, d] = np.count_nonzero(t_arr[start:i] == d)
                self.roll_ones[i, d] = np.count_nonzero(o_arr[start:i] == d)

    def _extract_features_for_row(self, idx):
        feats = {
            "dayofweek": self.dow_arr[idx],
            "day": self.day_arr[idx],
            "month": self.month_arr[idx]
        }

        # Preceding games on same day
        same_day_order = {
            "DSWR": [],
            "FRBD": ["DSWR"],
            "GBD": ["DSWR", "FRBD"],
            "GALI": ["DSWR", "FRBD", "GBD"]
        }

        for pre in same_day_order[self.target_game]:
            g_dict = self.game_data[pre]
            feats[f"same_day_{pre}_num"] = g_dict["num"][idx]
            feats[f"same_day_{pre}_tens"] = g_dict["tens"][idx]
            feats[f"same_day_{pre}_ones"] = g_dict["ones"][idx]
            feats[f"same_day_{pre}_rashi_t"] = g_dict["rashi_t"][idx]
            feats[f"same_day_{pre}_rashi_o"] = g_dict["rashi_o"][idx]
            feats[f"same_day_{pre}_sum"] = g_dict["sum"][idx]
            feats[f"same_day_{pre}_jora"] = g_dict["jora"][idx]

        # Lag 1 (yesterday)
        prev_idx = idx - 1 if idx > 0 else 0
        for g in ["DSWR", "FRBD", "GBD", "GALI"]:
            g_dict = self.game_data[g]
            feats[f"lag1_{g}_num"] = g_dict["num"][prev_idx] if idx > 0 else -1
            feats[f"lag1_{g}_tens"] = g_dict["tens"][prev_idx] if idx > 0 else -1
            feats[f"lag1_{g}_ones"] = g_dict["ones"][prev_idx] if idx > 0 else -1
            feats[f"lag1_{g}_rashi_t"] = g_dict["rashi_t"][prev_idx] if idx > 0 else -1
            feats[f"lag1_{g}_rashi_o"] = g_dict["rashi_o"][prev_idx] if idx > 0 else -1
            feats[f"lag1_{g}_jora"] = g_dict["jora"][prev_idx] if idx > 0 else -1

        # Lag 2 & 3
        for lag in [2, 3]:
            l_idx = idx - lag if idx >= lag else -1
            tg_dict = self.game_data[self.target_game]
            feats[f"lag{lag}_{self.target_game}_num"] = tg_dict["num"][l_idx] if l_idx >= 0 else -1
            feats[f"lag{lag}_{self.target_game}_tens"] = tg_dict["tens"][l_idx] if l_idx >= 0 else -1
            feats[f"lag{lag}_{self.target_game}_ones"] = tg_dict["ones"][l_idx] if l_idx >= 0 else -1

        # Rolling 14-day counts
        for d in range(10):
            feats[f"roll14_tens_{d}"] = self.roll_tens[idx, d]
            feats[f"roll14_ones_{d}"] = self.roll_ones[idx, d]

        return feats

    def train(self):
        t_arr = self.game_data[self.target_game]["num"]
        valid_indices = np.where(t_arr >= 0)[0]

        if len(valid_indices) < 50:
            raise ValueError(f"Not enough data to train for {self.target_game}")

        X_list = [self._extract_features_for_row(i) for i in valid_indices]
        X_df = pd.DataFrame(X_list)
        self.feature_names = list(X_df.columns)
        X = X_df.values
        y_andar = self.game_data[self.target_game]["tens"][valid_indices]
        y_bahar = self.game_data[self.target_game]["ones"][valid_indices]

        # 1. Random Forest Classifier
        self.model_andar_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=9,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        self.model_andar_rf.fit(X, y_andar)

        self.model_bahar_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=9,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        self.model_bahar_rf.fit(X, y_bahar)

        # 2. HistGradientBoostingClassifier
        self.model_andar_hgb = HistGradientBoostingClassifier(
            max_iter=80,
            max_depth=6,
            learning_rate=0.08,
            random_state=42
        )
        self.model_andar_hgb.fit(X, y_andar)

        self.model_bahar_hgb = HistGradientBoostingClassifier(
            max_iter=80,
            max_depth=6,
            learning_rate=0.08,
            random_state=42
        )
        self.model_bahar_hgb.fit(X, y_bahar)

        return len(X)

    def evaluate_backtest(self, test_draws=30):
        if self.cached_backtest is not None:
            return self.cached_backtest

        t_arr = self.game_data[self.target_game]["num"]
        valid_indices = np.where(t_arr >= 0)[0]
        if len(valid_indices) < test_draws + 20:
            test_draws = max(10, len(valid_indices) // 4)

        test_idx = valid_indices[-test_draws:]
        train_idx = valid_indices[:-test_draws]

        X_train = np.array([list(self._extract_features_for_row(i).values()) for i in train_idx])
        y_train_a = self.game_data[self.target_game]["tens"][train_idx]
        y_train_b = self.game_data[self.target_game]["ones"][train_idx]

        m_a = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=-1)
        m_b = RandomForestClassifier(n_estimators=40, max_depth=6, random_state=42, n_jobs=-1)
        m_a.fit(X_train, y_train_a)
        m_b.fit(X_train, y_train_b)

        top10_jodi_hits = 0
        top20_jodi_hits = 0
        andar_hits = 0
        bahar_hits = 0
        any_haruf_hits = 0

        for i in test_idx:
            actual_num = self.game_data[self.target_game]["num"][i]
            actual_a = self.game_data[self.target_game]["tens"][i]
            actual_b = self.game_data[self.target_game]["ones"][i]

            row_feats = np.array([list(self._extract_features_for_row(i).values())])
            probs_a = m_a.predict_proba(row_feats)[0]
            probs_b = m_b.predict_proba(row_feats)[0]

            full_pa = np.zeros(10)
            full_pb = np.zeros(10)
            for c_idx, c in enumerate(m_a.classes_):
                full_pa[c] = probs_a[c_idx]
            for c_idx, c in enumerate(m_b.classes_):
                full_pb[c] = probs_b[c_idx]

            pred_top2_a = np.argsort(full_pa)[-2:]
            pred_top2_b = np.argsort(full_pb)[-2:]

            if actual_a in pred_top2_a:
                andar_hits += 1
            if actual_b in pred_top2_b:
                bahar_hits += 1
            if actual_a in pred_top2_a or actual_b in pred_top2_b:
                any_haruf_hits += 1

            jodi_probs = np.outer(full_pa, full_pb).flatten()
            top10_jodis = np.argsort(jodi_probs)[-10:]
            top20_jodis = np.argsort(jodi_probs)[-20:]

            if actual_num in top10_jodis:
                top10_jodi_hits += 1
            if actual_num in top20_jodis:
                top20_jodi_hits += 1

        total = len(test_idx)
        self.cached_backtest = {
            "test_draws": total,
            "top10_jodi_accuracy_pct": round((top10_jodi_hits / total) * 100, 1),
            "top20_jodi_accuracy_pct": round((top20_jodi_hits / total) * 100, 1),
            "andar_top2_accuracy_pct": round((andar_hits / total) * 100, 1),
            "bahar_top2_accuracy_pct": round((bahar_hits / total) * 100, 1),
            "haruf_coverage_pct": round((any_haruf_hits / total) * 100, 1)
        }
        return self.cached_backtest

    def predict_next(self):
        if not self.model_andar_rf or not self.model_andar_hgb:
            self.train()

        last_idx = len(self.df) - 1
        target_val = self.df.iloc[last_idx][f"{self.target_game}"]

        if str(target_val).upper() != "XX":
            pred_date = "Next Draw"
            idx_for_feats = last_idx
        else:
            pred_date = self.df.iloc[last_idx]["date"]
            idx_for_feats = last_idx

        feats = self._extract_features_for_row(idx_for_feats)
        feats_df = pd.DataFrame([feats])[self.feature_names]

        # Ensemble soft probability voting
        rf_pa = self.model_andar_rf.predict_proba(feats_df.values)[0]
        rf_pb = self.model_bahar_rf.predict_proba(feats_df.values)[0]
        hgb_pa = self.model_andar_hgb.predict_proba(feats_df.values)[0]
        hgb_pb = self.model_bahar_hgb.predict_proba(feats_df.values)[0]

        full_pa = np.zeros(10)
        full_pb = np.zeros(10)

        for c_idx, c in enumerate(self.model_andar_rf.classes_):
            full_pa[c] += 0.45 * rf_pa[c_idx]
        for c_idx, c in enumerate(self.model_andar_hgb.classes_):
            full_pa[c] += 0.55 * hgb_pa[c_idx]

        for c_idx, c in enumerate(self.model_bahar_rf.classes_):
            full_pb[c] += 0.45 * rf_pb[c_idx]
        for c_idx, c in enumerate(self.model_bahar_hgb.classes_):
            full_pb[c] += 0.55 * hgb_pb[c_idx]

        full_pa = full_pa / full_pa.sum()
        full_pb = full_pb / full_pb.sum()

        # Cross-Game Haruf & Rashi relation
        same_day_mapping = {"FRBD": "DSWR", "GBD": "FRBD", "GALI": "GBD", "DSWR": "GALI"}
        ref_game = same_day_mapping[self.target_game]

        ref_idx = last_idx if self.target_game != "DSWR" else max(0, last_idx - 1)
        ref_val = self.game_data[ref_game]["num"][ref_idx]
        ref_desc = ""

        cross_game_harufs = set()
        cross_game_rashi = set()

        if ref_val >= 0:
            ref_num = int(ref_val)
            t_ref = ref_num // 10
            o_ref = ref_num % 10
            cross_game_harufs = {t_ref, o_ref}
            cross_game_rashi = {RASHI_MAP[t_ref], RASHI_MAP[o_ref]}
            ref_desc = f"Direct relation with {ref_game} ({ref_num:02d}): Harufs [{t_ref}, {o_ref}], Rashis [{RASHI_MAP[t_ref]}, {RASHI_MAP[o_ref]}]."

        # Yesterday's same game relation
        prev_target_val = self.game_data[self.target_game]["num"][max(0, idx_for_feats - 1)]
        if prev_target_val >= 0:
            pt_num = int(prev_target_val)
            cross_game_harufs.add(pt_num // 10)
            cross_game_harufs.add(pt_num % 10)
            cross_game_rashi.add(RASHI_MAP[pt_num // 10])
            cross_game_rashi.add(RASHI_MAP[pt_num % 10])

        jodi_matrix = np.outer(full_pa, full_pb)

        for i in range(10):
            for j in range(10):
                boost = 1.0
                if i in cross_game_harufs or j in cross_game_harufs:
                    boost += 0.30
                if i in cross_game_rashi or j in cross_game_rashi:
                    boost += 0.20
                jodi_matrix[i, j] *= boost

        jodi_scores = jodi_matrix.flatten()
        ranked_indices = np.argsort(jodi_scores)[::-1]

        top_primary_jodis = []
        for rank, idx in enumerate(ranked_indices[:5]):
            jodi_str = f"{idx:02d}"
            prob = float(jodi_scores[idx] / jodi_scores.sum())
            top_primary_jodis.append({
                "rank": rank + 1,
                "jodi": jodi_str,
                "confidence_pct": round(prob * 100, 2),
                "rashi": f"{RASHI_MAP[idx // 10]}{RASHI_MAP[idx % 10]}"
            })

        top_support_jodis = []
        for rank, idx in enumerate(ranked_indices[5:10]):
            jodi_str = f"{idx:02d}"
            prob = float(jodi_scores[idx] / jodi_scores.sum())
            top_support_jodis.append({
                "rank": rank + 6,
                "jodi": jodi_str,
                "confidence_pct": round(prob * 100, 2),
                "rashi": f"{RASHI_MAP[idx // 10]}{RASHI_MAP[idx % 10]}"
            })

        sorted_andar = np.argsort(full_pa)[::-1]
        sorted_bahar = np.argsort(full_pb)[::-1]

        solid_andar = {
            "primary": int(sorted_andar[0]),
            "primary_prob": round(float(full_pa[sorted_andar[0]]) * 100, 1),
            "secondary": int(sorted_andar[1]),
            "secondary_prob": round(float(full_pa[sorted_andar[1]]) * 100, 1),
            "rashi_primary": RASHI_MAP[int(sorted_andar[0])]
        }

        solid_bahar = {
            "primary": int(sorted_bahar[0]),
            "primary_prob": round(float(full_pb[sorted_bahar[0]]) * 100, 1),
            "secondary": int(sorted_bahar[1]),
            "secondary_prob": round(float(full_pb[sorted_bahar[1]]) * 100, 1),
            "rashi_primary": RASHI_MAP[int(sorted_bahar[0])]
        }

        # Exactly 6-Digit Crossing Selection
        raw_pool = [
            solid_andar["primary"], solid_bahar["primary"],
            solid_andar["secondary"], solid_bahar["secondary"],
            solid_andar["rashi_primary"], solid_bahar["rashi_primary"],
            RASHI_MAP.get(solid_andar["secondary"], -1),
            RASHI_MAP.get(solid_bahar["secondary"], -1)
        ]
        crossing_pool = []
        for d in raw_pool:
            if d >= 0 and d not in crossing_pool:
                crossing_pool.append(d)

        # Fill up to exactly 6 digits using top probability distributions
        for cand in list(sorted_andar) + list(sorted_bahar):
            if len(crossing_pool) >= 6:
                break
            if cand not in crossing_pool:
                crossing_pool.append(cand)

        crossing_digits = [str(d) for d in crossing_pool[:6]]

        reasoning = (
            f"Dual-Ensemble (Random Forest + HistGradientBoosting) trained on {len(self.df)} draws. {ref_desc} "
            f"Andar Haruf '{solid_andar['primary']}' ({solid_andar['primary_prob']}%), Bahar Haruf '{solid_bahar['primary']}' ({solid_bahar['primary_prob']}%). "
            f"Top synthesis pair is '{top_primary_jodis[0]['jodi']}'."
        )

        return {
            "game": self.target_game,
            "prediction_for_date": pred_date,
            "current_status": "Draw Pending" if str(target_val).upper() == "XX" else f"Result Out: {target_val}",
            "top_primary_jodis": top_primary_jodis,
            "top_support_jodis": top_support_jodis,
            "solid_andar": solid_andar,
            "solid_bahar": solid_bahar,
            "crossing_digits": [str(d) for d in crossing_digits],
            "reasoning": reasoning
        }

class AIModelManager:
    def __init__(self):
        self.predictors = {}

    def get_predictor(self, game, df):
        if game not in self.predictors:
            pkl_path = os.path.join(MODELS_DIR, f"{game}_predictor.pkl")
            if os.path.exists(pkl_path):
                try:
                    p = joblib.load(pkl_path)
                    # Update df reference in case dataset changed
                    p.df = df.copy()
                    p._prepare_dataset()
                    self.predictors[game] = p
                except Exception as e:
                    print(f"Error loading {pkl_path}, retraining in-memory: {e}")
                    p = GamePredictor(game, df)
                    p.train()
                    self.predictors[game] = p
            else:
                p = GamePredictor(game, df)
                p.train()
                self.predictors[game] = p
        return self.predictors[game]

    def predict_all(self, df):
        results = {}
        backtests = {}
        for game in ["DSWR", "FRBD", "GBD", "GALI"]:
            p = self.get_predictor(game, df)
            results[game] = p.predict_next()
            backtests[game] = p.evaluate_backtest(test_draws=30)
        return results, backtests

if __name__ == "__main__":
    from data_manager import get_latest_dataset
    df = get_latest_dataset()
    print(f"Dataset shape: {df.shape}")
    manager = AIModelManager()
    print("Training dual-ensemble AI models on full dataset...")
    preds, backtests = manager.predict_all(df)
    for g, pr in preds.items():
        print(f"\n--- {g} ---")
        print("Top 5 Jodis:", [j["jodi"] for j in pr["top_primary_jodis"]])
        print("Solid Andar:", pr["solid_andar"])
        print("Solid Bahar:", pr["solid_bahar"])
        print("Crossing:", pr["crossing_digits"])
        print("Backtest metrics:", backtests[g])
