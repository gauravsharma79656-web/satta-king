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
        all_cand_games = ["DSWR", "DLBZ", "SRGN", "FRBD", "GBD", "GALI"]
        active_games = [c for c in all_cand_games if c in self.df.columns]
        for col in active_games:
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
        for g in active_games:
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

        # Preceding games on same day (Preserving exact existing logic for DSWR, FRBD, GBD, GALI)
        same_day_order = {
            "DSWR": [],
            "DLBZ": ["DSWR"],
            "SRGN": ["DLBZ", "DSWR"],
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
            n_estimators=50,
            max_depth=3,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        self.model_andar_rf.fit(X, y_andar)

        self.model_bahar_rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )
        self.model_bahar_rf.fit(X, y_bahar)

        # 2. HistGradientBoostingClassifier
        self.model_andar_hgb = HistGradientBoostingClassifier(
            max_iter=50,
            max_depth=3,
            learning_rate=0.05,
            random_state=42
        )
        self.model_andar_hgb.fit(X, y_andar)

        self.model_bahar_hgb = HistGradientBoostingClassifier(
            max_iter=50,
            max_depth=3,
            learning_rate=0.05,
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
        today_date_str = str(self.df.iloc[last_idx]["date"]).strip()

        is_already_declared = (str(target_val).strip().upper() not in ["XX", ""])

        if is_already_declared:
            # If today's result is already declared (e.g. morning DSWR has opened),
            # the upcoming draw is for the NEXT DAY.
            next_date = (pd.to_datetime(today_date_str) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            virtual_row = {c: "XX" for c in self.df.columns}
            virtual_row["date"] = next_date

            # For games on today's row that are still XX, forward-fill with latest declared for feature extraction
            today_row = self.df.iloc[-1].copy()
            for col in [c for c in ["DSWR", "DLBZ", "SRGN", "FRBD", "GBD", "GALI"] if c in self.df.columns]:
                if str(today_row[col]).strip().upper() in ["XX", ""]:
                    valid_s = self.df[col].loc[~self.df[col].astype(str).str.strip().str.upper().isin(["XX", ""])]
                    if not valid_s.empty:
                        today_row[col] = valid_s.iloc[-1]

            df_copy = self.df.copy()
            df_copy.iloc[-1] = today_row
            df_extended = pd.concat([df_copy, pd.DataFrame([virtual_row])], ignore_index=True)

            eval_p = GamePredictor(self.target_game, df_extended)
            eval_p.model_andar_rf = self.model_andar_rf
            eval_p.model_bahar_rf = self.model_bahar_rf
            eval_p.model_andar_hgb = self.model_andar_hgb
            eval_p.model_bahar_hgb = self.model_bahar_hgb
            eval_p.feature_names = self.feature_names

            status_str = f"Next Draw ({pd.to_datetime(next_date).strftime('%b %d')})"
            return eval_p._compute_prediction(
                idx=len(df_extended) - 1,
                pred_date=next_date,
                status_label=status_str,
                today_declared=str(target_val),
                is_next_day=True
            )
        else:
            return self._compute_prediction(
                idx=last_idx,
                pred_date=today_date_str,
                status_label="Draw Pending",
                today_declared=None,
                is_next_day=False
            )

    def _compute_prediction(self, idx, pred_date, status_label, today_declared=None, is_next_day=False):
        feats = self._extract_features_for_row(idx)
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

        # Multi-Predecessor Hierarchy & Mathematical Relation Umbrella
        # (Literal Harufs, Rashi Mirrors, Sum/Jod, and Difference)
        same_day_order = {
            "DSWR": [],
            "DLBZ": ["DSWR"],
            "SRGN": ["DLBZ", "DSWR"],
            "FRBD": ["DSWR"],
            "GBD": ["FRBD", "DSWR"],
            "GALI": ["GBD", "FRBD", "DSWR"]
        }

        digit_boost = np.zeros(10)
        cross_game_harufs = set()
        ref_desc_parts = []

        preds = same_day_order.get(self.target_game, [])

        # 1. Immediate Predecessor Game (DSWR for FRBD, FRBD for GBD, GBD for GALI)
        if len(preds) > 0:
            imm_game = preds[0]
            imm_idx = idx if not is_next_day else idx - 1
            imm_val = self.game_data[imm_game]["num"][imm_idx]
            if imm_val >= 0:
                v_int = int(imm_val)
                t_ref = v_int // 10
                o_ref = v_int % 10
                s_ref = (t_ref + o_ref) % 10
                diff_ref = abs(t_ref - o_ref)
                rt_ref = RASHI_MAP[t_ref]
                ro_ref = RASHI_MAP[o_ref]

                cross_game_harufs.update([t_ref, o_ref, s_ref, diff_ref, rt_ref, ro_ref])

                digit_boost[t_ref] += 0.20
                digit_boost[o_ref] += 0.20
                digit_boost[rt_ref] += 0.15
                digit_boost[ro_ref] += 0.15
                digit_boost[s_ref] += 0.15
                digit_boost[diff_ref] += 0.12

                ref_desc_parts.append(f"{imm_game} ({v_int:02d}): Harufs [{t_ref},{o_ref}], Rashis [{rt_ref},{ro_ref}], Sum {s_ref}, Diff {diff_ref}")
        elif self.target_game == "DSWR":
            # DSWR's immediate predecessor is GALI (drawn right before morning 5:00 AM)
            if not is_next_day:
                y_gali_idx = max(0, idx - 1)
                y_gali = self.game_data["GALI"]["num"][y_gali_idx]
            else:
                # For tomorrow morning's DSWR, predecessor is today's GALI or latest declared GALI
                today_gali = self.game_data["GALI"]["num"][idx - 1]
                if today_gali >= 0:
                    y_gali = today_gali
                else:
                    y_gali_idx = max(0, idx - 2) if idx >= 2 else max(0, idx - 1)
                    y_gali = self.game_data["GALI"]["num"][y_gali_idx]

            if y_gali >= 0:
                v_int = int(y_gali)
                t_ref = v_int // 10
                o_ref = v_int % 10
                s_ref = (t_ref + o_ref) % 10
                diff_ref = abs(t_ref - o_ref)
                rt_ref = RASHI_MAP[t_ref]
                ro_ref = RASHI_MAP[o_ref]

                cross_game_harufs.update([t_ref, o_ref, s_ref, diff_ref, rt_ref, ro_ref])
                digit_boost[t_ref] += 0.20
                digit_boost[o_ref] += 0.20
                digit_boost[rt_ref] += 0.15
                digit_boost[ro_ref] += 0.15
                digit_boost[s_ref] += 0.15
                digit_boost[diff_ref] += 0.12
                ref_desc_parts.append(f"GALI ({v_int:02d}): Harufs [{t_ref},{o_ref}], Sum {s_ref}")

        # 2. Earlier Same-Day Ancestor Games (morning DSWR for GBD/GALI, FRBD for GALI)
        for anc_game in preds[1:]:
            anc_idx = idx if not is_next_day else idx - 1
            anc_val = self.game_data[anc_game]["num"][anc_idx]
            if anc_val >= 0:
                v_int = int(anc_val)
                t_anc = v_int // 10
                o_anc = v_int % 10
                digit_boost[t_anc] += 0.10
                digit_boost[o_anc] += 0.10
                cross_game_harufs.update([t_anc, o_anc])
                ref_desc_parts.append(f"{anc_game} ({v_int:02d}) [{t_anc},{o_anc}]")

        # 3. Yesterday's Same-Game Cycle (24-hour repeat pattern)
        prev_target_val = self.game_data[self.target_game]["num"][max(0, idx - 1)]
        if prev_target_val >= 0:
            pt_num = int(prev_target_val)
            t_pt = pt_num // 10
            o_pt = pt_num % 10
            digit_boost[t_pt] += 0.10
            digit_boost[o_pt] += 0.10
            digit_boost[RASHI_MAP[t_pt]] += 0.06
            digit_boost[RASHI_MAP[o_pt]] += 0.06
            cross_game_harufs.update([t_pt, o_pt])
            cycle_label = "Declared DSWR" if is_next_day and self.target_game == "DSWR" else f"Yesterday {self.target_game}"
            ref_desc_parts.append(f"{cycle_label} ({pt_num:02d}) [{t_pt},{o_pt}]")

        # 4. Cross-Game 24h Transfer (Yesterday's GBD)
        if self.target_game in ["FRBD", "DSWR"]:
            y_gbd_idx = idx - 1
            y_gbd_val = self.game_data["GBD"]["num"][max(0, y_gbd_idx)]
            if y_gbd_val >= 0:
                yg_int = int(y_gbd_val)
                digit_boost[yg_int // 10] += 0.08
                digit_boost[yg_int % 10] += 0.08

        # Joint Probability Matrix Synthesis
        jodi_matrix = np.outer(full_pa, full_pb)
        for i in range(10):
            for j in range(10):
                jodi_matrix[i, j] *= (1.0 + digit_boost[i] + digit_boost[j])

        # Extract Ranked Jodis with Diversity Protection
        # (Prevents identical Bahar or Andar digits from monopolizing top spots)
        ranked_indices = np.argsort(jodi_matrix.flatten())[::-1]

        top_primary_jodis = []
        andar_counts = {}
        bahar_counts = {}
        used_indices = set()

        for idx_cand in ranked_indices:
            t, o = idx_cand // 10, idx_cand % 10
            if andar_counts.get(t, 0) >= 2 or bahar_counts.get(o, 0) >= 2:
                continue

            jodi_str = f"{t}{o}"
            top_primary_jodis.append({
                "rank": len(top_primary_jodis) + 1,
                "jodi": jodi_str,
                "confidence_pct": round(float(jodi_matrix[t, o] * 100), 2),
                "rashi": f"{RASHI_MAP[t]}{RASHI_MAP[o]}"
            })
            andar_counts[t] = andar_counts.get(t, 0) + 1
            bahar_counts[o] = bahar_counts.get(o, 0) + 1
            used_indices.add(idx_cand)

            if len(top_primary_jodis) == 5:
                break

        # Support Jodis (Next 5 top probabilities from matrix)
        top_support_jodis = []
        for idx_cand in ranked_indices:
            if idx_cand in used_indices:
                continue
            t, o = idx_cand // 10, idx_cand % 10
            top_support_jodis.append({
                "rank": len(top_support_jodis) + 1,
                "jodi": f"{t}{o}",
                "confidence_pct": round(float(jodi_matrix[t, o] * 100), 2),
                "rashi": f"{RASHI_MAP[t]}{RASHI_MAP[o]}"
            })
            if len(top_support_jodis) == 5:
                break

        # Calculate Solid Harufs (Andar & Bahar)
        sorted_andar = np.argsort(full_pa)[::-1]
        sorted_bahar = np.argsort(full_pb)[::-1]

        solid_andar = {
            "primary": int(sorted_andar[0]),
            "primary_prob": round(float(full_pa[sorted_andar[0]] * 100), 1),
            "secondary": int(sorted_andar[1]),
            "secondary_prob": round(float(full_pa[sorted_andar[1]] * 100), 1),
            "rashi_primary": int(RASHI_MAP[sorted_andar[0]])
        }

        solid_bahar = {
            "primary": int(sorted_bahar[0]),
            "primary_prob": round(float(full_pb[sorted_bahar[0]] * 100), 1),
            "secondary": int(sorted_bahar[1]),
            "secondary_prob": round(float(full_pb[sorted_bahar[1]] * 100), 1),
            "rashi_primary": int(RASHI_MAP[sorted_bahar[0]])
        }

        # 6-Digit Crossing System
        crossing_pool = []
        def _add_cand(c):
            if c is not None and c >= 0 and c not in crossing_pool:
                crossing_pool.append(c)

        # 1. High Probability Cross-Game Relations
        for d in sorted(list(cross_game_harufs)):
            _add_cand(d)

        # 2. Top Pick digits
        top_j = top_primary_jodis[0]["jodi"]
        _add_cand(int(top_j[0]))
        _add_cand(int(top_j[1]))

        # 3. Model's top Andar and Bahar candidates
        _add_cand(solid_andar["primary"])
        _add_cand(solid_bahar["primary"])
        _add_cand(solid_andar["secondary"])
        _add_cand(solid_bahar["secondary"])

        # 4. Primary Rashi mirrors
        _add_cand(solid_andar["rashi_primary"])
        _add_cand(solid_bahar["rashi_primary"])

        # 5. Complete pool from sorted marginal distributions
        for a_cand, b_cand in zip(sorted_andar, sorted_bahar):
            _add_cand(a_cand)
            _add_cand(b_cand)

        crossing_digits = [str(d) for d in crossing_pool[:6]]

        ref_desc = " | ".join(ref_desc_parts) if ref_desc_parts else ""
        reasoning = (
            f"Dual-Ensemble (Random Forest + HistGradientBoosting) trained on {len(self.df)} draws. {ref_desc}. "
            f"Andar Haruf '{solid_andar['primary']}' ({solid_andar['primary_prob']}%), Bahar Haruf '{solid_bahar['primary']}' ({solid_bahar['primary_prob']}%). "
            f"Top synthesis pair is '{top_primary_jodis[0]['jodi']}'."
        )

        return {
            "game": self.target_game,
            "prediction_for_date": pred_date,
            "current_status": status_label,
            "today_declared": today_declared,
            "is_next_day": is_next_day,
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
        else:
            p = self.predictors[game]
            p.df = df.copy()
            p._prepare_dataset()
        return self.predictors[game]

    def predict_all(self, df):
        results = {}
        backtests = {}
        for game in ["DSWR", "DLBZ", "SRGN", "FRBD", "GBD", "GALI"]:
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
