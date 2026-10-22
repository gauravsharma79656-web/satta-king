import numpy as np
import pandas as pd
from collections import defaultdict, Counter

RASHI_MAP = {
    0: 5, 1: 6, 2: 7, 3: 8, 4: 9,
    5: 0, 6: 1, 7: 2, 8: 3, 9: 4
}

def get_digits(num_str):
    try:
        val = int(num_str)
        tens = val // 10
        ones = val % 10
        return tens, ones
    except (ValueError, TypeError):
        return None, None

def get_rashi_jodi(val):
    if val is None or val < 0 or val > 99:
        return None
    tens = val // 10
    ones = val % 10
    return RASHI_MAP[tens] * 10 + RASHI_MAP[ones]

class RelationAnalyzer:
    def __init__(self, df):
        self.df = df.copy()
        self._prepare_data()

    def _prepare_data(self):
        # Convert non-digits to NaN
        for col in ["DSWR", "FRBD", "GBD", "GALI"]:
            if col in self.df.columns:
                self.df[col + "_num"] = pd.to_numeric(self.df[col], errors="coerce")
                self.df[col + "_tens"] = self.df[col + "_num"].apply(lambda v: int(v // 10) if pd.notna(v) else np.nan)
                self.df[col + "_ones"] = self.df[col + "_num"].apply(lambda v: int(v % 10) if pd.notna(v) else np.nan)
                self.df[col + "_sum"] = self.df[col + "_num"].apply(lambda v: int((v // 10 + v % 10) % 10) if pd.notna(v) else np.nan)

    def compute_game_correlations(self):
        """Pearson correlation matrix between the 4 games."""
        num_cols = ["DSWR_num", "FRBD_num", "GBD_num", "GALI_num"]
        sub_df = self.df[num_cols].dropna()
        corr = sub_df.corr()
        corr.columns = ["DSWR", "FRBD", "GBD", "GALI"]
        corr.index = ["DSWR", "FRBD", "GBD", "GALI"]
        return corr.round(3).to_dict()

    def compute_haruf_transfer_rates(self):
        """
        How often does any digit from Game A appear in Game B?
        Pairs in chronological order:
        1. Same day: DSWR -> FRBD
        2. Same day: FRBD -> GBD
        3. Same day: GBD -> GALI
        4. Cross day: GALI (t) -> DSWR (t+1)
        5. Same day: DSWR -> GBD
        6. Same day: DSWR -> GALI
        """
        pairs = [
            ("DSWR", "FRBD", "same_day"),
            ("FRBD", "GBD", "same_day"),
            ("GBD", "GALI", "same_day"),
            ("DSWR", "GBD", "same_day"),
            ("DSWR", "GALI", "same_day"),
            ("GALI", "DSWR", "next_day")
        ]

        results = {}
        for g1, g2, timing in pairs:
            total = 0
            hits_direct = 0
            hits_andar = 0
            hits_bahar = 0
            hits_rashi = 0

            for i in range(len(self.df) - (1 if timing == "next_day" else 0)):
                idx1 = i
                idx2 = i if timing == "same_day" else i + 1

                n1 = self.df.iloc[idx1][f"{g1}_num"]
                n2 = self.df.iloc[idx2][f"{g2}_num"]

                if pd.isna(n1) or pd.isna(n2):
                    continue

                total += 1
                t1, o1 = int(n1 // 10), int(n1 % 10)
                t2, o2 = int(n2 // 10), int(n2 % 10)

                digits1 = {t1, o1}
                digits2 = {t2, o2}
                rashi1 = {RASHI_MAP[t1], RASHI_MAP[o1]}

                # Any digit match (Haruf pass)
                if digits1 & digits2:
                    hits_direct += 1
                if t1 == t2 or o1 == t2:
                    hits_andar += 1
                if t1 == o2 or o1 == o2:
                    hits_bahar += 1
                # Rashi match
                if (digits1 | rashi1) & digits2:
                    hits_rashi += 1

            if total > 0:
                results[f"{g1} -> {g2} ({timing.replace('_', ' ')})"] = {
                    "total_draws": total,
                    "haruf_pass_rate_pct": round((hits_direct / total) * 100, 1),
                    "andar_pass_pct": round((hits_andar / total) * 100, 1),
                    "bahar_pass_pct": round((hits_bahar / total) * 100, 1),
                    "rashi_included_pct": round((hits_rashi / total) * 100, 1)
                }

        return results

    def compute_digit_difference_distribution(self, game1="DSWR", game2="FRBD"):
        """Distribution of delta = (Game2 - Game1) mod 100."""
        sub = self.df[[f"{game1}_num", f"{game2}_num"]].dropna()
        deltas = (sub[f"{game2}_num"] - sub[f"{game1}_num"]) % 100
        counts = deltas.value_counts().head(10).to_dict()
        return {int(k): int(v) for k, v in counts.items()}

    def compute_markov_digit_transitions(self, game="FRBD"):
        """Computes 10x10 digit transition probability matrix for tens and ones."""
        tens_transitions = np.zeros((10, 10))
        ones_transitions = np.zeros((10, 10))

        series_tens = self.df[f"{game}_tens"].dropna().values.astype(int)
        series_ones = self.df[f"{game}_ones"].dropna().values.astype(int)

        for i in range(len(series_tens) - 1):
            prev_t = series_tens[i]
            next_t = series_tens[i + 1]
            tens_transitions[prev_t, next_t] += 1

        for i in range(len(series_ones) - 1):
            prev_o = series_ones[i]
            next_o = series_ones[i + 1]
            ones_transitions[prev_o, next_o] += 1

        # Normalize
        t_row_sums = tens_transitions.sum(axis=1, keepdims=True)
        t_row_sums[t_row_sums == 0] = 1
        tens_prob = tens_transitions / t_row_sums

        o_row_sums = ones_transitions.sum(axis=1, keepdims=True)
        o_row_sums[o_row_sums == 0] = 1
        ones_prob = ones_transitions / o_row_sums

        return tens_prob, ones_prob

    def compute_hot_cold_frequencies(self, game="FRBD", last_n=100):
        """Calculates most frequent (hot) and least frequent (cold) Jodis and Harufs."""
        recent_df = self.df.tail(last_n)
        series = recent_df[f"{game}_num"].dropna().astype(int)

        counts = Counter(series)
        hot_jodis = [{"jodi": f"{num:02d}", "count": count} for num, count in counts.most_common(10)]

        # All numbers 0-99
        all_counts = {i: counts.get(i, 0) for i in range(100)}
        sorted_cold = sorted(all_counts.items(), key=lambda x: x[1])
        cold_jodis = [{"jodi": f"{num:02d}", "count": count} for num, count in sorted_cold[:10]]

        # Tens and Ones frequency
        tens_counts = Counter([num // 10 for num in series])
        ones_counts = Counter([num % 10 for num in series])

        hot_andar = [{"digit": d, "count": tens_counts.get(d, 0)} for d in range(10)]
        hot_andar = sorted(hot_andar, key=lambda x: x["count"], reverse=True)

        hot_bahar = [{"digit": d, "count": ones_counts.get(d, 0)} for d in range(10)]
        hot_bahar = sorted(hot_bahar, key=lambda x: x["count"], reverse=True)

        return {
            "game": game,
            "sample_size": len(series),
            "hot_jodis": hot_jodis,
            "cold_jodis": cold_jodis,
            "hot_andar": hot_andar,
            "hot_bahar": hot_bahar
        }

if __name__ == "__main__":
    from data_manager import get_latest_dataset
    df = get_latest_dataset()
    analyzer = RelationAnalyzer(df)
    print("Correlations:")
    print(analyzer.compute_game_correlations())
    print("\nHaruf Transfer Rates:")
    print(analyzer.compute_haruf_transfer_rates())
    print("\nFRBD Hot/Cold:")
    print(analyzer.compute_hot_cold_frequencies("FRBD"))
