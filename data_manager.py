import os
import re
import requests
import datetime
import pandas as pd
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ALL_GAMES_CSV = os.path.join(DATA_DIR, "all_games_dataset.csv")
ORIGINAL_FRBD_CSV = os.path.join(DATA_DIR, "history_frbd_original.csv")

BASE_URL = "https://satta-king-fast.com"
MONTH_NAMES = [
    (1, "January"), (2, "February"), (3, "March"), (4, "April"),
    (5, "May"), (6, "June"), (7, "July"), (8, "August"),
    (9, "September"), (10, "October"), (11, "November"), (12, "December")
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

def clean_cell(val):
    if not val:
        return "XX"
    val = str(val).strip().upper()
    if val in ["XX", "WAIT", "--", "-", ""]:
        return "XX"
    # if it has digits, keep 2 digits
    digits = re.findall(r"\d+", val)
    if digits:
        num = int(digits[0]) % 100
        return f"{num:02d}"
    return "XX"

def scrape_month(year, month_num, month_name):
    url = f"{BASE_URL}/chart.php?ResultFor={month_name}-{year}&month={month_num:02d}&year={year}#mix-chart"
    records = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return records
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
            header_cells = [c.get_text(strip=True).upper() for c in rows[1].find_all(["th", "td"])]
            if "DATE" in header_cells and ("DSWR" in header_cells or "FRBD" in header_cells):
                # We found the target table
                # Check column indices
                dswr_idx = header_cells.index("DSWR") if "DSWR" in header_cells else 1
                frbd_idx = header_cells.index("FRBD") if "FRBD" in header_cells else 2
                gbd_idx = -1
                for cand in ["GZBD", "GBD", "GHAZIABAD"]:
                    if cand in header_cells:
                        gbd_idx = header_cells.index(cand)
                        break
                if gbd_idx == -1:
                    gbd_idx = 3
                gali_idx = header_cells.index("GALI") if "GALI" in header_cells else 4

                for row in rows[2:]:
                    cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                    if not cells or not cells[0].isdigit():
                        continue
                    day = int(cells[0])
                    date_str = f"{year}-{month_num:02d}-{day:02d}"
                    
                    dswr = clean_cell(cells[dswr_idx]) if len(cells) > dswr_idx else "XX"
                    frbd = clean_cell(cells[frbd_idx]) if len(cells) > frbd_idx else "XX"
                    gbd = clean_cell(cells[gbd_idx]) if len(cells) > gbd_idx else "XX"
                    gali = clean_cell(cells[gali_idx]) if len(cells) > gali_idx else "XX"

                    records.append({
                        "date": date_str,
                        "DSWR": dswr,
                        "FRBD": frbd,
                        "GBD": gbd,
                        "GALI": gali
                    })
                break
    except Exception as e:
        print(f"Error scraping {year}-{month_num}: {e}")
    return records

def fetch_live_table():
    """Scrapes the live current month table from the home page."""
    records = []
    try:
        resp = requests.get(BASE_URL, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return records
        soup = BeautifulSoup(resp.text, "html.parser")
        tables = soup.find_all("table")
        for table in tables:
            rows = table.find_all("tr")
            if len(rows) < 2:
                continue
            header_cells = [c.get_text(strip=True).upper() for c in rows[1].find_all(["th", "td"])]
            if "DATE" in header_cells and ("DSWR" in header_cells or "FRBD" in header_cells):
                now = datetime.datetime.now()
                year = now.year
                month = now.month
                for row in rows[2:]:
                    cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
                    if not cells or not cells[0].isdigit():
                        continue
                    day = int(cells[0])
                    date_str = f"{year}-{month:02d}-{day:02d}"
                    dswr = clean_cell(cells[1]) if len(cells) > 1 else "XX"
                    frbd = clean_cell(cells[2]) if len(cells) > 2 else "XX"
                    gbd = clean_cell(cells[3]) if len(cells) > 3 else "XX"
                    gali = clean_cell(cells[4]) if len(cells) > 4 else "XX"
                    records.append({
                        "date": date_str,
                        "DSWR": dswr,
                        "FRBD": frbd,
                        "GBD": gbd,
                        "GALI": gali
                    })
                break
    except Exception as e:
        print(f"Error fetching live table: {e}")
    return records

from concurrent.futures import ThreadPoolExecutor

def build_full_dataset(years=range(2015, 2027), progress_callback=None):
    os.makedirs(DATA_DIR, exist_ok=True)
    all_rows = {}
    now = datetime.datetime.now()

    tasks = []
    for y in years:
        for m_num, m_name in MONTH_NAMES:
            if y == now.year and m_num > now.month:
                break
            tasks.append((y, m_num, m_name))

    print(f"Fetching {len(tasks)} months from {min(years)} to {max(years)} using parallel threads...")
    
    def worker(item):
        y, m_num, m_name = item
        return scrape_month(y, m_num, m_name)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(worker, tasks)
        for rows in results:
            for r in rows:
                all_rows[r["date"]] = r

    # Also update with live table
    live_rows = fetch_live_table()
    for r in live_rows:
        if r["date"] in all_rows:
            for k in ["DSWR", "FRBD", "GBD", "GALI"]:
                if r[k] != "XX":
                    all_rows[r["date"]][k] = r[k]
        else:
            all_rows[r["date"]] = r

    # Cross-verify with original FRBD if exists
    if os.path.exists(ORIGINAL_FRBD_CSV):
        try:
            user_df = pd.read_csv(ORIGINAL_FRBD_CSV)
            if "date" in user_df.columns and "result" in user_df.columns:
                for _, urow in user_df.iterrows():
                    d = str(urow["date"]).strip()
                    res = clean_cell(urow["result"])
                    if d in all_rows and res != "XX":
                        all_rows[d]["FRBD"] = res
                    elif d not in all_rows and res != "XX":
                        all_rows[d] = {
                            "date": d,
                            "DSWR": "XX",
                            "FRBD": res,
                            "GBD": "XX",
                            "GALI": "XX"
                        }
        except Exception as e:
            print("Error loading original FRBD csv:", e)

    sorted_dates = sorted(all_rows.keys())
    data_list = [all_rows[d] for d in sorted_dates]
    df = pd.DataFrame(data_list)
    df.to_csv(ALL_GAMES_CSV, index=False)
    print(f"Dataset successfully compiled: {len(df)} days recorded at {ALL_GAMES_CSV}")
    return df

def get_latest_dataset(force_refresh=False):
    if force_refresh or not os.path.exists(ALL_GAMES_CSV):
        return build_full_dataset()
    return pd.read_csv(ALL_GAMES_CSV, dtype=str)

def sync_latest_today():
    """Quick sync for today and yesterday only, without re-scraping all 3 years."""
    df = get_latest_dataset()
    live_rows = fetch_live_table()
    if not live_rows:
        return df

    df_dict = {row["date"]: row.to_dict() for _, row in df.iterrows()}
    updated = False
    for r in live_rows:
        d = r["date"]
        if d in df_dict:
            for k in ["DSWR", "FRBD", "GBD", "GALI"]:
                if r[k] != "XX" and df_dict[d][k] != r[k]:
                    df_dict[d][k] = r[k]
                    updated = True
        else:
            df_dict[d] = r
            updated = True

    if updated:
        sorted_dates = sorted(df_dict.keys())
        new_df = pd.DataFrame([df_dict[d] for d in sorted_dates])
        new_df.to_csv(ALL_GAMES_CSV, index=False)
        return new_df
    return df

if __name__ == "__main__":
    print("Testing data manager...")
    df = build_full_dataset()
    print("Shape:", df.shape)
    print("Last 5 records:")
    print(df.tail(5))
