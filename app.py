import streamlit as st
import pandas as pd
import duckdb
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import date
import re
import urllib.parse

# ---------------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------------
DATA_URL ="https://github.com/PawelWyrodek/PowerliftingCompare/releases/download/latest-data/openpowerlifting.parquet"
st.set_page_config(page_title="Performance Engine", layout="wide", initial_sidebar_state="expanded")

# Cache refreshes once a day
@st.cache_data(ttl="1d")
def load_data():
    return pd.read_parquet(DATA_URL)

st.title("Powerlifting Performance Comparison")

# Load data
df = load_data()
st.markdown("""
    <style>
        div[data-testid="InputInstructions"] { display: none !important; }
        .block-container { padding-top: 3rem; padding-bottom: 3rem; }
        a.header-anchor { display: none !important; }
        div[data-testid="stHeader"] a { display: none !important; }
        h1:hover a, h2:hover a, h3:hover a { display: none !important; }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# INITIALIZATION & STATE MANAGEMENT
# ---------------------------------------------------------------------------
for key in ["df_a", "df_b", "ath_df", "ath_df_a", "ath_df_b", "df_comp"]:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame()
        
# Read URL only on the FIRST app launch to avoid conflict with menu changes
if "app_initialized" not in st.session_state:
    params = st.query_params
    if "mode" in params:
        st.session_state.active_mode = params["mode"]
    else:
        st.session_state.active_mode = "Result vs group"
        
    if params.get("mode") == "Athlete" and "athlete" in params:
        st.session_state.ath_selected = urllib.parse.unquote(params["athlete"])
        st.session_state.submit_clicked = True
        
    st.session_state.app_initialized = True

if "active_mode" not in st.session_state: 
    st.session_state.active_mode = "Result vs group"

if "score_sys" not in st.session_state: st.session_state.score_sys = "Dots"
if "submit_clicked" not in st.session_state: st.session_state.submit_clicked = False

def fmt(val):
    if pd.isna(val): return ""
    val = float(val)
    return f"{int(val)}" if val == int(val) else f"{val:.1f}"

def format_attempt(val):
    if pd.isna(val) or val == 0: return ""
    f_val = float(val)
    if f_val < 0: return f"{abs(f_val)} (Failed)"
    return str(f_val)

def normalize_meet_name(name):
    if not isinstance(name, str): return ""
    n = re.sub(r'\b\d+(st|nd|rd|th)\b', '', name, flags=re.IGNORECASE)
    n = re.sub(r'\b\d{4}\b', '', n)
    n = re.sub(r'\bannual\b', '', n, flags=re.IGNORECASE)
    return " ".join(n.split()).strip()

def calc_1rm(weight, reps):
    if reps <= 1: return weight
    return weight * (1.0 + (reps / 30.0))

def sort_weight_class(value):
    s = str(value).replace('+', '').strip()
    try: return float(s)
    except ValueError: return 9999.0

# ---------------------------------------------------------------------------
# DUCKDB CONNECTION & DATA WASHER
# ---------------------------------------------------------------------------
@st.cache_resource
def get_connection(_data):
    conn = duckdb.connect()
    
    # Register the loaded Pandas DataFrame directly instead of using a SQLite file
    conn.register("main_db_powerlifting", _data)
    
    conn.execute("""
        CREATE OR REPLACE TEMPORARY VIEW clean_db AS
        SELECT * REPLACE (
            TRY_CAST(Date AS DATE) as Date,
            TRY_CAST(Age AS DOUBLE) as Age,
            TRY_CAST(BodyweightKg AS DOUBLE) as BodyweightKg,
            TRY_CAST(Best3SquatKg AS DOUBLE) as Best3SquatKg,
            TRY_CAST(Best3BenchKg AS DOUBLE) as Best3BenchKg,
            TRY_CAST(Best3DeadliftKg AS DOUBLE) as Best3DeadliftKg,
            TRY_CAST(TotalKg AS DOUBLE) as TotalKg,
            TRY_CAST(Dots AS DOUBLE) as Dots,
            TRY_CAST(Wilks AS DOUBLE) as Wilks,
            TRY_CAST(Squat1Kg AS DOUBLE) as Squat1Kg,
            TRY_CAST(Squat2Kg AS DOUBLE) as Squat2Kg,
            TRY_CAST(Squat3Kg AS DOUBLE) as Squat3Kg,
            TRY_CAST(Bench1Kg AS DOUBLE) as Bench1Kg,
            TRY_CAST(Bench2Kg AS DOUBLE) as Bench2Kg,
            TRY_CAST(Bench3Kg AS DOUBLE) as Bench3Kg,
            TRY_CAST(Deadlift1Kg AS DOUBLE) as Deadlift1Kg,
            TRY_CAST(Deadlift2Kg AS DOUBLE) as Deadlift2Kg,
            TRY_CAST(Deadlift3Kg AS DOUBLE) as Deadlift3Kg
        ),
        CASE WHEN Sex = 'M' THEN
            TRY_CAST(TotalKg AS DOUBLE) * 100.0 / (1199.72839 - 925.40462 * EXP(-0.00510531 * TRY_CAST(BodyweightKg AS DOUBLE)))
        WHEN Sex = 'F' THEN
            TRY_CAST(TotalKg AS DOUBLE) * 100.0 / (610.79046 - 451.04414 * EXP(-0.00735665 * TRY_CAST(BodyweightKg AS DOUBLE)))
        ELSE NULL END as GLPoints
        FROM main_db_powerlifting;
    """)
    return conn

@st.cache_data
def load_countries_and_continents(_data):
    default_continents = {
        "Africa": ["Niger", "Nigeria", "South Africa", "Egypt", "Algeria", "Morocco", "Ghana", "Cameroon", "Zimbabwe", "Kenya", "Senegal"],
        "Asia": ["Japan", "Kazakhstan", "China", "India", "Iran", "Taiwan", "South Korea", "Mongolia", "Singapore", "Malaysia", "Philippines", "Indonesia", "UAE"],
        "Europe": ["Poland", "Russia", "USSR", "Ukraine", "UK", "Great Britain", "England", "France", "Germany", "Sweden", "Norway", "Finland", "Italy", "Spain", "Denmark", "Ireland", "Czechia", "Hungary", "Romania", "Belarus", "Lithuania", "Latvia", "Estonia", "Netherlands", "Belgium", "Austria", "Switzerland", "Greece", "Portugal"],
        "North America": ["USA", "US", "United States", "Canada", "Mexico", "Puerto Rico", "Costa Rica", "Jamaica"],
        "Oceania": ["Australia", "New Zealand", "Nauru", "Fiji", "Samoa"],
        "South America": ["Brazil", "Argentina", "Chile", "Colombia", "Peru", "Ecuador", "Venezuela", "Uruguay"]
    }
    try:
        conn = get_connection(_data)
        df_db = conn.execute("SELECT DISTINCT Country FROM clean_db WHERE Country IS NOT NULL").df()
        db_countries = sorted([str(c).strip() for c in df_db['Country'].dropna().unique() if str(c).strip()])
    except Exception:
        all_c = []
        for c_list in default_continents.values(): all_c.extend(c_list)
        db_countries = sorted(list(set(all_c)))

    display_countries = sorted(list(set(["USA" if c in ["US", "United States"] else c for c in db_countries])))
    country_to_continent = {}
    for country in display_countries:
        lower_c = country.lower().strip()
        matched = False
        for cont, countries_list in default_continents.items():
            if any(x.lower() == lower_c for x in countries_list):
                country_to_continent[country] = cont
                matched = True
                break
        if not matched: country_to_continent[country] = "Other"

    return display_countries, country_to_continent

@st.cache_data
def load_federations(_data):
    try:
        conn = get_connection(_data)
        feds_df = conn.execute("SELECT DISTINCT Federation, ParentFederation FROM clean_db WHERE Federation IS NOT NULL").df()
        feds_list = sorted(feds_df['Federation'].dropna().unique().tolist())
        parent_feds_list = sorted(feds_df['ParentFederation'].dropna().unique().tolist())
        fed_to_parent = dict(zip(feds_df['Federation'], feds_df['ParentFederation']))
        return feds_list, parent_feds_list, fed_to_parent
    except Exception: 
        return [], [], {}

@st.cache_data
def load_weight_class_options(_data):
    popular_classes = ('43', '44', '47', '48', '52', '53', '56', '57', '59', '60', '63', '66', '67.5', '69', '74', '75', '76', '82.5', '83', '84', '84+', '90', '90+', '93', '100', '100+', '105', '110', '115', '120', '120+', '125', '140', '140+')
    try:
        conn = get_connection(_data)
        placeholders = ",".join(["?"] * len(popular_classes))
        q = f"SELECT DISTINCT WeightClassKg FROM clean_db WHERE WeightClassKg IN ({placeholders})"
        df_db = conn.execute(q, popular_classes).df()
        if df_db.empty: return []
        
        df_db['WeightClassKg'] = df_db['WeightClassKg'].astype(str).str.strip()
        options = df_db['WeightClassKg'].unique()
        return sorted(list(options), key=lambda x: sort_weight_class(x))
    except Exception: return []

display_countries, country_to_continent = load_countries_and_continents(df)
feds_list, parent_feds_list, fed_to_parent = load_federations(df)
weight_class_options = load_weight_class_options(df)

METRIC_OPTIONS = ["Total", "Dots", "Wilks", "GL Points", "Squat", "Bench", "Deadlift", "Bodyweight", "Date"]
METRIC_SQL_MAP = {
    "Total": "r.TotalKg",
    "Dots": "r.Dots",
    "Wilks": "r.Wilks",
    "GL Points": "r.GLPoints",
    "Squat": "r.Best3SquatKg",
    "Bench": "r.Best3BenchKg",
    "Deadlift": "r.Best3DeadliftKg",
    "Bodyweight": "r.BodyweightKg"
}

# ---------------------------------------------------------------------------
# CORE DATA PROCESSING
# ---------------------------------------------------------------------------
def build_query(cfg):
    params = []
    base_where = "WHERE 1=1"

    if cfg.get("events"):
        event_conditions = []
        for ev in cfg["events"]:
            if ev == "SBD":
                event_conditions.append("(p.Event = 'SBD' AND p.Best3SquatKg > 0 AND p.Best3BenchKg > 0 AND p.Best3DeadliftKg > 0 AND p.TotalKg > 0)")
            elif ev == "S":
                event_conditions.append("(p.Event = 'S' AND p.Best3SquatKg > 0)")
            elif ev == "B":
                event_conditions.append("(p.Event = 'B' AND p.Best3BenchKg > 0)")
            elif ev == "D":
                event_conditions.append("(p.Event = 'D' AND p.Best3DeadliftKg > 0)")
            elif ev == "SB":
                event_conditions.append("(p.Event = 'SB' AND p.Best3SquatKg > 0 AND p.Best3BenchKg > 0)")
            elif ev == "BD":
                event_conditions.append("(p.Event = 'BD' AND p.Best3BenchKg > 0 AND p.Best3DeadliftKg > 0)")
            elif ev == "SD":
                event_conditions.append("(p.Event = 'SD' AND p.Best3SquatKg > 0 AND p.Best3DeadliftKg > 0)")
            else:
                event_conditions.append(f"(p.Event = '{ev}')")
        
        if event_conditions:
            base_where += " AND (" + " OR ".join(event_conditions) + ")"

    if cfg["metric"] == "Total" and cfg.get("events") and "SBD" not in cfg.get("events"):
        base_where += " AND p.TotalKg > 0"

    if cfg["date_preset"] != "Any":
        base_where += " AND p.Date >= ? AND p.Date <= ?"
        params += [cfg["start_date"], cfg["end_date"]]

    if cfg["sex"] != "Any":
        base_where += " AND p.Sex = ?"
        params.append(cfg["sex"])
        
    if cfg["equip"]:
        placeholders = ",".join(["?"] * len(cfg["equip"]))
        base_where += f" AND p.Equipment IN ({placeholders})"
        params += cfg["equip"]

    base_where += " AND p.Age BETWEEN ? AND ?"
    params += [cfg["a_min"], cfg["a_max"]]
    base_where += " AND p.BodyweightKg BETWEEN ? AND ?"
    params += [cfg["w_min"], cfg["w_max"]]

    if cfg["tested"] == "No": base_where += " AND p.Tested IS NULL"
    elif cfg["tested"] == "Yes": base_where += " AND p.Tested = 'Yes'"

    if cfg["weight_classes"]:
        clauses = []
        for opt in cfg["weight_classes"]:
            clauses.append("(p.WeightClassKg = ?)")
            params.append(opt)
        base_where += " AND (" + " OR ".join(clauses) + ")"

    if cfg["parent_feds"]:
        placeholders = ",".join(["?"] * len(cfg["parent_feds"]))
        base_where += f" AND p.ParentFederation IN ({placeholders})"
        params += cfg["parent_feds"]
        
    if cfg["feds"]:
        placeholders = ",".join(["?"] * len(cfg["feds"]))
        base_where += f" AND p.Federation IN ({placeholders})"
        params += cfg["feds"]

    geo_countries = set(cfg["countries"])
    if geo_countries:
        expanded = set()
        for c in geo_countries:
            expanded.add(c)
            if c == "USA": expanded.update(["US", "United States"])
        placeholders = ",".join(["?"] * len(expanded))
        base_where += f" AND p.Country IN ({placeholders})"
        params += list(expanded)

    base_params = list(params) 

    filtered_params = []
    filtered_where = "WHERE r.MeetNum BETWEEN ? AND ?"
    filtered_params += [cfg["meet_min"], cfg["meet_max"]]
    
    filtered_where += " AND (r.Dots BETWEEN ? AND ? OR r.Dots IS NULL)"
    filtered_params += [cfg["dots_min"], cfg["dots_max"]]
    
    filtered_where += " AND (date_diff('day', r.CareerStart::DATE, r.Date::DATE))/365.25 BETWEEN ? AND ?"
    filtered_params += [cfg["long_min"], cfg["long_max"]]

    filtered_where += " AND (r.TotalKg BETWEEN ? AND ? OR r.TotalKg IS NULL)"
    filtered_params += [cfg["tot_min"], cfg["tot_max"]]
    filtered_where += " AND (r.Best3SquatKg BETWEEN ? AND ? OR r.Best3SquatKg IS NULL)"
    filtered_params += [cfg["sq_min"], cfg["sq_max"]]
    filtered_where += " AND (r.Best3BenchKg BETWEEN ? AND ? OR r.Best3BenchKg IS NULL)"
    filtered_params += [cfg["bn_min"], cfg["bn_max"]]
    filtered_where += " AND (r.Best3DeadliftKg BETWEEN ? AND ? OR r.Best3DeadliftKg IS NULL)"
    filtered_params += [cfg["dl_min"], cfg["dl_max"]]

    metric = cfg["metric"]
    if metric == "Date":
        metric_sql = "r.Date"
        partition_order = "ORDER BY r.Date ASC"
        final_order = "ORDER BY RankMetric ASC"
    else:
        metric_sql = METRIC_SQL_MAP[metric]
        partition_order = f"ORDER BY {metric_sql} DESC"
        final_order = "ORDER BY RankMetric DESC"

    try: 
        limit_val = str(cfg["top_n"]).strip()
        if limit_val.lower() == "any" or limit_val == "":
            limit_sql = ""
        else:
            limit_sql = f"LIMIT {int(limit_val)}"
    except: 
        limit_sql = ""

    query = f"""
        WITH TargetAthletes AS (
            SELECT DISTINCT p.Name, IFNULL(p.Country, '') as CountryStr
            FROM clean_db p
            {base_where}
        ),
        CareerData AS (
            SELECT p.*,
                   ROW_NUMBER() OVER(PARTITION BY p.Name, IFNULL(p.Country, '') ORDER BY p.Date) as MeetNum,
                   MIN(p.Date) OVER(PARTITION BY p.Name, IFNULL(p.Country, '')) as CareerStart,
                   MAX(p.Date) OVER(PARTITION BY p.Name, IFNULL(p.Country, '')) as CareerEnd
            FROM clean_db p
            INNER JOIN TargetAthletes ta ON p.Name = ta.Name AND IFNULL(p.Country, '') = ta.CountryStr
        ),
        RawMeets AS (
            SELECT r.*
            FROM CareerData r
            {base_where.replace('p.', 'r.')}
        ),
        FilteredMeets AS (
            SELECT r.*,
                   (date_diff('day', r.CareerStart::DATE, r.Date::DATE)) / 365.25 as LongevityToMeet,
                   {metric_sql} as RankMetric
            FROM RawMeets r
            {filtered_where}
        ),
        RankedMeets AS (
            SELECT *,
                   ROW_NUMBER() OVER(PARTITION BY Name, Country {partition_order}) as PR_Rank
            FROM FilteredMeets r
        )
        SELECT * FROM RankedMeets WHERE PR_Rank = 1
        {final_order}
        {limit_sql}
    """
    
    final_params = base_params + base_params + filtered_params
    return query, final_params

def run_group_analysis(cfg, _data):
    conn = get_connection(_data)
    query, params = build_query(cfg)
    df_res = conn.execute(query, params).df()

    if df_res.empty: return df_res
    
    df_res["Age"] = df_res["Age"].fillna(0).astype(int)
    df_res['Date'] = pd.to_datetime(df_res['Date'], errors='coerce')
    
    df_res.rename(columns={
        "TotalKg": "Total", "Best3SquatKg": "Squat", "Best3BenchKg": "Bench", 
        "Best3DeadliftKg": "Deadlift", "BodyweightKg": "Bodyweight", 
        "MeetNum": "Meet#", "LongevityToMeet": "Experience", "WeightClassKg": "WeightClass",
        "GLPoints": "GL Points"
    }, inplace=True)
    
    for col in ["Squat", "Bench", "Deadlift", "Total", "Dots", "Wilks", "GL Points"]:
        if col in df_res.columns: df_res[col] = df_res[col].fillna(0)
        
    return df_res

def check_weight_conflict(cfg):
    if cfg["weight_classes"]:
        w_min = cfg["w_min"]
        w_max = cfg["w_max"]
        overlap = False
        for wc in cfg["weight_classes"]:
            val = float(str(wc).replace('+', '').strip())
            if w_min <= val <= w_max:
                overlap = True
                break
        if not overlap:
            st.warning("Warning: Your selected Bodyweight Range might not completely overlap with the chosen Weight classes. Some athletes may be naturally lighter than their upper weight class limit.")

def render_dataframe(df_disp, key_prefix="table", set_index=None):
    if df_disp is None or df_disp.empty:
        st.dataframe(df_disp, use_container_width=True)
        return

    df_disp = df_disp.copy()
    config = {}
    
    if "Name" in df_disp.columns:
        search_val = st.text_input("Search Athlete Name:", key=f"{key_prefix}_search").strip().lower()
        if search_val:
            mask = df_disp["Name"].astype(str).str.lower().str.contains(search_val, na=False)
            df_disp = df_disp[mask]
        
        def make_url(name):
            n_str = str(name)
            enc_name = urllib.parse.quote(n_str)
            return f"?mode=Athlete&athlete={enc_name}#{n_str}"
        
        df_disp["Name"] = df_disp["Name"].apply(make_url)
        config["Name"] = st.column_config.LinkColumn("Name", display_text=r"#(.*)")
        
    if set_index and set_index in df_disp.columns:
        st.dataframe(df_disp.set_index(set_index), use_container_width=True, column_config=config)
    else:
        st.dataframe(df_disp, use_container_width=True, column_config=config)

# ---------------------------------------------------------------------------
# UI HELPERS & COMPONENTS
# ---------------------------------------------------------------------------
def numeric_range_input(label, min_val, max_val, default_low, default_high, step, key_prefix, slider_max=None):
    sl_key = f"{key_prefix}_{label}_sl"
    min_key = f"{key_prefix}_{label}_min"
    max_key = f"{key_prefix}_{label}_max"
    
    sm = float(slider_max if slider_max is not None else max_val)

    if sl_key not in st.session_state:
        st.session_state[sl_key] = (min(float(default_low), sm), min(float(default_high), sm))
    if min_key not in st.session_state:
        st.session_state[min_key] = float(default_low)
    if max_key not in st.session_state:
        st.session_state[max_key] = float(default_high)

    def sync_from_num():
        low = st.session_state[min_key]
        high = st.session_state[max_key]
        if low > high: low = high
        st.session_state[sl_key] = (min(low, sm), min(high, sm))

    def sync_from_sl():
        st.session_state[min_key] = st.session_state[sl_key][0]
        st.session_state[max_key] = st.session_state[sl_key][1]

    st.markdown(f"<span style='font-size: 0.9em; font-weight: bold;'>{label}</span>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    c1.number_input("Min", min_value=float(min_val), max_value=float(max_val), step=float(step), key=min_key, on_change=sync_from_num, label_visibility="collapsed")
    c2.number_input("Max", min_value=float(min_val), max_value=float(max_val), step=float(step), key=max_key, on_change=sync_from_num, label_visibility="collapsed")
    st.slider(f"Slider {label}", min_value=float(min_val), max_value=float(sm), step=float(step), key=sl_key, on_change=sync_from_sl, label_visibility="collapsed")

    return st.session_state[min_key], st.session_state[max_key]

# SAFE ATHLETE RENDERING (Solves the problem of app freezing with a large number of options)
def render_athlete_selectbox(label, key, _data):
    search_key = f"{key}_search_input"
    current_val = st.session_state.get(key, None)
    
    # Step 1: Text input for dynamic search
    search_term = st.text_input(f"{label} (min. 3 characters):", key=search_key)
    
    options = []
        
    # If user types at least 3 characters, query the database
    if search_term and len(search_term) >= 3:
        try:
            conn = get_connection(_data)
            # ILIKE ensures case-insensitive search
            res = conn.execute("SELECT DISTINCT Name FROM clean_db WHERE Name ILIKE ? ORDER BY Name LIMIT 100", [f"%{search_term}%"]).df()
            if not res.empty:
                options.extend(res['Name'].tolist())
        except Exception:
            pass
            
    # Keep the previous selection (e.g., from URL or previous search)
    if current_val and current_val != "Type above to search..." and current_val not in options:
        options.insert(0, current_val)
    
    # Step 2: Selectbox limited to search results (safe for Streamlit)
    if options:
        return st.selectbox("Select exact match:", options=options, key=key)
    else:
        # Added missing key=key, resolving the ID conflict
        return st.selectbox("Select exact match:", options=["Type above to search..."], disabled=True, key=key)

# ---------------------------------------------------------------------------
# MAIN LAYOUT
# ---------------------------------------------------------------------------
st.sidebar.title("Navigation")
use_lbs = st.sidebar.toggle("Pounds (lbs)", value=False)
unit = "lbs" if use_lbs else "kg"
mult = 2.20462262 if use_lbs else 1.0

MODES = ["Result vs group", "Athlete vs group", "Group", "Group vs group", "Athlete", "Athlete vs athlete", "Competition Analysis", "Calculators"]

analysis_mode = st.sidebar.radio("Select Analysis Mode", MODES, index=MODES.index(st.session_state.active_mode))
st.session_state.active_mode = analysis_mode
st.query_params["mode"] = analysis_mode 

def render_group_filters(prefix, default_label, default_sex="M", default_equip=None, default_countries=None):
    if default_equip is None:
        default_equip = ["Raw"]
    default_countries = default_countries or []
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Name this Group</span>", unsafe_allow_html=True)
        group_name = st.text_input("Name this Group", value=default_label, key=f"{prefix}_name", label_visibility="collapsed")
        
        st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Event Types (Lifts)</span>", unsafe_allow_html=True)
        events_sel = st.multiselect("Event Types (Lifts)", ["SBD", "S", "B", "D", "SB", "BD", "SD"], default=["SBD"], key=f"{prefix}_events", label_visibility="collapsed")
        
        st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Sex</span>", unsafe_allow_html=True)
        sex = st.selectbox("Sex", ["M", "F", "Any"], index=["M", "F", "Any"].index(default_sex), key=f"{prefix}_sex", label_visibility="collapsed")
    
    with col2:
        st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Date</span>", unsafe_allow_html=True)
        date_preset = st.selectbox("Date", ["Any", "Last week", "Last month", "Last year", "Custom"], key=f"{prefix}_date_preset", label_visibility="collapsed")
        today = date.today()
        if date_preset == "Any": start_date, end_date = "1970-01-01", today.strftime("%Y-%m-%d")
        elif date_preset == "Last week": start_date, end_date = (today - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        elif date_preset == "Last month": start_date, end_date = (today - pd.Timedelta(days=30)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        elif date_preset == "Last year": start_date, end_date = (today - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), today.strftime("%Y-%m-%d")
        else:
            st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Custom date range</span>", unsafe_allow_html=True)
            date_range = st.date_input("Custom date range", value=(date(1970, 1, 1), today), key=f"{prefix}_date_custom", label_visibility="collapsed")
            start_date = date_range[0].strftime("%Y-%m-%d") if len(date_range) >= 1 else "1970-01-01"
            end_date = date_range[1].strftime("%Y-%m-%d") if len(date_range) == 2 else start_date

    with col3:
        st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Sort by</span>", unsafe_allow_html=True)
        metric_label = st.selectbox("Sort by", METRIC_OPTIONS, key=f"{prefix}_metric", label_visibility="collapsed")
        
        st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>TOP Limit</span>", unsafe_allow_html=True)
        top_n_choice = st.selectbox("TOP Limit", ["Any", "10", "50", "100", "1000", "Custom"], key=f"{prefix}_topn_choice", label_visibility="collapsed")
        if top_n_choice == "Custom":
            st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Enter exact limit number</span>", unsafe_allow_html=True)
            top_n_val = st.text_input("Enter exact limit number", value="", key=f"{prefix}_topn_custom", label_visibility="collapsed")
        else:
            top_n_val = "" if top_n_choice == "Any" else top_n_choice

    st.markdown("---")
    st.caption("Note: To effectively restrict bodyweight independently of weight categories, please adjust the Bodyweight slider below.")
    c1, c2, c3 = st.columns(3)
    with c1:
        a_min, a_max = numeric_range_input("Age Range", 1, 99, 18, 40, 1, prefix)
        w_min, w_max = numeric_range_input("Bodyweight Range", 0.0, 500.0, 0.0, 500.0, 0.5, prefix, slider_max=200.0)
        dots_min, dots_max = numeric_range_input("Score Range", 0.0, 750.0, 0.0, 750.0, 5.0, prefix)
    with c2:
        long_min, long_max = numeric_range_input("Experience (Years)", 0.0, 40.0, 0.0, 40.0, 0.5, prefix)
        meet_min, meet_max = numeric_range_input("Meet# Range", 1, 500, 1, 500, 1, prefix)
        tot_min, tot_max = numeric_range_input("Total Range", 0.0, 1500.0, 0.0, 1500.0, 2.5, prefix)
    with c3:
        sq_min, sq_max = numeric_range_input("Squat Range", 0.0, 600.0, 0.0, 600.0, 2.5, prefix)
        bn_min, bn_max = numeric_range_input("Bench Range", 0.0, 400.0, 0.0, 400.0, 2.5, prefix)
        dl_min, dl_max = numeric_range_input("Deadlift Range", 0.0, 500.0, 0.0, 500.0, 2.5, prefix)

    wco = weight_class_options
        
    c4, c5, c6 = st.columns(3)
    with c4:
        st.markdown("<span style='font-size: 0.9em; font-weight: bold;'>Weight classes Preset</span>", unsafe_allow_html=True)
        wc_preset = st.radio("Preset", ["Custom", "All Men", "All Women"], key=f"{prefix}_wc_preset", horizontal=True, label_visibility="collapsed")
        
        if wc_preset == "All Men":
            default_wc = ['52', '53', '56', '59', '60', '63', '66', '67.5', '69', '74', '75', '76', '82.5', '83', '90', '93', '100', '105', '110', '115', '120', '120+', '125', '140', '140+']
        elif wc_preset == "All Women":
            default_wc = ['43', '44', '47', '48', '52', '53', '56', '57', '60', '63', '67.5', '69', '75', '76', '82.5', '83', '84', '84+', '90', '90+']
        else:
            default_wc = []
            
        valid_default_wc = [w for w in default_wc if w in wco]
        weight_classes_sel = st.multiselect(f"Weight classes", options=wco, default=valid_default_wc, key=f"{prefix}_wc")
        tested = st.selectbox(f"Antidoping tested", ["Any", "Yes", "No"], key=f"{prefix}_tested")
        equip = st.multiselect(f"Equipment", ["Raw", "Wraps", "Multi-ply", "Single-ply"], default=default_equip, key=f"{prefix}_equip")
    with c5:
        parent_feds_sel = st.multiselect(f"Parent Federation", parent_feds_list, key=f"{prefix}_pfed")
        valid_feds = [f for f in feds_list if fed_to_parent.get(f) in parent_feds_sel] if parent_feds_sel else feds_list
        feds_sel = st.multiselect(f"Federation", valid_feds, key=f"{prefix}_fed")
    with c6:
        continents_sel = st.multiselect(f"Continents", ["Africa", "Asia", "Europe", "North America", "Oceania", "South America"], key=f"{prefix}_cont")
        valid_countries = [c for c in display_countries if country_to_continent.get(c) in continents_sel] if continents_sel else display_countries
        safe_defaults = [c for c in default_countries if c in valid_countries]
        cont_key_suffix = "_".join(continents_sel).replace(" ", "") if continents_sel else "all"
        countries_sel = st.multiselect(f"Countries", options=valid_countries, default=safe_defaults, key=f"{prefix}_country_{cont_key_suffix}")

    return dict(
        group_name=group_name, events=events_sel, sex=sex, date_preset=date_preset, start_date=start_date, end_date=end_date, 
        a_min=a_min, a_max=a_max, w_min=w_min, w_max=w_max, 
        dots_min=dots_min, dots_max=dots_max, long_min=long_min, long_max=long_max, meet_min=meet_min, meet_max=meet_max,
        tot_min=tot_min, tot_max=tot_max, sq_min=sq_min, sq_max=sq_max, bn_min=bn_min, bn_max=bn_max, dl_min=dl_min, dl_max=dl_max,
        weight_classes=weight_classes_sel, tested=tested, equip=equip, 
        parent_feds=parent_feds_sel, feds=feds_sel, continents=continents_sel, countries=countries_sel,
        metric=metric_label, top_n=top_n_val
    )

def render_strength_standards(df_src, cfg, exact_bw_target=None):
    with st.expander(f"{cfg['group_name']} Strength Standards", expanded=False):
        st.markdown("Standards are generated based on the currently filtered metric (greater than 0) inside the chosen group time range and filters.")
        
        target_metric = cfg["metric"] if cfg["metric"] in ["Total", "Squat", "Bench", "Deadlift"] else "Total"
        st.write(f"**Target Metric for standards**: {target_metric}")
        
        valid_df = df_src[df_src[target_metric] > 0].copy()
        valid_df['WeightClass'] = pd.to_numeric(valid_df['WeightClass'], errors='coerce')
        valid_df = valid_df.dropna(subset=['WeightClass'])
        
        if valid_df.empty:
            st.warning("No data available for strength standards with current filters.")
            return

        if exact_bw_target is not None and exact_bw_target > 0:
            exact_bw = exact_bw_target
            exact_bw_kg = exact_bw / mult if use_lbs else exact_bw
            st.info(f"Interpolated standards for your exact Bodyweight: {exact_bw} {unit}")
            
            agg_df = valid_df.groupby('WeightClass')[target_metric].quantile([0.1667, 0.3333, 0.5000, 0.6667, 0.8333]).unstack()
            agg_df = agg_df * mult
            
            if agg_df.empty or len(agg_df) < 2:
                st.warning("Not enough weight classes in filtered data to interpolate.")
                return

            wcs = agg_df.index.values
            interp_vals = []
            for col in agg_df.columns:
                interp_vals.append(np.interp(exact_bw_kg, wcs, agg_df[col].values))
            
            exact_df = pd.DataFrame([interp_vals], columns=["Beginner", "Novice", "Intermediate", "Advanced", "Elite"], index=[f"{exact_bw} {unit}"])
            exact_df = exact_df.round(1)
            st.dataframe(
                exact_df,
                use_container_width=True,
                column_config={
                    "Beginner": st.column_config.NumberColumn("Beginner", help="Better than 16.6%"),
                    "Novice": st.column_config.NumberColumn("Novice", help="Better than 33.3%"),
                    "Intermediate": st.column_config.NumberColumn("Intermediate", help="Better than 50.0%"),
                    "Advanced": st.column_config.NumberColumn("Advanced", help="Better than 66.6%"),
                    "Elite": st.column_config.NumberColumn("Elite", help="Better than 83.3%"),
                }
            )
            return

        col1, col2 = st.columns(2)
        with col1:
            divide_by_age = st.checkbox("Divide by Age (Subjunior, Junior, Open, Masters)", key=f"{cfg['group_name']}_ss_age")

        results = []
        quantiles = [0.1667, 0.3333, 0.5000, 0.6667, 0.8333]
        
        if divide_by_age:
            def get_age_group(age):
                if pd.isna(age) or age == 0: return "Unknown"
                if age <= 18: return "Subjunior"
                elif age <= 23: return "Junior"
                elif age <= 39: return "Open"
                else: return "Masters"
                
            valid_df['Age_Bin'] = valid_df['Age'].apply(get_age_group)
            grouped = valid_df.groupby(['WeightClass', 'Age_Bin'], observed=True)
            for (wc, age_bin), group in grouped:
                if group.empty: continue
                q_vals = group[target_metric].quantile(quantiles)
                if len(q_vals) < 5: continue
                results.append({
                    "Weight Class": f"{int(wc) if wc%1==0 else wc} {unit}",
                    "Age Group": str(age_bin),
                    "Beginner": q_vals.iloc[0] * mult,
                    "Novice": q_vals.iloc[1] * mult,
                    "Intermediate": q_vals.iloc[2] * mult,
                    "Advanced": q_vals.iloc[3] * mult,
                    "Elite": q_vals.iloc[4] * mult,
                })
            
            if not results:
                st.warning("Not enough data to generate strength standards with the selected groupings.")
                return
            res_df = pd.DataFrame(results)
            age_order = {"Subjunior": 1, "Junior": 2, "Open": 3, "Masters": 4, "Unknown": 5}
            res_df["age_sort"] = res_df["Age Group"].map(age_order)
            res_df["wc_sort"] = res_df["Weight Class"].str.replace(f" {unit}", "").apply(sort_weight_class)
            res_df = res_df.sort_values(by=["wc_sort", "age_sort"]).drop(columns=["age_sort", "wc_sort"])
        else:
            grouped = valid_df.groupby('WeightClass')
            for wc, group in grouped:
                if group.empty: continue
                q_vals = group[target_metric].quantile(quantiles)
                if len(q_vals) < 5: continue
                results.append({
                    "Weight Class": f"{int(wc) if wc%1==0 else wc} {unit}",
                    "Beginner": q_vals.iloc[0] * mult,
                    "Novice": q_vals.iloc[1] * mult,
                    "Intermediate": q_vals.iloc[2] * mult,
                    "Advanced": q_vals.iloc[3] * mult,
                    "Elite": q_vals.iloc[4] * mult,
                })
                
            if not results:
                st.warning("Not enough data to generate strength standards for this group.")
                return
            res_df = pd.DataFrame(results)
            res_df["wc_sort"] = res_df["Weight Class"].str.replace(f" {unit}", "").apply(sort_weight_class)
            res_df = res_df.sort_values(by="wc_sort").drop(columns=["wc_sort"])

        for c in ["Beginner", "Novice", "Intermediate", "Advanced", "Elite"]: res_df[c] = res_df[c].round(1)

        st.dataframe(
            res_df,
            use_container_width=True,
            column_config={
                "Beginner": st.column_config.NumberColumn("Beginner", help="Better than 16.6%"),
                "Novice": st.column_config.NumberColumn("Novice", help="Better than 33.3%"),
                "Intermediate": st.column_config.NumberColumn("Intermediate", help="Better than 50.0%"),
                "Advanced": st.column_config.NumberColumn("Advanced", help="Better than 66.6%"),
                "Elite": st.column_config.NumberColumn("Elite", help="Better than 83.3%"),
            }
        )

def render_competition_section(df_src, cfg):
    comp_tabs = st.tabs(["Athletes Chart", "Category Difficulty", "Results Table"])
    
    with comp_tabs[0]:
        st.write("Compare categories or athletes based on performance.")
        
        valid_wcs = sorted(df_src['WeightClass'].dropna().astype(str).unique(), key=sort_weight_class)
        sel_comp_wcs = st.multiselect("Select Weight Classes for Chart", valid_wcs, default=valid_wcs, key=f"{cfg['group_name']}_comp_wc")
        comp_metric = st.selectbox("Select Metric", ["Total", "Dots", "Wilks", "GL Points"], key=f"{cfg['group_name']}_comp_metric")
        
        c_df = df_src[df_src['WeightClass'].astype(str).isin(sel_comp_wcs)].copy()
        if not c_df.empty:
            fig = px.scatter(
                c_df, x="Bodyweight", y=comp_metric, color="WeightClass",
                hover_data=["Name", "Total", "Squat", "Bench", "Deadlift"],
                title=f"{comp_metric} Distribution by Selected Categories"
            )
            st.plotly_chart(fig, use_container_width=True)
            
    with comp_tabs[1]:
        st.write("Weight categories ranked by difficulty based on group performance.")
        diff_metric = st.selectbox("Points System", ["Dots", "Wilks", "GL Points"], key=f"{cfg['group_name']}_diff_met")
        agg_type = st.selectbox("Comparison Method", ["Average", "Median", "Placement (Top N)", "All Points (Scatter)"], key=f"{cfg['group_name']}_agg_type")
        
        place_n = 1
        if agg_type == "Placement (Top N)":
            place_n = st.selectbox("Select Placement", [1, 2, 3, 4, 5, 6, 7, 8], key=f"{cfg['group_name']}_place_n")
        
        diff_df = df_src.dropna(subset=['WeightClass', diff_metric]).copy()
        diff_df['WeightClass'] = diff_df['WeightClass'].astype(str)
        
        if agg_type in ["Average", "Median", "Placement (Top N)"]:
            if agg_type == "Average":
                agg_diff = diff_df.groupby('WeightClass')[diff_metric].mean().reset_index()
                y_col = diff_metric
            elif agg_type == "Median":
                agg_diff = diff_df.groupby('WeightClass')[diff_metric].median().reset_index()
                y_col = diff_metric
            else:
                def get_nth(series, n):
                    sorted_vals = series.nlargest(n)
                    return sorted_vals.min() if len(sorted_vals) >= n else np.nan
                agg_diff = diff_df.groupby('WeightClass')[diff_metric].apply(lambda x: get_nth(x, place_n)).reset_index()
                y_col = diff_metric

            agg_diff['SortKey'] = agg_diff['WeightClass'].apply(sort_weight_class)
            agg_diff = agg_diff.sort_values('SortKey').drop(columns=['SortKey'])
            
            fig_diff = px.line(
                agg_diff, x="WeightClass", y=y_col,
                markers=True, title=f"Difficulty Level per Category - {agg_type} ({diff_metric})"
            )
        else:
            diff_df['SortKey'] = diff_df['WeightClass'].apply(sort_weight_class)
            diff_df = diff_df.sort_values('SortKey')
            fig_diff = px.scatter(
                diff_df, x="WeightClass", y=diff_metric, color="WeightClass",
                hover_data=["Name", "Place", "Total"],
                title=f"Difficulty Level per Category - All Points ({diff_metric})"
            )
            
        st.plotly_chart(fig_diff, use_container_width=True)

    with comp_tabs[2]:
        st.write("Dynamically display weight category results.")
        t_wcs = sorted(df_src['WeightClass'].dropna().astype(str).unique(), key=sort_weight_class)
        sel_t_wcs = st.multiselect("Select Categories to Display in Table", t_wcs, default=t_wcs, key=f"{cfg['group_name']}_t_wcs")
        
        t_df = df_src[df_src['WeightClass'].astype(str).isin(sel_t_wcs)].copy()
        
        default_cols = ["Place", "Name", "Age", "WeightClass", "Total", "Dots", "Squat1Kg", "Squat2Kg", "Squat3Kg", "Bench1Kg", "Bench2Kg", "Bench3Kg", "Deadlift1Kg", "Deadlift2Kg", "Deadlift3Kg"]
        available_cols = ["Place", "Name", "Age", "AgeCategory", "WeightClass", "Bodyweight", "Total", "Dots", "Wilks", "GL Points", 
                            "Squat1Kg", "Squat2Kg", "Squat3Kg", "Bench1Kg", "Bench2Kg", "Bench3Kg", 
                            "Deadlift1Kg", "Deadlift2Kg", "Deadlift3Kg", "Date"]
        
        sel_cols = st.multiselect("Columns", available_cols, default=default_cols, key=f"{cfg['group_name']}_t_cols")
        
        group_by_age = st.checkbox("Group by Age Category", key=f"{cfg['group_name']}_t_group_age")
        
        display_t_df = t_df.copy()
        for c in ["Squat1Kg", "Squat2Kg", "Squat3Kg", "Bench1Kg", "Bench2Kg", "Bench3Kg", "Deadlift1Kg", "Deadlift2Kg", "Deadlift3Kg"]:
            if c in display_t_df.columns: display_t_df[c] = display_t_df[c].apply(format_attempt)
        for c in ["Total", "Bodyweight"]:
            if c in display_t_df.columns: display_t_df[c] = (display_t_df[c] * mult).round(1)
        for c in ["Dots", "Wilks", "GL Points"]:
            if c in display_t_df.columns: display_t_df[c] = display_t_df[c].round(2)
            
        if group_by_age and "AgeCategory" in display_t_df.columns:
            for age_cat in display_t_df["AgeCategory"].dropna().unique():
                st.markdown(f"#### {age_cat}")
                cat_df = display_t_df[display_t_df["AgeCategory"] == age_cat]
                display_cols = [c for c in sel_cols if c in cat_df.columns]
                render_dataframe(cat_df[display_cols], key_prefix=f"{cfg['group_name']}_t_{age_cat}")
        else:
            display_cols = [c for c in sel_cols if c in display_t_df.columns]
            render_dataframe(display_t_df[display_cols], key_prefix=f"{cfg['group_name']}_t")

def render_group_tab(df_src, cfg, exact_bw_target=None):
    if df_src is None or df_src.empty:
        st.warning(f"No data found for {cfg['group_name']}.")
        return
    st.success(f"Loaded {len(df_src):,} athletes in {cfg['group_name']}.")
    
    st.session_state.score_sys = st.radio("Select Points System for Analysis (Tables and Charts):", ["Dots", "Wilks", "GL Points"], horizontal=True, key=f"{cfg['group_name']}_score_sys")
    sys_col = st.session_state.score_sys
    
    render_strength_standards(df_src, cfg, exact_bw_target)
    
    st.subheader("Distribution Statistics", anchor=False)
    sc1, sc2, sc3 = st.columns([1, 1, 2])
    with sc1:
        default_target_metric = cfg["metric"] if cfg["metric"] in ["Total", sys_col, "Squat", "Bench", "Deadlift"] else "Total"
        target_met = st.selectbox("Select Metric", ["Total", sys_col, "Squat", "Bench", "Deadlift"], index=["Total", sys_col, "Squat", "Bench", "Deadlift"].index(default_target_metric), key=f"{cfg['group_name']}_tgtmet")
        p_val = st.number_input("Better than %", min_value=0, max_value=100, value=90, step=1, key=f"{cfg['group_name']}_pval")
    
    valid_scores = df_src[df_src[target_met] > 0][target_met].dropna()
    if not valid_scores.empty:
        c_mult = 1.0 if target_met in ["Dots", "Wilks", "GL Points"] else mult
        avg_val = valid_scores.mean() * c_mult
        med_val = valid_scores.median() * c_mult
        p_req = np.percentile(valid_scores, p_val) * c_mult
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Average", f"{avg_val:.1f}")
        m2.metric("Median", f"{med_val:.1f}")
        m3.metric(f"Required to be better than {p_val}%", f"{p_req:.1f}")
    else:
        st.info("No valid records for this metric.")
        
    if "SBD" in cfg.get("events", []):
        sbd_df = df_src[(df_src["Squat"] > 0) & (df_src["Bench"] > 0) & (df_src["Deadlift"] > 0) & (df_src["Total"] > 0)].copy()
        if not sbd_df.empty:
            sq_prop = (sbd_df["Squat"] / sbd_df["Total"]).mean() * 100
            bp_prop = (sbd_df["Bench"] / sbd_df["Total"]).mean() * 100
            dl_prop = (sbd_df["Deadlift"] / sbd_df["Total"]).mean() * 100
            st.caption(f"**Group Proportions in Total:** Squat {sq_prop:.1f}% | Bench {bp_prop:.1f}% | Deadlift {dl_prop:.1f}%")

    st.subheader("Distribution Chart", anchor=False)
    default_dist = cfg["metric"] if cfg["metric"] in ["Total", "Dots", "Wilks", "GL Points", "Squat", "Bench", "Deadlift", "Bodyweight"] else "Total"
    dist_met = st.selectbox("Chart Metric", ["Total", "Dots", "Wilks", "GL Points", "Squat", "Bench", "Deadlift", "Bodyweight", "Age", "Experience", "Equipment"], index=["Total", "Dots", "Wilks", "GL Points", "Squat", "Bench", "Deadlift", "Bodyweight", "Age", "Experience", "Equipment"].index(default_dist), key=f"{cfg['group_name']}_distmet")
    
    if dist_met == "Equipment":
        fig_dist = px.histogram(df_src.dropna(subset=["Equipment"]), x="Equipment", color_discrete_sequence=["#1565C0"])
    else:
        dist_df = df_src[df_src[dist_met] > 0] if dist_met not in ["Dots", "Wilks", "GL Points", "Bodyweight", "Age", "Experience"] else df_src.dropna(subset=[dist_met])
        if not dist_df.empty:
            d_mult = mult if dist_met not in ["Dots", "Wilks", "GL Points", "Age", "Experience"] else 1.0
            fig_dist = px.histogram(dist_df, x=dist_df[dist_met]*d_mult, nbins=40, marginal="box", color_discrete_sequence=["#1565C0"])
            
    st.plotly_chart(fig_dist, use_container_width=True)

    st.subheader("Dynamic Correlation", anchor=False)
    c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
    opts = ["Dots", "Wilks", "GL Points", "Total", "Squat", "Bench", "Deadlift", "Bodyweight", "Age", "Meet#", "Experience"]
    c_opts = ["None", "Country", "Federation", "ParentFederation", "Age", "Sex", "Equipment"]
    if cfg.get("tested", "Any") == "Any": c_opts.append("Tested")
        
    with c1: x_ax = st.selectbox("Horizontal Axis", opts, index=opts.index("Bodyweight"), key=f"{cfg['group_name']}_x")
    with c2: y_ax = st.selectbox("Vertical Axis", opts, index=opts.index(sys_col), key=f"{cfg['group_name']}_y")
    with c3: c_ax = st.selectbox("Color by", c_opts, key=f"{cfg['group_name']}_c")
    with c4: show_trend = st.checkbox("Trendline", key=f"{cfg['group_name']}_trend")
    
    df_plot = df_src.dropna(subset=[x_ax, y_ax]).copy()
    if c_ax != "None":
        if pd.api.types.is_numeric_dtype(df_plot[c_ax]): df_plot[c_ax] = df_plot[c_ax].fillna(0)
        else: df_plot[c_ax] = df_plot[c_ax].fillna("Unknown")
            
    if not df_plot.empty:
        try:
            import statsmodels.api as sm
            trend = "ols" if show_trend else None
        except ImportError:
            trend = None
        fig_custom = px.scatter(
            df_plot, x=x_ax, y=y_ax, color=c_ax if c_ax != "None" else None, 
            hover_data=["Name", "Squat", "Bench", "Deadlift", "Total", sys_col, "Bodyweight", "Date", "Event", "MeetName"], 
            trendline=trend
        )
        st.plotly_chart(fig_custom, use_container_width=True)

    st.subheader("Leaderboard Table", anchor=False)
    df_table = df_src.copy()
    for col in ["Bodyweight", "Squat", "Bench", "Deadlift", "Total"]: 
        df_table[col] = (df_table[col] * mult).round(1)
        
    for col in ["Dots", "Wilks", "GL Points"]: df_table[col] = df_table[col].round(2)
    df_table["Experience"] = df_table["Experience"].round(1)
    df_table["Date"] = pd.to_datetime(df_table["Date"], errors='coerce').dt.strftime("%Y-%m-%d")
    
    all_table_cols = ["Rank", "Name", "Sex", "Country", "Age", "Bodyweight", "Event", "Squat", "Bench", "Deadlift", "Total", sys_col, "Date", "MeetName", "Meet#", "Experience", "Equipment", "Tested", "Federation", "ParentFederation", "Place"]
    default_table_cols = ["Rank", "Name", "Age", "Bodyweight", "Date", "MeetName", "Squat", "Bench", "Deadlift", "Total", sys_col, "Place"]
    
    selected_leaderboard_cols = st.multiselect("Customize columns:", all_table_cols, default=default_table_cols, key=f"{cfg['group_name']}_l_cols")
    
    df_display = df_table.copy()
    df_display.insert(0, "Rank", range(1, len(df_display) + 1))
    display_cols = [c for c in selected_leaderboard_cols if c in df_display.columns]
    
    render_dataframe(df_display[display_cols], key_prefix=f"{cfg['group_name']}_l", set_index="Rank" if "Rank" in display_cols else None)

def fetch_athlete_data(name, _data):
    conn = get_connection(_data)
    q = """
        SELECT p.*,
               ROW_NUMBER() OVER(PARTITION BY p.Name, IFNULL(p.Country, '') ORDER BY p.Date) as MeetNum,
               MIN(p.Date) OVER(PARTITION BY p.Name, IFNULL(p.Country, '')) as CareerStart
        FROM clean_db p
        WHERE p.Name = ?
        ORDER BY p.Date ASC
    """
    df_res = conn.execute(q, [name]).df()
    if not df_res.empty:
        df_res['Date'] = pd.to_datetime(df_res['Date'], errors='coerce')
        df_res['CareerStart'] = pd.to_datetime(df_res['CareerStart'], errors='coerce')
        df_res['LongevityToMeet'] = (df_res['Date'] - df_res['CareerStart']).dt.days / 365.25
        df_res['EstBirthYear'] = df_res['Date'].dt.year - df_res['Age']
        df_res.rename(columns={
            "TotalKg": "Total", "Best3SquatKg": "Squat", "Best3BenchKg": "Bench", 
            "Best3DeadliftKg": "Deadlift", "BodyweightKg": "Bodyweight", 
            "MeetNum": "Meet#", "LongevityToMeet": "Experience", "WeightClassKg": "WeightClass",
            "GLPoints": "GL Points"
        }, inplace=True)
        df_res["NormMeet"] = df_res["MeetName"].apply(normalize_meet_name)
    return df_res

def apply_athlete_filters(df_src, key_prefix):
    filtered = df_src.copy()
    if filtered.empty: return filtered
    
    st.markdown("#### Filter Athlete Career")
    cols = st.columns(4)
    filter_keys = ["WeightClass", "Equipment", "Tested", "Federation"]
    
    for i, col in enumerate(filter_keys):
        unique_vals = filtered[col].dropna().unique().tolist()
        if len(unique_vals) > 1:
            sel = cols[i%4].multiselect(col, unique_vals, default=unique_vals, key=f"{key_prefix}_filt_{col}")
            filtered = filtered[filtered[col].isin(sel)]
    return filtered

# ---------------------------------------------------------------------------
# UI LOGIC & INPUTS
# ---------------------------------------------------------------------------
if analysis_mode == "Group":
    cfg_a = render_group_filters("a", "Group 1", default_countries=[])
    if st.button("Run Analysis", use_container_width=True):
        check_weight_conflict(cfg_a)
        st.session_state.submit_clicked = True
        st.session_state.cfg_a_req = cfg_a

elif analysis_mode == "Result vs group":
    st.markdown("**Your Data**")
    
    col_bw, col_sq, col_bn, col_dl = st.columns(4)
    with col_bw: 
        u_bw = st.number_input(f"Bodyweight ({unit})", value=70.0, step=0.5)
        u_sex = st.selectbox("Sex", ["M", "F"])
    with col_sq: 
        c1, c2 = st.columns(2)
        u_sq_w = c1.number_input(f"Squat ({unit})", value=0.0, step=5.0)
        u_sq_r = c2.number_input("Reps SQ", value=1, min_value=1)
        if u_sq_r > 1: st.caption(f"Est 1RM: {calc_1rm(u_sq_w, u_sq_r):.1f} {unit}")
    with col_bn: 
        c1, c2 = st.columns(2)
        u_bn_w = c1.number_input(f"Bench ({unit})", value=0.0, step=2.5)
        u_bn_r = c2.number_input("Reps BN", value=1, min_value=1)
        if u_bn_r > 1: st.caption(f"Est 1RM: {calc_1rm(u_bn_w, u_bn_r):.1f} {unit}")
    with col_dl: 
        c1, c2 = st.columns(2)
        u_dl_w = c1.number_input(f"Deadlift ({unit})", value=0.0, step=5.0)
        u_dl_r = c2.number_input("Reps DL", value=1, min_value=1)
        if u_dl_r > 1: st.caption(f"Est 1RM: {calc_1rm(u_dl_w, u_dl_r):.1f} {unit}")
        
    u_squat = calc_1rm(u_sq_w, u_sq_r)
    u_bench = calc_1rm(u_bn_w, u_bn_r)
    u_deadlift = calc_1rm(u_dl_w, u_dl_r)
    
    u_squat_kg = u_squat/2.20462 if use_lbs else u_squat
    u_bench_kg = u_bench/2.20462 if use_lbs else u_bench
    u_deadlift_kg = u_deadlift/2.20462 if use_lbs else u_deadlift
    u_bw_kg = u_bw/2.20462 if use_lbs else u_bw
    u_tot_kg = u_squat_kg + u_bench_kg + u_deadlift_kg
    
    st.markdown("---")
    cfg_a = render_group_filters("a", "Reference Group", default_countries=[])
    if st.button("Run Analysis", use_container_width=True):
        check_weight_conflict(cfg_a)
        st.session_state.submit_clicked = True
        st.session_state.cfg_a_req = cfg_a

elif analysis_mode == "Athlete vs group":
    st.markdown("**Select Athlete**")
    render_athlete_selectbox("Search Athlete (Type to search):", "ath_selected_vg", df)
    st.markdown("---")
    cfg_g = render_group_filters("avg", "Reference Group", default_countries=[])
    if st.button("Run Analysis", use_container_width=True):
        check_weight_conflict(cfg_g)
        st.session_state.submit_clicked = True
        st.session_state.cfg_a_req = cfg_g

elif analysis_mode == "Group vs group":
    cfg_a = render_group_filters("a", "Group A", default_countries=[])
    st.markdown("---")
    cfg_b = render_group_filters("b", "Group B", default_equip=["Raw"], default_countries=["France"])
    if st.button("Run Analysis", use_container_width=True):
        check_weight_conflict(cfg_a)
        check_weight_conflict(cfg_b)
        st.session_state.submit_clicked = True
        st.session_state.cfg_a_req = cfg_a
        st.session_state.cfg_b_req = cfg_b

elif analysis_mode == "Athlete":
    render_athlete_selectbox("Search Athlete (Type to search):", "ath_selected", df)
    if st.button("Run Analysis", use_container_width=True):
        st.session_state.submit_clicked = True

elif analysis_mode == "Athlete vs athlete":
    col_s1, col_s2 = st.columns(2)
    with col_s1: render_athlete_selectbox("Select Athlete A:", "ath_a_sel", df)
    with col_s2: render_athlete_selectbox("Select Athlete B:", "ath_b_sel", df)
        
    vs_event_opt = st.selectbox("Compare Lift Type (Event):", ["Any", "SBD", "S", "B", "D", "SB", "BD", "SD"], index=0)
    if st.button("Run Analysis", use_container_width=True):
        st.session_state.submit_clicked = True
        st.session_state.vs_event_opt = vs_event_opt

elif analysis_mode == "Competition Analysis":
    st.markdown("## Competition Analysis")
    comp_search = st.text_input("Search Competition Name (Type at least 3 characters):", key="comp_search_val")
    
    if comp_search and len(comp_search) >= 3:
        conn = get_connection(df)
        comps_df = conn.execute("SELECT DISTINCT MeetName FROM clean_db WHERE MeetName ILIKE ?", [f"%{comp_search}%"]).df()
        if comps_df is not None and not comps_df.empty:
            comp_list = sorted(comps_df['MeetName'].dropna().unique().tolist())
            sel_comp = st.selectbox("Select Competition:", comp_list, key="sel_comp_ui")
            
            years_df = conn.execute("SELECT DISTINCT extract(year from Date) as Year FROM clean_db WHERE MeetName = ? ORDER BY Year DESC", [sel_comp]).df()
            years = years_df['Year'].dropna().astype(int).astype(str).tolist()
            year_opt = st.selectbox("Select Year:", ["All Years (Compare)"] + years, key="sel_comp_year_ui")
            
            if st.button("Load Competition Data", use_container_width=True):
                st.session_state.submit_clicked = True
                st.session_state.sel_comp = sel_comp
                st.session_state.sel_comp_year = year_opt
        else:
            st.warning("No competitions found matching the search.")

elif analysis_mode == "Calculators":
    st.session_state.submit_clicked = False


# ---------------------------------------------------------------------------
# ANALYSIS DISPLAY
# ---------------------------------------------------------------------------
if st.session_state.submit_clicked:
    if analysis_mode in ["Group", "Result vs group"]:
        with st.spinner("Crunching data with DuckDB..."):
            st.session_state.df_a = run_group_analysis(st.session_state.cfg_a_req, df)
            st.session_state.cfg_a = st.session_state.cfg_a_req
            
    elif analysis_mode == "Athlete vs group":
        ath_name = st.session_state.get("ath_selected_vg")
        if ath_name and isinstance(ath_name, str):
            with st.spinner("Fetching athlete..."): 
                st.session_state.ath_df = fetch_athlete_data(ath_name, df)
        with st.spinner("Crunching group data..."):
            st.session_state.df_a = run_group_analysis(st.session_state.cfg_a_req, df)
            st.session_state.cfg_a = st.session_state.cfg_a_req
                
    elif analysis_mode == "Group vs group":
        with st.spinner(f"Crunching {st.session_state.cfg_a_req['group_name']}..."): 
            st.session_state.df_a = run_group_analysis(st.session_state.cfg_a_req, df)
        with st.spinner(f"Crunching {st.session_state.cfg_b_req['group_name']}..."): 
            st.session_state.df_b = run_group_analysis(st.session_state.cfg_b_req, df)
        st.session_state.cfg_a = st.session_state.cfg_a_req
        st.session_state.cfg_b = st.session_state.cfg_b_req

    elif analysis_mode == "Athlete":
        ath_name = st.session_state.get("ath_selected")
        if ath_name and isinstance(ath_name, str):
            with st.spinner("Fetching athlete..."): 
                st.session_state.ath_df = fetch_athlete_data(ath_name, df)

    elif analysis_mode == "Athlete vs athlete":
        a_name = st.session_state.get("ath_a_sel")
        b_name = st.session_state.get("ath_b_sel")
        if a_name and isinstance(a_name, str) and b_name and isinstance(b_name, str):
            with st.spinner("Fetching Athlete A..."): 
                st.session_state.ath_df_a = fetch_athlete_data(a_name, df)
            with st.spinner("Fetching Athlete B..."): 
                st.session_state.ath_df_b = fetch_athlete_data(b_name, df)
            
    elif analysis_mode == "Competition Analysis":
        sel_comp = st.session_state.get("sel_comp")
        sel_year = st.session_state.get("sel_comp_year")
        if sel_comp:
            with st.spinner("Fetching competition data..."):
                conn = get_connection(df)
                if sel_year and sel_year != "All Years (Compare)":
                    df_comp = conn.execute("SELECT * FROM clean_db WHERE MeetName = ? AND extract(year from Date) = ?", [sel_comp, int(sel_year)]).df()
                    comp_display_name = f"{sel_comp} ({sel_year})"
                else:
                    df_comp = conn.execute("SELECT * FROM clean_db WHERE MeetName = ?", [sel_comp]).df()
                    comp_display_name = f"{sel_comp} (All Years)"

                if df_comp is not None and not df_comp.empty:
                    df_comp["Age"] = df_comp["Age"].fillna(0).astype(int)
                    df_comp['Date'] = pd.to_datetime(df_comp['Date'], errors='coerce')

                    df_comp.rename(columns={
                        "TotalKg": "Total", "Best3SquatKg": "Squat", "Best3BenchKg": "Bench",
                        "Best3DeadliftKg": "Deadlift", "BodyweightKg": "Bodyweight",
                        "WeightClassKg": "WeightClass", "GLPoints": "GL Points"
                    }, inplace=True)

                    for col in ["Squat", "Bench", "Deadlift", "Total", "Dots", "Wilks", "GL Points"]:
                        if col in df_comp.columns: df_comp[col] = df_comp[col].fillna(0)

                    st.session_state.df_comp = df_comp
                    st.session_state.comp_name = comp_display_name

    st.session_state.submit_clicked = False 


show_analysis = False
if analysis_mode in ["Group", "Result vs group"] and not st.session_state.df_a.empty: show_analysis = True
elif analysis_mode == "Athlete vs group" and not st.session_state.df_a.empty and not st.session_state.ath_df.empty: show_analysis = True
elif analysis_mode == "Group vs group" and not st.session_state.df_a.empty and not st.session_state.df_b.empty: show_analysis = True
elif analysis_mode == "Athlete" and not st.session_state.ath_df.empty: show_analysis = True
elif analysis_mode == "Athlete vs athlete" and not st.session_state.ath_df_a.empty and not st.session_state.ath_df_b.empty: show_analysis = True
elif analysis_mode == "Competition Analysis" and st.session_state.get("df_comp") is not None and not st.session_state.df_comp.empty: show_analysis = True

if analysis_mode != "Calculators" and show_analysis:
    st.header("Analysis", anchor=False)
    st.markdown("---")

if analysis_mode in ["Group", "Result vs group"] and not st.session_state.df_a.empty:
    df_a = st.session_state.df_a
    cfg_a_mem = st.session_state.get("cfg_a", {})
    if analysis_mode == "Result vs group" and not df_a.empty:
        st.subheader(f"Your Percentile vs {cfg_a_mem.get('group_name', 'Group')}", anchor=False)
        c1, c2, c3, c4, c5 = st.columns(5)
        
        def get_pctl(val, col):
            pool = df_a[df_a[col]>0][col].dropna()
            return f"Better than {((pool < val).mean() * 100):.1f}%" if len(pool) else "N/A"
            
        sys_metric = st.session_state.score_sys
        u_score = 0.0
        if u_bw_kg > 0:
            if sys_metric == "Dots":
                u_score = u_tot_kg * (500.0 / (-0.000001093 * (u_bw_kg ** 4) + 0.0007391293 * (u_bw_kg ** 3) -0.1918759221 * (u_bw_kg ** 2) + 24.0900756 * u_bw_kg - 307.75076)) if u_sex=="M" else u_tot_kg * (500.0 / (-0.000010706 * (u_bw_kg ** 4) + 0.005158568 * (u_bw_kg ** 3) -0.92501065 * (u_bw_kg ** 2) + 75.323049 * u_bw_kg - 516.39869))
            elif sys_metric == "Wilks":
                if u_sex == "M":
                    denom = -216.0475144 + 16.2606339*u_bw_kg -0.002388645*(u_bw_kg**2) -0.00113732*(u_bw_kg**3) + 7.01863E-06*(u_bw_kg**4) -1.291E-08*(u_bw_kg**5)
                else:
                    denom = 594.3174777 -27.23842536*u_bw_kg + 0.821122268*(u_bw_kg**2) -0.009307339*(u_bw_kg**3) + 4.73158E-05*(u_bw_kg**4) -9.054E-08*(u_bw_kg**5)
                u_score = u_tot_kg * (500.0 / denom) if denom != 0 else 0
            else:
                denom = (1199.72839 - 925.40462 * np.exp(-0.00510531 * u_bw_kg)) if u_sex=="M" else (610.79046 - 451.04414 * np.exp(-0.00735665 * u_bw_kg))
                u_score = u_tot_kg * (100.0 / denom) if denom != 0 else 0
        
        c1.metric(f"Squat ({u_squat/u_bw:.1f}x BW)", fmt(u_squat), get_pctl(u_squat_kg, 'Squat'))
        c2.metric(f"Bench ({u_bench/u_bw:.1f}x BW)", fmt(u_bench), get_pctl(u_bench_kg, 'Bench'))
        c3.metric(f"Deadlift ({u_deadlift/u_bw:.1f}x BW)", fmt(u_deadlift), get_pctl(u_deadlift_kg, 'Deadlift'))
        c4.metric(f"Total ({u_tot_kg*mult/u_bw:.1f}x BW)", fmt(u_tot_kg*mult), get_pctl(u_tot_kg, 'Total'))
        c5.metric(sys_metric, fmt(u_score), get_pctl(u_score, sys_metric))
        
        if u_tot_kg > 0:
            u_sq_prop = (u_squat_kg / u_tot_kg) * 100
            u_bp_prop = (u_bench_kg / u_tot_kg) * 100
            u_dl_prop = (u_deadlift_kg / u_tot_kg) * 100
            st.caption(f"**Your Proportions in Total:** Squat {u_sq_prop:.1f}% | Bench {u_bp_prop:.1f}% | Deadlift {u_dl_prop:.1f}%")
            
        st.markdown("---")
        if not df_a.empty: render_group_tab(df_a, cfg_a_mem, exact_bw_target=u_bw)
        
    elif analysis_mode == "Group":
        if not df_a.empty: render_group_tab(df_a, cfg_a_mem)

elif analysis_mode == "Athlete vs group" and not st.session_state.df_a.empty and not st.session_state.ath_df.empty:
    ath_df = st.session_state.ath_df
    df_a = st.session_state.df_a
    cfg_g = st.session_state.get("cfg_a", {})
    
    st.subheader(f"{ath_df['Name'].iloc[0]} vs {cfg_g.get('group_name', 'Group')}", anchor=False)

    pr_total = ath_df["Total"].max()
    pr_sq = ath_df["Squat"].max()
    pr_bn = ath_df["Bench"].max()
    pr_dl = ath_df["Deadlift"].max()
    sys_metric = st.session_state.score_sys
    pr_score = ath_df[sys_metric].max() if sys_metric in ath_df.columns else ath_df["Dots"].max()

    def get_pctl_ath(val, col):
        if pd.isna(val) or val == 0: return "N/A"
        pool = df_a[df_a[col]>0][col].dropna()
        return f"Better than {((pool < val).mean() * 100):.1f}%" if len(pool) else "N/A"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("PR Squat", fmt(pr_sq*mult), get_pctl_ath(pr_sq, 'Squat'))
    c2.metric("PR Bench", fmt(pr_bn*mult), get_pctl_ath(pr_bn, 'Bench'))
    c3.metric("PR Deadlift", fmt(pr_dl*mult), get_pctl_ath(pr_dl, 'Deadlift'))
    c4.metric("PR Total", fmt(pr_total*mult), get_pctl_ath(pr_total, 'Total'))
    c5.metric(f"PR {sys_metric}", fmt(pr_score), get_pctl_ath(pr_score, sys_metric))

    st.markdown("---")
    render_group_tab(df_a, cfg_g)

elif analysis_mode == "Group vs group":
    df_a, df_b = st.session_state.df_a, st.session_state.df_b
    cfg_a_mem, cfg_b_mem = st.session_state.get("cfg_a", {}), st.session_state.get("cfg_b", {})
    
    if not df_a.empty and not df_b.empty:
        st.session_state.score_sys = st.radio("Select Points System:", ["Dots", "Wilks", "GL Points"], horizontal=True)
        name_a, name_b = cfg_a_mem.get("group_name", "Group A"), cfg_b_mem.get("group_name", "Group B")
        df_a["Group Label"] = name_a
        df_b["Group Label"] = name_b
        combined = pd.concat([df_a, df_b])
        
        st.subheader(f"Head-to-Head: {name_a} vs {name_b}", anchor=False)
        st.columns(2)
        s1, s2 = st.columns(2)
        s1.metric(f"{name_a} Athletes", len(df_a), f"Avg Total: {(df_a['Total'].mean()*mult):.1f}")
        s2.metric(f"{name_b} Athletes", len(df_b), f"Avg Total: {(df_b['Total'].mean()*mult):.1f}")
        
        fig = px.histogram(combined, x=combined["Total"]*mult, color="Group Label", barmode="overlay", title="Total Distribution Overlay")
        st.plotly_chart(fig, use_container_width=True)
        
        fig_s = px.scatter(combined, x="Bodyweight", y=st.session_state.score_sys, color="Group Label", title=f"{st.session_state.score_sys} vs Bodyweight Overlay", hover_data=["Name"])
        st.plotly_chart(fig_s, use_container_width=True)
        
        render_competition_section(combined, {"group_name": f"{name_a} vs {name_b}"})

elif analysis_mode == "Athlete":
    ath_df_raw = st.session_state.ath_df
    if not ath_df_raw.empty:
        ath_df = apply_athlete_filters(ath_df_raw, "ath_single")
        st.markdown("---")
        st.subheader(f"Profile: {ath_df['Name'].iloc[0]}", anchor=False)
        
        pr_total = ath_df["Total"].max()
        pr_dots = ath_df["Dots"].max()
        pr_squat = ath_df["Squat"].max()
        pr_bench = ath_df["Bench"].max()
        pr_deadlift = ath_df["Deadlift"].max()
        longevity_yrs = ath_df["Experience"].max()
        
        est_by = ath_df['EstBirthYear'].median()
        est_by_str = f"{int(est_by)}" if pd.notna(est_by) else "Unknown"

        m1, m2, m3, m4, m5, m6, m7, m8 = st.columns(8)
        m1.metric("Est. Birth Year", est_by_str)
        m2.metric("Meets Total", len(ath_df))
        m3.metric("Career (Years)", f"{longevity_yrs:.1f}")
        m4.metric("PR Squat", f"{pr_squat * mult:.1f}")
        m5.metric("PR Bench", f"{pr_bench * mult:.1f}")
        m6.metric("PR Deadlift", f"{pr_deadlift * mult:.1f}")
        m7.metric("PR Total", f"{pr_total * mult:.1f}")
        m8.metric("PR DOTS", f"{pr_dots:.1f}")
        
        st.markdown("---")
        st.subheader("Performance History", anchor=False)
        metric_opts = ["Total", "Dots", "Wilks", "GL Points", "Bodyweight", "Squat", "Bench", "Deadlift"]
        hist_metrics = st.multiselect("Select metrics for history chart", metric_opts, default=["Total"])
        
        if hist_metrics:
            df_plot_ath = ath_df.copy()
            if any(m in hist_metrics for m in ["Total", "Dots", "Wilks", "GL Points"]):
                df_plot_ath = df_plot_ath[df_plot_ath['Event'].isin(['SBD', 'Full Power'])]
                
            df_plot_ath["Date"] = pd.to_datetime(df_plot_ath["Date"])
            fig = go.Figure()
            for met in hist_metrics:
                fig.add_trace(go.Scatter(x=df_plot_ath['Date'], y=df_plot_ath[met]* (mult if met not in ["Dots", "Wilks", "GL Points"] else 1), mode='lines+markers', name=met, hovertext=df_plot_ath['MeetName']))
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Dynamic Correlation", anchor=False)
        c1_a, c2_a, c3_a = st.columns(3)
        with c1_a: x_ax_a = st.selectbox("Horizontal Axis", metric_opts + ["Meet#", "Experience", "Age"], index=0, key="ath_x")
        with c2_a: y_ax_a = st.selectbox("Vertical Axis", metric_opts + ["Meet#", "Experience", "Age"], index=1, key="ath_y")
        with c3_a: c_ax_a = st.selectbox("Color by", ["None", "Equipment", "Federation", "ParentFederation", "Place", "Event"], key="ath_c")
        
        ath_dyn_df = ath_df.dropna(subset=[x_ax_a, y_ax_a]).copy()
        if c_ax_a != "None":
            if pd.api.types.is_numeric_dtype(ath_dyn_df[c_ax_a]): ath_dyn_df[c_ax_a] = ath_dyn_df[c_ax_a].fillna(0)
            else: ath_dyn_df[c_ax_a] = ath_dyn_df[c_ax_a].fillna("Unknown")
                
        if not ath_dyn_df.empty:
            fig_ath = px.scatter(
                ath_dyn_df, x=x_ax_a, y=y_ax_a, color=c_ax_a if c_ax_a != "None" else None, 
                hover_data=["MeetName", "Event", "Squat", "Bench", "Deadlift", "Total", "Bodyweight", "Dots", "Wilks", "Meet#"]
            )
            st.plotly_chart(fig_ath, use_container_width=True)
        
        st.markdown("---")
        st.subheader("Selected Competitions History", anchor=False)
        all_meets = sorted(ath_df["NormMeet"].dropna().unique().tolist())
        selected_meets = st.multiselect("Select competitions to view historical details:", all_meets, default=[], key="ath_meet_sel")
        
        if selected_meets:
            hist_df = ath_df[ath_df["NormMeet"].isin(selected_meets)].copy()
            hist_df["Date"] = pd.to_datetime(hist_df["Date"], errors='coerce').dt.strftime("%Y-%m-%d")
            
            for c in ["Bodyweight", "Squat", "Bench", "Deadlift", "Total"]:
                if c in hist_df.columns: hist_df[c] = (hist_df[c] * mult).round(1)
            for c in ["Dots", "Wilks", "GL Points"]:
                if c in hist_df.columns: hist_df[c] = hist_df[c].round(2)
            if "Experience" in hist_df.columns: hist_df["Experience"] = hist_df["Experience"].round(1)

            attempt_cols = [c for c in ["Squat1Kg", "Squat2Kg", "Squat3Kg", "Bench1Kg", "Bench2Kg", "Bench3Kg", "Deadlift1Kg", "Deadlift2Kg", "Deadlift3Kg"] if c in hist_df.columns]
            for ac in attempt_cols: hist_df[ac] = hist_df[ac].apply(format_attempt)
            
            all_possible_cols = ["MeetName", "NormMeet", "Event", "Place", "Date", "WeightClass", "Age", "Experience", "Meet#", "Bodyweight", "Total", "Dots", "Wilks", "GL Points"] + attempt_cols + ["Federation", "ParentFederation", "Equipment", "Tested"]
            avail_cols = [c for c in all_possible_cols if c in hist_df.columns]
            
            default_cols_hist = ["MeetName", "Event", "Place", "Date", "Bodyweight", "Total", "Dots", "Wilks"]
            selected_cols = st.multiselect("Select columns to display:", avail_cols, default=[c for c in default_cols_hist if c in avail_cols], key="ath_champ_cols")
            
            render_dataframe(hist_df[selected_cols].sort_values("Date", ascending=False), key_prefix="ath_hist")

elif analysis_mode == "Athlete vs athlete":
    df_a_orig = st.session_state.ath_df_a
    df_b_orig = st.session_state.ath_df_b
    
    if not df_a_orig.empty and not df_b_orig.empty:
        c1_vs, c2_vs = st.columns(2)
        with c1_vs:
            st.markdown(f"**Filter {df_a_orig['Name'].iloc[0]}**")
            df_a_raw = apply_athlete_filters(df_a_orig, "vs_a")
        with c2_vs:
            st.markdown(f"**Filter {df_b_orig['Name'].iloc[0]}**")
            df_b_raw = apply_athlete_filters(df_b_orig, "vs_b")
            
        vs_event = st.session_state.get("vs_event_opt", "Any")
        if vs_event != "Any":
            df_a = df_a_raw[df_a_raw['Event'] == vs_event].copy()
            df_b = df_b_raw[df_b_raw['Event'] == vs_event].copy()
        else:
            df_a, df_b = df_a_raw.copy(), df_b_raw.copy()
            
        name_a = df_a['Name'].iloc[0] if not df_a.empty else df_a_orig['Name'].iloc[0]
        name_b = df_b['Name'].iloc[0] if not df_b.empty else df_b_orig['Name'].iloc[0]
        
        st.subheader(f"Head-to-Head: {name_a} vs {name_b}", anchor=False)
        if vs_event != "Any": st.caption(f"Filtered to Event: {vs_event}")
        
        def get_stat(df_stat, col, agg="max"): 
            if df_stat.empty: return np.nan
            if agg == "median": return df_stat[col].median()
            return df_stat[col].max() if agg=="max" else df_stat[col].min()
        
        col1, col2, col3 = st.columns(3)
        col1.markdown(f"### {name_a}")
        col2.markdown("### Metric")
        col3.markdown(f"### {name_b}")
        
        stats = [
            ("Est. Birth Year", "EstBirthYear", 1, "median"),
            ("Max DOTS", "Dots", 1),
            ("Max Wilks", "Wilks", 1),
            ("Max GL Points", "GL Points", 1),
            ("Max Total", "Total", mult),
            ("Max Squat", "Squat", mult),
            ("Max Bench", "Bench", mult),
            ("Max Deadlift", "Deadlift", mult),
            ("Competitions Count", "Meet#", 1),
            ("Experience (Yrs)", "Experience", 1),
            ("Min Age", "Age", 1, "min"),
            ("Max Age", "Age", 1, "max")
        ]
        
        selected_stats = st.multiselect("Display metrics:", [s[1] for s in stats], default=[s[1] for s in stats])
        stats = [s for s in stats if s[1] in selected_stats]
        
        for label, col, multiplier, *agg in stats:
            agg_func = agg[0] if agg else "max"
            val_a = get_stat(df_a, col, agg_func) * multiplier
            val_b = get_stat(df_b, col, agg_func) * multiplier
            
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**{val_a:.1f}**" if pd.notna(val_a) else "N/A")
            c2.markdown(f"*{label}*")
            c3.markdown(f"**{val_b:.1f}**" if pd.notna(val_b) else "N/A")

        st.markdown("---")
        st.subheader("Common Meets (Cross-Year Comparison)", anchor=False)
        
        df_a_dedup = df_a.sort_values('Total', ascending=False).drop_duplicates(subset=['NormMeet'])
        df_b_dedup = df_b.sort_values('Total', ascending=False).drop_duplicates(subset=['NormMeet'])
        
        common = pd.merge(df_a_dedup, df_b_dedup, on="NormMeet", suffixes=(f'_{name_a}', f'_{name_b}'))
        
        if not common.empty:
            common_display = common[["NormMeet", "Date_" + name_a, "Date_" + name_b, "Event_" + name_a, "Event_" + name_b, "Place_" + name_a, "Place_" + name_b, "Total_" + name_a, "Total_" + name_b, "Dots_" + name_a, "Dots_" + name_b]].copy()
            common_display.rename(columns={
                f"Date_{name_a}": f"Date ({name_a})", f"Date_{name_b}": f"Date ({name_b})",
                f"Event_{name_a}": f"Event ({name_a})", f"Event_{name_b}": f"Event ({name_b})",
                f"Place_{name_a}": f"Place ({name_a})", f"Place_{name_b}": f"Place ({name_b})",
                f"Total_{name_a}": f"Total ({name_a})", f"Total_{name_b}": f"Total ({name_b})",
                f"Dots_{name_a}": f"Dots ({name_a})", f"Dots_{name_b}": f"Dots ({name_b})",
            }, inplace=True)
            for col in [f"Total ({name_a})", f"Total ({name_b})"]: common_display[col] = (common_display[col] * mult).round(1)
            render_dataframe(common_display, key_prefix="ath_vs_common")
        else:
            st.info("These athletes have no matching competition names (under selected constraints).")

        st.markdown("---")
        st.subheader("Dynamic Correlation (Comparative)", anchor=False)
        
        combined_ath = pd.concat([df_a, df_b])
        metric_opts = ["Total", "Dots", "Wilks", "GL Points", "Bodyweight", "Squat", "Bench", "Deadlift", "Date", "Meet#", "Experience", "Age"]
        
        cc1, cc2 = st.columns(2)
        with cc1: cx_ax = st.selectbox("Horizontal Axis", metric_opts, index=metric_opts.index("Date"), key="ath_vs_x")
        with cc2: cy_ax = st.selectbox("Vertical Axis", metric_opts, index=metric_opts.index("Dots"), key="ath_vs_y")
        
        plot_df = combined_ath.dropna(subset=[cx_ax, cy_ax]).copy()
        if not plot_df.empty:
            fig_vs = px.scatter(
                plot_df, x=cx_ax, y=cy_ax, color="Name", 
                hover_data=["MeetName", "Event", "Squat", "Bench", "Deadlift", "Total", "Bodyweight", "Dots", "Wilks"]
            )
            fig_vs.update_traces(mode='lines+markers')
            st.plotly_chart(fig_vs, use_container_width=True)
            
elif analysis_mode == "Competition Analysis" and st.session_state.get("df_comp") is not None and not st.session_state.get("df_comp", pd.DataFrame()).empty:
    df_comp = st.session_state.df_comp
    st.subheader(f"Competition Analysis: {st.session_state.comp_name}", anchor=False)

    st.markdown("### Filters")
    col_f1, col_f2, col_f3 = st.columns(3)
    
    def map_age(age):
        if pd.isna(age) or age == 0: return "Unknown"
        if age <= 18: return "Subjunior"
        if age <= 23: return "Junior"
        if age <= 39: return "Open"
        return "Masters"
    
    if "AgeCategory" not in df_comp.columns:
        df_comp["AgeCategory"] = df_comp["Age"].apply(map_age)
        
    with col_f1:
        f_events = st.multiselect("Event", df_comp["Event"].dropna().unique().tolist(), default=df_comp["Event"].dropna().unique().tolist())
        f_equip = st.multiselect("Equipment", df_comp["Equipment"].dropna().unique().tolist(), default=df_comp["Equipment"].dropna().unique().tolist())
    with col_f2:
        f_fed = st.multiselect("Federation", df_comp["Federation"].dropna().unique().tolist(), default=df_comp["Federation"].dropna().unique().tolist())
        f_pfed = st.multiselect("Parent Federation", df_comp["ParentFederation"].dropna().unique().tolist(), default=df_comp["ParentFederation"].dropna().unique().tolist())
    with col_f3:
        tested_opts = df_comp["Tested"].fillna("No").unique().tolist()
        f_tested = st.multiselect("Tested", tested_opts, default=tested_opts)
        f_age = st.multiselect("Age Category", df_comp["AgeCategory"].dropna().unique().tolist(), default=df_comp["AgeCategory"].dropna().unique().tolist())
        
    df_filtered = df_comp[
        df_comp["Event"].isin(f_events) &
        df_comp["Equipment"].isin(f_equip) &
        df_comp["Federation"].isin(f_fed) &
        df_comp["ParentFederation"].isin(f_pfed) &
        df_comp["AgeCategory"].isin(f_age) &
        df_comp["Tested"].fillna("No").isin(f_tested)
    ]
    
    render_competition_section(df_filtered, {"group_name": st.session_state.comp_name})

elif analysis_mode == "Calculators":
    st.header("Formulas Point Calculator", anchor=False)
    
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        calc_sex = st.selectbox("Biological Sex", ["M", "F"])
        calc_bw = st.number_input(f"Bodyweight ({unit})", min_value=1.0, value=80.0, step=0.5)
    with cc2:
        c2a, c2b = st.columns(2)
        calc_sq_w = c2a.number_input(f"Squat ({unit})", min_value=0.0, value=150.0, step=2.5)
        calc_sq_r = c2b.number_input("SQ Reps", value=1, min_value=1)
        if calc_sq_r > 1: st.caption(f"Est 1RM: {calc_1rm(calc_sq_w, calc_sq_r):.1f} {unit}")
        
        c3a, c3b = st.columns(2)
        calc_bn_w = c3a.number_input(f"Bench ({unit})", min_value=0.0, value=100.0, step=2.5)
        calc_bn_r = c3b.number_input("BN Reps", value=1, min_value=1)
        if calc_bn_r > 1: st.caption(f"Est 1RM: {calc_1rm(calc_bn_w, calc_bn_r):.1f} {unit}")
        
        c4a, c4b = st.columns(2)
        calc_dl_w = c4a.number_input(f"Deadlift ({unit})", min_value=0.0, value=200.0, step=2.5)
        calc_dl_r = c4b.number_input("DL Reps", value=1, min_value=1)
        if calc_dl_r > 1: st.caption(f"Est 1RM: {calc_1rm(calc_dl_w, calc_dl_r):.1f} {unit}")
        
    calc_sq_1rm = calc_1rm(calc_sq_w, calc_sq_r)
    calc_bn_1rm = calc_1rm(calc_bn_w, calc_bn_r)
    calc_dl_1rm = calc_1rm(calc_dl_w, calc_dl_r)
    calc_total = calc_sq_1rm + calc_bn_1rm + calc_dl_1rm
    
    # Conversion to KG for scoring systems
    calc_bw_kg = calc_bw / 2.20462 if use_lbs else calc_bw
    calc_tot_kg = calc_total / 2.20462 if use_lbs else calc_total
    
    # DOTS
    dots_score = calc_tot_kg * (500.0 / (-0.000001093 * (calc_bw_kg ** 4) + 0.0007391293 * (calc_bw_kg ** 3) -0.1918759221 * (calc_bw_kg ** 2) + 24.0900756 * calc_bw_kg - 307.75076)) if calc_sex=="M" else calc_tot_kg * (500.0 / (-0.000010706 * (calc_bw_kg ** 4) + 0.005158568 * (calc_bw_kg ** 3) -0.92501065 * (calc_bw_kg ** 2) + 75.323049 * calc_bw_kg - 516.39869))
    
    # Wilks
    if calc_sex == "M":
        denom = -216.0475144 + 16.2606339*calc_bw_kg -0.002388645*(calc_bw_kg**2) -0.00113732*(calc_bw_kg**3) + 7.01863E-06*(calc_bw_kg**4) -1.291E-08*(calc_bw_kg**5)
    else:
        denom = 594.3174777 -27.23842536*calc_bw_kg + 0.821122268*(calc_bw_kg**2) -0.009307339*(calc_bw_kg**3) + 4.73158E-05*(calc_bw_kg**4) -9.054E-08*(calc_bw_kg**5)
    wilks_score = calc_tot_kg * (500.0 / denom) if denom != 0 else 0
    
    # GL Points
    denom_gl = (1199.72839 - 925.40462 * np.exp(-0.00510531 * calc_bw_kg)) if calc_sex=="M" else (610.79046 - 451.04414 * np.exp(-0.00735665 * calc_bw_kg))
    gl_score = calc_tot_kg * (100.0 / denom_gl) if denom_gl != 0 else 0
    
    with cc3:
        st.metric("Total", f"{calc_total:.1f} {unit}")
        st.metric("DOTS", f"{dots_score:.2f}")
        st.metric("Wilks", f"{wilks_score:.2f}")
        st.metric("GL Points", f"{gl_score:.2f}")
