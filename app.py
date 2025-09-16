# -*- coding: utf-8 -*-
# Company & Role Recommender — tailored for your dataset
# pip install -r requirements.txt

import re
import sqlite3
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ----------------- App config -----------------
st.set_page_config(page_title="Company & Role Recommender", layout="wide")
st.title("🏢 Company & Role Recommender (DB-backed)")
st.caption("Upload your dataset, save to a local DB, then filter by Major & Semester to see company/role summaries.")

DB_PATH = "company_data.db"
TABLE = "records"

GRADE_ORDER = ["A", "B", "C", "D", "E", "F"]
GRADE_TO_SCORE = {"A": 5, "B": 4, "C": 3, "D": 2, "E": 1, "F": 0}

# Adjacency map for “include adjacent semesters”
ADJ = {
    "Semester 1": {"Summer"},
    "Summer": {"Semester 1", "Semester 2"},
    "Semester 2": {"Summer"},
    "Year long": {"Semester 1", "Summer", "Semester 2"},
}

# ----------------- DB helpers -----------------
@st.cache_resource(show_spinner=False)
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_resource(show_spinner=False)
def ensure_table():
    conn = get_conn()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company  TEXT,
            major    TEXT,
            semester TEXT,
            role     TEXT,
            grade    TEXT,
            note     TEXT
        );
    """)
    conn.commit()
    return True

# ----------------- Semester classification -----------------
def _overlap_days(a_start, a_end, b_start, b_end):
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0, (end - start).days + 1)

def _term_windows_for_year(y: int):
    sem1 = (date(y, 8, 15), date(y, 12, 31))   # Aug 15 – Dec 31
    summer = (date(y, 6, 1), date(y, 8, 14))   # Jun 1 – Aug 14
    sem2 = (date(y, 1, 1), date(y, 5, 31))     # Jan 1 – May 31
    return {"Semester 1": sem1, "Summer": summer, "Semester 2": sem2}

def classify_by_period(start_val, finish_val) -> Optional[str]:
    if pd.isna(start_val) or pd.isna(finish_val):
        return None
    try:
        s = pd.to_datetime(start_val).date()
        f = pd.to_datetime(finish_val).date()
    except Exception:
        return None
    if f < s:
        s, f = f, s
    duration = (f - s).days + 1
    if duration >= 240:
        return "Year long"
    years = range(s.year - 1, f.year + 2)
    overlaps = {"Semester 1": 0, "Summer": 0, "Semester 2": 0}
    for y in years:
        windows = _term_windows_for_year(y)
        for k, (ws, we) in windows.items():
            overlaps[k] += _overlap_days(s, f, ws, we)
    order = ["Semester 1", "Summer", "Semester 2"]
    best = max(order, key=lambda k: (overlaps[k], -order.index(k)))
    return best

# ----------------- Grade normalization -----------------
def normalize_grade(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    return s.str.extract(r"([A-F])", expand=False).where(lambda x: x.isin(GRADE_ORDER))

# ----------------- Anonymization (optional) -----------------
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d{1,3}[-\s]?)?(?:0)?1\d[-.\s]?\d{3,4}[-.\s]?\d{4}")
STUID_RE = re.compile(r"\b(?:20\d{2}|19\d{2})-?\d{4,6}\b")

def anonymize_series(texts: pd.Series) -> pd.Series:
    def _mask(t: str) -> str:
        if not isinstance(t, str): return t
        x = EMAIL_RE.sub("[EMAIL]", t)
        x = PHONE_RE.sub("[PHONE]", x)
        x = STUID_RE.sub("[STUDENT_ID]", x)
        return x
    return texts.apply(_mask)

# ----------------- DB insert/query -----------------
def insert_df(df: pd.DataFrame, mode: str = "append"):
    conn = get_conn()
    if mode == "replace":
        conn.execute(f"DELETE FROM {TABLE}")
        conn.commit()
    df.to_sql(TABLE, conn, if_exists="append", index=False)

def db_count() -> int:
    conn = get_conn()
    return conn.execute(f"SELECT COUNT(*) FROM {TABLE}").fetchone()[0]

def query_uniques(col: str) -> List[str]:
    conn = get_conn()
    q = f"SELECT DISTINCT {col} FROM {TABLE} WHERE {col} IS NOT NULL AND {col}<>'' ORDER BY 1"
    return [r[0] for r in conn.execute(q).fetchall()]

def load_filtered(major: str, sem: str, include_adjacent: bool) -> pd.DataFrame:
    conn = get_conn()
    sems = [sem] + (list(ADJ.get(sem, set())) if include_adjacent else [])
    placeholders = ",".join(["?"] * len(sems))
    sql = f"""
        SELECT company, major, semester, role, grade, note
        FROM {TABLE}
        WHERE major = ? AND semester IN ({placeholders})
    """
    params = [major] + sems
    return pd.read_sql_query(sql, conn, params=params)

# ----------------- Sidebar: Upload & Save -----------------
ensure_table()
with st.sidebar:
    st.header("① Upload & Save to DB")
    up = st.file_uploader("Upload Excel/CSV", type=["csv","xlsx","xls"])
    tmp_df = None
    if up is not None:
        try:
            if up.name.lower().endswith((".xlsx",".xls")):
                xls = pd.ExcelFile(up)
                sheet = st.selectbox("Pick a sheet", xls.sheet_names)
                tmp_df = xls.parse(sheet)
            else:
                tmp_df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Failed to read file: {e}")

    if tmp_df is not None:
        cols = tmp_df.columns.tolist()
        st.caption("Column mapping (defaults set for your dataset):")
        m_company = st.selectbox("Company", cols, index=cols.index("Company name"))
        m_major   = st.selectbox("Major", cols, index=cols.index("Major"))
        m_start   = st.selectbox("Period start", cols, index=cols.index("Period start"))
        m_finish  = st.selectbox("Period finish", cols, index=cols.index("Period finish"))
        m_role    = st.selectbox("Role/Position", cols, index=cols.index("Job Position"))
        m_grade   = st.selectbox("Grade", cols, index=cols.index("Overall performance"))
        m_note    = st.selectbox("Comment (optional)", ["(none)"] + cols)

        mode = st.radio("Save mode", ["Overwrite","Append"], horizontal=True)

        if st.button("Save to DB", use_container_width=True):
            try:
                df_in = pd.DataFrame({
                    "company":  tmp_df[m_company].astype(str).str.strip(),
                    "major":    tmp_df[m_major].astype(str).str.strip(),
                    "semester": [classify_by_period(a, b) for a, b in zip(tmp_df[m_start], tmp_df[m_finish])],
                    "role":     tmp_df[m_role].astype(str).str.strip(),
                    "grade":    normalize_grade(tmp_df[m_grade]),
                    "note":     tmp_df[m_note] if m_note != "(none)" else None,
                })
                df_in = df_in.dropna(subset=["company","major","semester","role","grade"]).copy()
                insert_df(df_in, mode=("replace" if mode=="Overwrite" else "append"))
                st.success(f"Saved! Total rows in DB: {db_count()}")
            except Exception as e:
                st.error(f"Save failed: {e}")

    st.divider()
    st.header("② DB Tools")
    st.metric("Rows in DB", db_count())

# ----------------- Filters -----------------
with st.sidebar:
    st.header("③ Filters")
    majors = query_uniques("major")
    if len(majors) == 0:
        st.info("DB is empty. Please upload & save data.")
    sel_major = st.selectbox("Major", majors if majors else ["(no data)"])
    sel_sem   = st.radio("Internship semester", ["Semester 1","Summer","Semester 2","Year long"], horizontal=True)
    include_adj = st.checkbox("Include adjacent semesters", value=True)
    min_samples_company = st.number_input("Min samples (company)", 0, 100, 3)
    min_samples_role    = st.number_input("Min samples (role)", 0, 100, 2)

# ----------------- Query & results -----------------
if db_count() == 0:
    st.stop()

qdf = load_filtered(sel_major, sel_sem, include_adjacent=include_adj)
if qdf.empty:
    st.info("No data for this major/semester.")
    st.stop()

qdf["grade"] = normalize_grade(qdf["grade"])
qdf = qdf.dropna(subset=["grade"]).copy()
qdf["grade_score"] = qdf["grade"].map(GRADE_TO_SCORE)

st.header("Results")
tab_company, tab_role = st.tabs(["Company summary","Role summary"])

# ---- Company summary ----
with tab_company:
    grp = qdf.groupby("company")
    probs = grp["grade"].value_counts(normalize=True).unstack(fill_value=0)
    for g in GRADE_ORDER:
        if g not in probs.columns: probs[g] = 0.0
    probs = probs[GRADE_ORDER]

    def top_roles(series, topn=3):
        vc = series.dropna().astype(str).value_counts()
        return "-" if vc.empty else " · ".join([f"{r}({c})" for r,c in vc.head(topn).items()])

    summary = pd.DataFrame({
        "Samples": grp.size(),
        "AvgScore": grp["grade_score"].mean(),
        "TopRoles": grp["role"].apply(top_roles),
    })
    out = summary.join(probs)
    out = out[out["Samples"] >= min_samples_company]
    out["ExpectedGrade"] = pd.cut(
        out["AvgScore"], bins=[-1,0.5,1.5,2.5,3.5,4.5,6],
        labels=["F","E","D","C","B","A"]
    )
    st.dataframe(out.round(3), use_container_width=True)

# ---- Role summary ----
with tab_role:
    grp2 = qdf.groupby(["company","role"])
    probs2 = grp2["grade"].value_counts(normalize=True).unstack(fill_value=0)
    for g in GRADE_ORDER:
        if g not in probs2.columns: probs2[g] = 0.0
    probs2 = probs2[GRADE_ORDER]

    summary2 = pd.DataFrame({
        "Samples": grp2.size(),
        "AvgScore": grp2["grade_score"].mean(),
    })
    out2 = summary2.join(probs2)
    out2 = out2[out2["Samples"] >= min_samples_role]
    out2["ExpectedGrade"] = pd.cut(
        out2["AvgScore"], bins=[-1,0.5,1.5,2.5,3.5,4.5,6],
        labels=["F","E","D","C","B","A"]
    )
    st.dataframe(out2.round(3), use_container_width=True)

with st.expander("ℹ️ Notes"):
    st.markdown("""
    - Semesters are derived from Period start/finish:
        - Semester 1: Aug 15 – Dec 31
        - Semester 2: Jan 1 – May 31
        - Summer: Jun 1 – Aug 14
        - Year long: duration ≥ 240 days
    - Grades like A+, A-, B+ are reduced to A–F.
    """)

