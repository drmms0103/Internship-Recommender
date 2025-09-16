# -*- coding: utf-8 -*-
# Internship Recommender — Streamlit (replicates your HTML/JS flow)
# Run:
#   pip install -r requirements.txt
#   streamlit run app.py

import re
import sqlite3
from datetime import date
from typing import List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="Internship Recommender", layout="wide")
st.title("🎓 Internship Recommendation Tool (Streamlit)")
st.caption("Upload Excel/CSV → map columns → pick Major & Semester → see company/role summaries & charts.")

DB_PATH = "company_data.db"
TABLE = "records"

GRADE_ORDER = ["A","B","C","D","E","F"]
GRADE_TO_SCORE = {"A":5,"B":4,"C":3,"D":2,"E":1,"F":0}

ADJ = {
    "Semester 1": {"Summer"},
    "Summer": {"Semester 1", "Semester 2"},
    "Semester 2": {"Summer"},
    "Year long": {"Semester 1","Summer","Semester 2"},
}

# ----------------- DB helpers -----------------
@st.cache_resource
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

@st.cache_resource
def ensure_table():
    conn = get_conn()
    conn.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE}(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            company  TEXT,
            major    TEXT,
            semester TEXT,
            role     TEXT,
            grade    TEXT,
            note     TEXT
        )
    """)
    conn.commit()
    return True

def insert_df(df: pd.DataFrame, mode="append"):
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

# ----------------- Semester classification by period -----------------
def _overlap_days(a_start, a_end, b_start, b_end):
    start = max(a_start, b_start)
    end = min(a_end, b_end)
    return max(0, (end - start).days + 1)

def _term_windows_for_year(y: int):
    sem1  = (date(y,8,15), date(y,12,31))  # Aug 15 – Dec 31
    summer= (date(y,6,1),  date(y,8,14))   # Jun 1 – Aug 14
    sem2  = (date(y,1,1),  date(y,5,31))   # Jan 1 – May 31
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
    overlaps = {"Semester 1":0,"Summer":0,"Semester 2":0}
    for y in range(s.year-1, f.year+2):
        windows = _term_windows_for_year(y)
        for k,(ws,we) in windows.items():
            overlaps[k] += _overlap_days(s,f,ws,we)
    # tie-breaking order: Sem1 > Summer > Sem2
    order = ["Semester 1","Summer","Semester 2"]
    return max(order, key=lambda k: overlaps[k])

# ----------------- Grade normalization -----------------
def normalize_grade(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.upper()
    return s.str.extract(r"([A-F])", expand=False).where(lambda x: x.isin(GRADE_ORDER))

# ----------------- Sidebar: Upload & Map & Save -----------------
ensure_table()
with st.sidebar:
    st.header("① Upload & Map")
    up = st.file_uploader("Excel/CSV", type=["xlsx","xls","csv"])
    tmp_df = None
    if up is not None:
        try:
            if up.name.lower().endswith((".xlsx",".xls")):
                xls = pd.ExcelFile(up)
                sheet = st.selectbox("Sheet", xls.sheet_names)
                tmp_df = xls.parse(sheet)
            else:
                tmp_df = pd.read_csv(up)
        except Exception as e:
            st.error(f"Failed to read file: {e}")
            tmp_df = None

    if tmp_df is not None:
        tmp_df.columns = [str(c).strip() for c in tmp_df.columns]
        cols = tmp_df.columns.tolist()

        st.caption("Map your columns (auto-detected; change if needed).")

        def find_index(cols, *cands):
            low = [c.lower() for c in cols]
            cand_low = [str(x).strip().lower() for x in cands]
            for x in cand_low:
                if x in low: return low.index(x)
            # partial fallback
            for i,c in enumerate(low):
                if any(x in c for x in cand_low):
                    return i
            return 0

        idx_company = find_index(cols, "company name", "company")
        idx_major   = find_index(cols, "major", "programme", "program", "department")
        idx_role    = find_index(cols, "job position", "role", "position")
        idx_grade   = find_index(cols, "overall performance", "overall grade", "grade")
        idx_start   = find_index(cols, "period start", "start date", "start")
        idx_finish  = find_index(cols, "period finish", "end date", "finish", "end")

        m_company = st.selectbox("Company", cols, index=idx_company)
        m_major   = st.selectbox("Major", cols, index=idx_major)
        m_role    = st.selectbox("Role/Position", cols, index=idx_role)
        m_grade   = st.selectbox("Overall performance (grade)", cols, index=idx_grade)
        m_start   = st.selectbox("Period start", cols, index=idx_start)
        m_finish  = st.selectbox("Period finish", cols, index=idx_finish)
        m_note    = st.selectbox("Note (optional)", ["(none)"] + cols)

        st.divider()
        st.header("② Save to DB")
        mode = st.radio("Save mode", ["Overwrite","Append"], horizontal=True)
        if st.button("Save", use_container_width=True):
            try:
                df_in = pd.DataFrame({
                    "company":  tmp_df[m_company].astype(str).str.strip(),
                    "major":    tmp_df[m_major].astype(str).str.strip(),
                    "semester": [classify_by_period(a,b) for a,b in zip(tmp_df[m_start], tmp_df[m_finish])],
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
    st.header("③ DB Status")
    st.metric("Rows in DB", db_count())

# ----------------- Filters -----------------
with st.sidebar:
    st.header("④ Filters")
    majors = query_uniques("major")
    if not majors:
        st.info("DB is empty. Upload & save first.")
    sel_major = st.selectbox("Major", majors if majors else ["(no data)"])
    sel_sem   = st.selectbox("Semester", ["Semester 1","Summer","Semester 2","Year long"])
    include_adj = st.checkbox("Include adjacent semesters", value=True)
    min_samples_company = st.number_input("Min samples (company)", 0, 100, 3)
    min_samples_role    = st.number_input("Min samples (role)", 0, 100, 2)

# ----------------- Query & Compute -----------------
if db_count() == 0:
    st.stop()

dfq = load_filtered(sel_major, sel_sem, include_adjacent=include_adj)
if dfq.empty:
    st.info("No data for this major/semester.")
    st.stop()

dfq["grade"] = normalize_grade(dfq["grade"])
dfq = dfq.dropna(subset=["grade"]).copy()
dfq["grade_score"] = dfq["grade"].map(GRADE_TO_SCORE)

st.subheader("Results")

tab1, tab2 = st.tabs(["Company summary", "Role summary (Company × Role)"])

# ---------- Company summary ----------
with tab1:
    grp = dfq.groupby("company")
    probs = grp["grade"].value_counts(normalize=True).unstack(fill_value=0)
    # ensure column order A..F
    for g in GRADE_ORDER:
        if g not in probs.columns:
            probs[g] = 0.0
    probs = probs[GRADE_ORDER]

    def top_roles(series, topn=3):
        vc = series.dropna().astype(str).str.strip().value_counts()
        return "-" if vc.empty else " · ".join([f"{r}({c})" for r,c in vc.head(topn).items()])

    summary = pd.DataFrame({
        "Samples": grp.size(),
        "AvgScore": grp["grade_score"].mean(),
        "TopRoles": grp["role"].apply(top_roles),
    })
    out = summary.join(probs)
    out = out[out["Samples"] >= min_samples_company]
    out["Expected"] = pd.cut(
        out["AvgScore"], bins=[-1,0.5,1.5,2.5,3.5,4.5,6],
        labels=["F","E","D","C","B","A"]
    ).astype(str)
    out = out.sort_values(["AvgScore","Samples"], ascending=[False, False])

    st.dataframe(out.round(3), use_container_width=True)

    # Chart: Top 10 by AvgScore
    if not out.empty:
        top10 = out.reset_index().rename(columns={"index":"Company"}).head(10)
        top10 = top10.reset_index(names="Rank")
        chart = (
            alt.Chart(top10)
            .mark_bar()
            .encode(
                x=alt.X("AvgScore:Q", scale=alt.Scale(domain=[0,5])),
                y=alt.Y("company:N", sort="-x", title="Company"),
                tooltip=["company","AvgScore","Samples","Expected","TopRoles"]
            )
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)

    csv1 = out.reset_index().to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 Download company_summary.csv", data=csv1, file_name="company_summary.csv", mime="text/csv")

# ---------- Role summary ----------
with tab2:
    grp2 = dfq.groupby(["company","role"])
    probs2 = grp2["grade"].value_counts(normalize=True).unstack(fill_value=0)
    for g in GRADE_ORDER:
        if g not in probs2.columns:
            probs2[g] = 0.0
    probs2 = probs2[GRADE_ORDER]

    summary2 = pd.DataFrame({
        "Samples": grp2.size(),
        "AvgScore": grp2["grade_score"].mean(),
    })
    out2 = summary2.join(probs2)
    out2 = out2[out2["Samples"] >= min_samples_role]
    out2["Expected"] = pd.cut(
        out2["AvgScore"], bins=[-1,0.5,1.5,2.5,3.5,4.5,6],
        labels=["F","E","D","C","B","A"]
    ).astype(str)
    out2 = out2.sort_values(["AvgScore","Samples"], ascending=[False, False])

    st.dataframe(out2.round(3), use_container_width=True)

    csv2 = out2.reset_index().to_csv(index=False, encoding="utf-8-sig")
    st.download_button("📥 Download role_summary.csv", data=csv2, file_name="role_summary.csv", mime="text/csv")

with st.expander("ℹ️ Notes"):
    st.markdown("""
- Semesters derive from **Period start/finish** windows:
  - Semester 1: Aug 15 – Dec 31
  - Summer: Jun 1 – Aug 14
  - Semester 2: Jan 1 – May 31
  - Year long: duration ≥ 240 days
- Grades like A+, A-, B+ are normalized to **A–F (first letter)** and scored A=5 … F=0.
- Use **Min samples** filters to reduce noise.
    """)



