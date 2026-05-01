import os
import re
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from openai import OpenAI
from PIL import Image

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="VAR Decision Intelligence Simulator",
    page_icon="🧭",
    layout="wide",
)

logo = Image.open("utils/assets/fuerte.png")

# =========================================================
# STYLE
# =========================================================
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ECEDE8;
    }

    .main-title {
        text-align: center;
        color: #163D1F;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        text-align: center;
        color: #4B5563;
        font-size: 1.05rem;
        margin-bottom: 1.25rem;
    }

    .card {
        background: white;
        border: 1px solid #D1D5DB;
        border-radius: 16px;
        padding: 1rem 1rem 0.8rem 1rem;
        box-shadow: 0 4px 10px rgba(0,0,0,0.06);
    }

    .section-label {
        color: #166534;
        font-weight: 700;
        font-size: 1.1rem;
        margin-bottom: 0.4rem;
    }

    section[data-testid="stSidebar"] {
        background-color: #163D1F;
        border-right: 1px solid #0F2A15;
    }

    /* Sidebar headers */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #F9FAFB !important;
        font-weight: 600;
    }

    /* Sidebar text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span {
        color: #E5E7EB;
    }

    /* Inputs (keep readable) */
    section[data-testid="stSidebar"] input,
    section[data-testid="stSidebar"] select,
    section[data-testid="stSidebar"] textarea {
        color: #111827 !important;
        background-color: #F9FAFB !important;
        border-radius: 8px;
    }

    /* Improve selectbox text visibility */
    section[data-testid="stSidebar"] .stSelectbox div {
        color: #111827 !important;
    }

    /* Sidebar Mode radio card */
    section[data-testid="stSidebar"] .stRadio {
        background-color: #FFFFFF !important;
        padding: 14px 16px;
        border-radius: 14px;
        margin-top: 8px;
        margin-bottom: 18px;
        border: 1px solid #D1D5DB;
    }

    /* Mode title inside white card */
    section[data-testid="stSidebar"] .stRadio > label,
    section[data-testid="stSidebar"] .stRadio > label p {
        color: #111827 !important;
        font-weight: 700;
    }

    /* Radio option labels inside white card */
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label,
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label *,
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] span,
    section[data-testid="stSidebar"] .stRadio [role="radiogroup"] p {
        color: #111827 !important;
        background-color: transparent !important;
        font-weight: 500;
    }

    /* Keep the selected radio dot green */
    section[data-testid="stSidebar"] .stRadio input[type="radio"] {
        accent-color: #22C55E;
    }
    
    .small-muted {
        color: #6B7280;
        font-size: 0.88rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid #D1D5DB;
        padding: 0.7rem 0.8rem;
        border-radius: 14px;
        font-weight: 600;
    }

    .stProgress > div > div {
        background-color: #22C55E;
    }

    /* Chart container spacing */
    .block-container {
        padding-top: 1.5rem;
    }

    /* Dataframe cleanup */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        border: 1px solid #E5E7EB;
    }

    .custom-metric-card {
        background: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        min-height: 118px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.04);
    }

    .metric-label {
        color: #374151;
        font-size: 1rem;
        font-weight: 600;
        margin-bottom: 0.55rem;
    }

    .metric-value {
        color: #111827;
        font-size: 1.9rem;
        font-weight: 800;
        line-height: 1.1;
        white-space: normal;
        overflow: visible;
        text-overflow: unset;
        word-break: normal;
    }

    .metric-caption {
        color: #6B7280;
        font-size: 0.8rem;
        margin-top: 0.35rem;
    }

    img {
        max-height: 80px;
    }
</style>
    """,
    unsafe_allow_html=True
)


col1, col2 = st.columns([1, 4])

with col1:
    st.image(logo, width=120)

with col2:
    st.markdown(
        '<div class="main-title">VAR Decision Intelligence Simulator</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        '<div class="subtitle">Powered by Team Fuerte • Vigilance • Agility • Resilience</div>',
        unsafe_allow_html=True
    )

# =========================================================
# CONFIG
# =========================================================
DEFAULT_MODEL = "gpt-4o-mini"
CSV_PATH = "data/var_main_deaths_capstone_ready.csv"

PROFESSIONS = {
    "Nursing": {
        "description": "Healthcare professionals making fast, high-stakes decisions in dynamic patient care environments.",
        "keywords": ["health", "hospital", "medical", "clinical", "patient", "care"],
        "suggested_categories": ["Health events"],
        "scenarios": [
            "A patient suddenly deteriorates during a busy shift and staffing is thin. What do you do first?",
            "You suspect a medication order may be unsafe, but the unit is under pressure and time is limited. How do you respond?",
            "Several patients need attention at once after a rapid change in one patient's condition. How do you prioritize and communicate?",
            "A patient becomes unresponsive and you are the first to arrive before the code team. What actions do you take?",
            "A family member is demanding immediate intervention while you are managing another critical patient. How do you handle this?",
            "You discover a potential medication error just before administration. What steps do you take?",
            "Two patients begin deteriorating simultaneously and you have limited support. How do you prioritize?",
            "A patient is showing subtle signs of sepsis but labs are pending. What do you do?",
            "You receive conflicting handoff information about a high-risk patient. How do you verify and act?",
            "A patient refuses a critical treatment that could prevent rapid decline. How do you respond?",
            "You are assigned a float patient in an unfamiliar unit who begins to decline. What is your approach?",
            "A physician dismisses your concern about a patient who is worsening. What do you do next?",
            "You notice a pattern of near-miss errors during your shift. How do you address it in real time?",
            "A patient falls while attempting to ambulate independently. What are your immediate priorities?",
            "You are short-staffed and must delegate tasks quickly. How do you ensure safe coverage?",
            "A patient shows signs of internal bleeding post-procedure. What actions do you take?",
            "You suspect a colleague is making unsafe decisions due to fatigue. How do you handle it?",
            "A patient’s condition improves, but you are concerned about premature discharge. What do you do?",
            "You are managing multiple IV drips with competing priorities. How do you ensure safety?",
            "A patient develops sudden respiratory distress without a clear cause. What is your first move?",
            "You are behind on charting while patient acuity increases. How do you balance documentation and care?",
            "You identify a potential infection control breach on the unit. What steps do you take?",
        ],
    },

    "Policing": {
        "description": "Law enforcement professionals making decisions under uncertainty, public pressure, and time-sensitive risk.",
        "keywords": ["shooting", "violence", "attack", "security", "public safety", "law enforcement", "threat"],
        "suggested_categories": [
            "Fatal shootings (schools, workplaces, worship)",
            "Other mass-casualty / infrastructure / terror",
        ],
        "scenarios": [
            "You arrive at an active, chaotic public scene with conflicting witness reports. What is your first move?",
            "A routine contact escalates rapidly and bystanders are present. How do you stabilize the situation?",
            "You receive incomplete information about a possible threat at a crowded location. How do you assess and act?",
            "A suspect is behaving erratically and may be armed, but no weapon is visible. What is your approach?",
            "You respond to a mental health crisis involving a non-compliant individual. What do you do?",
            "A peaceful crowd begins to escalate after a triggering event. How do you manage the situation?",
            "Dispatch reports a possible active shooter, but witnesses provide conflicting descriptions. What do you do?",
            "Your partner is injured during an unstable situation. What are your immediate priorities?",
            "You arrive first at a potentially dangerous domestic disturbance. How do you proceed?",
            "A suspect flees into a populated area. How do you balance pursuit and public safety?",
            "You are outnumbered during a volatile encounter. What is your strategy?",
            "A weapon is reported, but visibility is poor and information is unclear. What do you do?",
            "You must make a split-second use-of-force decision with incomplete information. How do you approach it?",
            "You are managing multiple suspects with limited backup. How do you maintain control?",
            "A bystander interferes during a critical situation. How do you respond?",
            "You receive a report of a suspicious package in a public area. What steps do you take?",
            "You encounter a language barrier during a high-risk situation. How do you adapt?",
            "A high-speed pursuit is putting others at risk. Do you continue or disengage?",
            "You suspect misinformation is spreading during a public incident. What do you do?",
            "You are assigned to secure a scene while new threats may still exist. How do you operate?",
            "A juvenile suspect is involved in a serious incident. How do you handle the situation?",
            "You are under public scrutiny during a live-streamed incident. How does this affect your decision-making?",
        ],
    },

    "Grid Operator": {
        "description": "Operators responsible for maintaining system stability, triaging disruptions, and coordinating recovery across critical infrastructure.",
        "keywords": ["grid", "electrical", "outage", "power", "cyber", "infrastructure", "storm", "fuel", "critical systems"],
        "suggested_categories": [
            "Weather events (including storm-related outages)",
            "Non-weather infrastructure disruptions (electrical + cyber + fuel + critical systems)",
        ],
        "scenarios": [
            "A major transmission element fails during rising demand. How do you triage and stabilize the situation?",
            "Multiple alarms trigger across the system at once and the underlying cause is not yet clear. What do you do first?",
            "Severe weather is degrading conditions across multiple assets. How do you maintain service and coordinate recovery?",
            "A cascading failure risk begins to emerge across neighboring regions. What is your response?",
            "You receive alerts that could indicate either cyber intrusion or equipment malfunction. What do you do?",
            "Demand rapidly exceeds forecasts during extreme weather. How do you stabilize the grid?",
            "Communication with field teams is lost during a critical event. How do you proceed?",
            "A major outage affects multiple regions simultaneously. How do you prioritize restoration?",
            "A critical substation goes offline unexpectedly. What is your first move?",
            "Voltage instability is detected across several nodes. How do you respond?",
            "A backup system fails during peak load conditions. What do you do?",
            "Fuel supply disruptions threaten generation capacity. How do you manage the risk?",
            "A cyber alert coincides with system instability. How do you assess the situation?",
            "You must shed load to prevent collapse. How do you decide where?",
            "A key monitoring system goes down during a disturbance. What is your plan?",
            "Intermittent faults are occurring without a clear pattern. How do you investigate?",
            "A storm is approaching faster than forecasted. How do you prepare?",
            "Coordination between regions is breaking down during a crisis. What do you do?",
            "A restoration effort is underway but new failures are occurring. How do you adapt?",
            "You must balance speed of recovery with system stability. How do you approach it?",
            "A transformer shows signs of imminent failure. What actions do you take?",
            "You are managing simultaneous cyber and physical risks. How do you prioritize?",
        ],
    },
}

# =========================================================
# OPENAI HELPERS
# =========================================================
def get_api_key() -> str | None:
    try:
        if "OPENAI_API_KEY" in st.secrets:
            return st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass
    return os.getenv("OPENAI_API_KEY")


def get_client() -> OpenAI | None:
    api_key = get_api_key()
    if not api_key:
        return None
    return OpenAI(api_key=api_key)


def ask_model(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> str:
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )
    return response.output_text.strip()


# =========================================================
# DATA LOAD / CLEAN
# =========================================================
@st.cache_data
def load_var_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop unnamed columns if present
    drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
    if drop_cols:
        df = df.drop(columns=drop_cols, errors="ignore")

    # Numeric cleanup
    numeric_cols = [
        "vigilance",
        "agility",
        "resilience",
        "var_score",
        "fatalities",
        "duration_days",
        "estimated_cost_2024",
        "recovery_efficiency",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Date cleanup
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")

    # Text cleanup
    text_cols = [
        "event",
        "category",
        "insight",
        "notes",
        "source_title",
        "source_url",
        "location",
        "cost_impact",
        "cost_estimation_method",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Boolean cleanup
    if "cost_was_imputed" in df.columns:
        df["cost_was_imputed"] = df["cost_was_imputed"].fillna(False).astype(bool)

    # --- DYNAMIC VAR SCORING ---
    if "fatalities" in df.columns:
        fatality_bins = pd.cut(
            df["fatalities"].fillna(0),
            bins=[-1, 5, 25, 100, 500, float("inf")],
            labels=[1, 2, 3, 4, 5],
        ).astype(float)
    
        # Higher fatality events indicate greater need for vigilance
        df["vigilance"] = fatality_bins
    
        # Shorter duration = stronger agility proxy
        if "duration_days" in df.columns:
            df["agility"] = df["duration_days"].apply(
                lambda x: 5 if pd.notna(x) and x <= 2
                else 4 if pd.notna(x) and x <= 5
                else 3 if pd.notna(x) and x <= 14
                else 2
            )
        else:
            df["agility"] = 3
    
        # Lower cost impact = stronger resilience proxy
        if "cost_impact" in df.columns:
            df["resilience"] = df["cost_impact"].map({
                "Low": 5,
                "Medium": 4,
                "High": 3,
                "Unknown": 3,
            }).fillna(3)
        else:
            df["resilience"] = 3
    
        df["var_score"] = (
            (df["vigilance"] + df["agility"] + df["resilience"]) / 15 * 100
        )

    

    return df


# =========================================================
# DATA HELPERS
# =========================================================
def filter_by_profession(df: pd.DataFrame, profession: str) -> pd.DataFrame:
    if profession not in PROFESSIONS:
        return df.copy()

    info = PROFESSIONS[profession]
    categories = set(info.get("suggested_categories", []))
    keywords = [k.lower() for k in info.get("keywords", [])]

    work = df.copy()

    category_mask = pd.Series(False, index=work.index)
    if "category" in work.columns and categories:
        category_mask = work["category"].isin(categories)

    keyword_mask = pd.Series(False, index=work.index)
    text_cols = [c for c in ["event", "category", "insight", "notes"] if c in work.columns]
    if text_cols and keywords:
        combined = work[text_cols].fillna("").agg(" ".join, axis=1).str.lower()
        keyword_mask = combined.apply(lambda x: any(k in x for k in keywords))

    result = work[category_mask | keyword_mask].copy()

    # Fallback: if filter is too tight, use category-only if possible
    if result.empty and "category" in work.columns and categories:
        result = work[work["category"].isin(categories)].copy()

    # Final fallback
    if result.empty:
        result = work.copy()

    return result


def apply_user_filters(
    df: pd.DataFrame,
    selected_categories: List[str],
    fatality_min: float,
    keyword: str,
    selected_cost_impacts: List[str] | None = None,
) -> pd.DataFrame:
    work = df.copy()

    if selected_categories and "category" in work.columns:
        work = work[work["category"].isin(selected_categories)]

    if selected_cost_impacts and "cost_impact" in work.columns:
        work = work[work["cost_impact"].isin(selected_cost_impacts)]

    if "fatalities" in work.columns:
        work = work[work["fatalities"].fillna(0) >= fatality_min]

    if keyword.strip():
        q = keyword.strip().lower()
        text_cols = [
            c for c in ["event", "category", "insight", "notes", "location", "cost_impact"]
            if c in work.columns
        ]
        if text_cols:
            combined = work[text_cols].fillna("").agg(" ".join, axis=1).str.lower()
            work = work[combined.str.contains(q, na=False)]

    return work


def summarize_dataset(df: pd.DataFrame) -> Dict[str, float | int | str]:
    summary = {"records": len(df)}

    for col in ["var_score", "vigilance", "agility", "resilience", "fatalities", "duration_days", "estimated_cost_2024"]:
        if col in df.columns and not df[col].dropna().empty:
            summary[f"{col}_mean"] = float(df[col].mean())
        else:
            summary[f"{col}_mean"] = None

    if "category" in df.columns and not df.empty:
        summary["top_categories"] = df["category"].value_counts().head(5).to_dict()
    else:
        summary["top_categories"] = {}

    return summary


def summary_text(df: pd.DataFrame) -> str:
    s = summarize_dataset(df)

    lines = [
        f"Total records: {s['records']}",
    ]

    if s.get("var_score_mean") is not None:
        lines.append(f"Average VAR score: {s['var_score_mean']:.1f}")
    if s.get("vigilance_mean") is not None:
        lines.append(f"Average Vigilance: {s['vigilance_mean']:.2f}")
    if s.get("agility_mean") is not None:
        lines.append(f"Average Agility: {s['agility_mean']:.2f}")
    if s.get("resilience_mean") is not None:
        lines.append(f"Average Resilience: {s['resilience_mean']:.2f}")
    if s.get("fatalities_mean") is not None:
        lines.append(f"Average fatalities: {s['fatalities_mean']:.1f}")
    if s.get("estimated_cost_2024_mean") is not None:
        lines.append(f"Average estimated cost (2024$): ${s['estimated_cost_2024_mean']:,.0f}")

    top_categories = s.get("top_categories", {})
    if top_categories:
        lines.append("Top categories:")
        for k, v in top_categories.items():
            lines.append(f"- {k}: {v}")

    return "\n".join(lines)


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", str(text).lower())


def retrieve_relevant_events(df: pd.DataFrame, query: str, top_n: int = 5) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    query_tokens = set(tokenize(query))
    work = df.copy()

    def score_row(row) -> float:
        event = str(row.get("event", "")).lower()
        category = str(row.get("category", "")).lower()
        insight = str(row.get("insight", "")).lower()
        notes = str(row.get("notes", "")).lower()

        text = f"{event} {category} {insight} {notes}"
        row_tokens = set(tokenize(text))

        overlap = len(query_tokens & row_tokens)

        score = 0.0
        score += overlap * 3.0

        if query.strip():
            q = query.lower()
            if q in event:
                score += 6.0
            if q in category:
                score += 4.0
            if q in insight:
                score += 4.0

        # Favor more complete / impactful records slightly
        fatalities = row.get("fatalities", 0)
        var_score = row.get("var_score", 0)
        try:
            if pd.notna(fatalities):
                score += min(float(fatalities), 1000) / 1000.0
        except Exception:
            pass

        try:
            if pd.notna(var_score):
                score += float(var_score) / 100.0
        except Exception:
            pass

        return score

    work["_retrieval_score"] = work.apply(score_row, axis=1)
    work = work.sort_values("_retrieval_score", ascending=False)

    if query_tokens:
        top = work.head(top_n).copy()
    else:
        sort_cols = [c for c in ["fatalities", "var_score"] if c in work.columns]
        if sort_cols:
            top = work.sort_values(sort_cols, ascending=False).head(top_n).copy()
        else:
            top = work.head(top_n).copy()

    return top.drop(columns=["_retrieval_score"], errors="ignore")


def build_benchmark(df: pd.DataFrame) -> Dict[str, float | None]:
    out = {}
    for col in ["vigilance", "agility", "resilience", "var_score"]:
        if col in df.columns and not df[col].dropna().empty:
            out[f"{col}_mean"] = float(df[col].mean())
            out[f"{col}_p75"] = float(df[col].quantile(0.75))
        else:
            out[f"{col}_mean"] = None
            out[f"{col}_p75"] = None
    return out


def extract_scores(text: str) -> Tuple[int | None, int | None, int | None]:
    def get_one(label: str) -> int | None:
        match = re.search(rf"{label}[^\d]*(\d)", text, flags=re.IGNORECASE)
        if match:
            val = int(match.group(1))
            if 1 <= val <= 5:
                return val
        return None

    return get_one("Vigilance"), get_one("Agility"), get_one("Resilience")


def score_label(score: int | None) -> str:
    if score is None:
        return "NA"
    if score >= 4:
        return "Strong"
    if score == 3:
        return "Moderate"
    return "Needs improvement"


def format_events_for_prompt(df: pd.DataFrame, limit: int = 5) -> List[dict]:
    keep = [c for c in ["event", "category", "fatalities", "estimated_cost_2024", "cost_impact", "var_score", "vigilance", "agility", "resilience", "insight", "location"] if c in df.columns]
    if not keep:
        return []
    return df[keep].head(limit).to_dict(orient="records")


# =========================================================
# PROMPTS
# =========================================================
def scenario_prompt(
    profession: str,
    scenario: str,
    user_response: str,
    dataset_summary: str,
    benchmark: Dict[str, float | None],
    evidence_rows: List[dict],
) -> str:
    return f"""
You are evaluating a user's decision response using the VAR framework.

Profession:
{profession}

Definitions:
- Vigilance = awareness, anticipation, risk recognition, situational scanning
- Agility = prioritization, speed, adaptability, action under pressure
- Resilience = stabilization, recovery, follow-through, learning, continuity

Scenario:
{scenario}

User response:
{user_response}

Dataset summary:
{dataset_summary}

Benchmark context:
- Mean Vigilance: {benchmark.get('vigilance_mean')}
- Mean Agility: {benchmark.get('agility_mean')}
- Mean Resilience: {benchmark.get('resilience_mean')}
- Mean VAR Score: {benchmark.get('var_score_mean')}

Relevant real-world examples:
{evidence_rows}

Evaluate the user's response carefully.

Return in exactly this format:

Vigilance: <1-5>
Agility: <1-5>
Resilience: <1-5>
Overall VAR Readiness: <Low / Moderate / High>
What the response does well: <3-5 concise sentences>
Primary gaps: <3-5 concise sentences>
Comparison to historical patterns: <2-4 concise sentences tied to the evidence provided>
Worst-case consequence if handled poorly: <1-3 concise sentences>
Best next improvement: <1-2 concise sentences>

Do not invent facts outside the provided evidence.
""".strip()


def chat_system_prompt(profession: str, description: str) -> str:
    return f"""
You are a practical analyst and career coach for {profession}.

Profession description:
{description}

Use the VAR framework:
- Vigilance
- Agility
- Resilience

Instructions:
- Answer using only the supplied dataset summary and evidence rows.
- Be practical, concise, and insightful.
- If the evidence is thin or ambiguous, say so directly.
- When useful, organize the answer under Vigilance / Agility / Resilience.
- Prefer grounded interpretation over generic career advice.
""".strip()


# =========================================================
# RENDER HELPERS
# =========================================================
def render_score_metrics(v: int | None, a: int | None, r: int | None) -> None:
    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric("Vigilance", f"{v}/5" if v is not None else "NA", score_label(v))
        if v is not None:
            st.progress(v / 5)

    with c2:
        st.metric("Agility", f"{a}/5" if a is not None else "NA", score_label(a))
        if a is not None:
            st.progress(a / 5)

    with c3:
        st.metric("Resilience", f"{r}/5" if r is not None else "NA", score_label(r))
        if r is not None:
            st.progress(r / 5)


def safe_metric(value, decimals: int = 1, prefix: str = "", suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{prefix}{value:,.{decimals}f}{suffix}"


def compact_number(value, decimals=1):
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    return f"{value:,.{decimals}f}"


def compact_currency(value):
    if value is None or pd.isna(value):
        return "NA"
    value = float(value)
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"${value / 1_000:.1f}K"
    return f"${value:,.0f}"


def metric_card(label, value, caption=None):
    caption_html = f"<div class='metric-caption'>{caption}</div>" if caption else ""
    st.markdown(
        f"""
        <div class="custom-metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {caption_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# LOAD DATA
# =========================================================
try:
    df_all = load_var_data(CSV_PATH)
except FileNotFoundError:
    st.error(
        "Could not find `data/var_main_deaths_capstone_ready.csv`. Make sure the cleaned CSV is saved in the data folder."
    )
    st.stop()
except Exception as e:
    st.error(f"Failed to load the dataset: {e}")
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Controls")

profession = st.sidebar.selectbox("Profession lens", list(PROFESSIONS.keys()))
mode = st.sidebar.radio("Mode", ["Overview", "Scenario Simulator", "Chat Coach", "Data Explorer"])

client = get_client()
model_name = st.sidebar.selectbox("Model", [DEFAULT_MODEL, "gpt-4o"], index=0)

base_df = filter_by_profession(df_all, profession)

cost_impact_options = []
if "cost_impact" in base_df.columns:
    cost_impact_options = [
        x for x in ["Low", "Medium", "High", "Unknown"]
        if x in set(base_df["cost_impact"].dropna().astype(str))
    ]

selected_cost_impacts = st.sidebar.multiselect(
    "Cost impact",
    options=cost_impact_options,
    default=[],
)

category_options = []
if "category" in base_df.columns:
    category_options = sorted(base_df["category"].dropna().astype(str).unique().tolist())

default_categories = PROFESSIONS[profession].get("suggested_categories", [])
selected_categories = st.sidebar.multiselect(
    "Category filter",
    options=category_options,
    default=[c for c in default_categories if c in category_options],
)

fatality_min = st.sidebar.slider("Minimum fatalities", 0, int(max(0, base_df["fatalities"].fillna(0).max())) if "fatalities" in base_df.columns and not base_df.empty else 100, 0)
keyword_filter = st.sidebar.text_input("Keyword filter", "")

filtered_df = apply_user_filters(
    base_df,
    selected_categories=selected_categories,
    fatality_min=fatality_min,
    keyword=keyword_filter,
    selected_cost_impacts=selected_cost_impacts,
)

st.sidebar.markdown("---")
if client is None:
    st.sidebar.warning("OpenAI key not found. Overview and Data Explorer will work, but AI features will be disabled.")
else:
    st.sidebar.success("OpenAI client ready")

st.sidebar.markdown(
    '<div class="small-muted">Tip: keep filters fairly broad so the AI has enough evidence to work with.</div>',
    unsafe_allow_html=True,
)

# =========================================================
# MAIN HEADER
# =========================================================
st.markdown(f"### {profession}")
st.write(PROFESSIONS[profession]["description"])

summary = summarize_dataset(filtered_df)
benchmark = build_benchmark(filtered_df)

# =========================================================
# MODE: OVERVIEW
# =========================================================
if mode == "Overview":
    metric_row1 = st.columns(4)

    with metric_row1[0]:
        metric_card("Records", f"{len(filtered_df):,}")

    with metric_row1[1]:
        metric_card("Avg VAR Score", safe_metric(summary.get("var_score_mean"), 1))

    with metric_row1[2]:
        metric_card("Avg Fatalities", compact_number(summary.get("fatalities_mean"), 1))

    with metric_row1[3]:
        metric_card("Avg Cost", compact_currency(summary.get("estimated_cost_2024_mean")))

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    metric_row2 = st.columns(4)

    with metric_row2[0]:
        metric_card("Avg Vigilance", safe_metric(summary.get("vigilance_mean"), 2))

    with metric_row2[1]:
        metric_card("Avg Agility", safe_metric(summary.get("agility_mean"), 2))

    with metric_row2[2]:
        metric_card("Avg Resilience", safe_metric(summary.get("resilience_mean"), 2))

    with metric_row2[3]:
        top_cat = "NA"
        if "category" in filtered_df.columns and not filtered_df.empty:
            top_cat = filtered_df["category"].value_counts().idxmax()
        metric_card("Top Category", top_cat)

    left, right = st.columns([1.2, 1])

    with left:
        st.markdown("#### Category distribution")
        st.markdown(
            "<div style='margin-top:10px;'></div>",
            unsafe_allow_html=True,
        )
        if "category" in filtered_df.columns and not filtered_df.empty:
            cat_counts = (
                filtered_df["category"]
                .value_counts()
                .reset_index()
            )
            cat_counts.columns = ["Category", "Count"]
            cat_counts = cat_counts.sort_values("Count", ascending=True)
            fig_cat = px.bar(
                cat_counts,
                x="Count",
                y="Category",
                orientation="h",
                color_discrete_sequence=["#166534"],
            )
            fig_cat.update_layout(
                showlegend=False,
                height=460,
                margin=dict(l=0, r=60, t=8, b=8),
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis=dict(
                    title="Count",
                    showgrid=True,
                    gridcolor="#E5E7EB",
                    zeroline=False,
                ),
                yaxis=dict(title="", showgrid=False),
            )
            fig_cat.update_traces(
                marker_line_width=0,
                text=cat_counts["Count"],
                textposition="outside",
                cliponaxis=False,
            )
            st.plotly_chart(fig_cat, use_container_width=True)
        else:
            st.info("No category data available for the current filters.")

    with right:
        st.markdown("#### Highest-impact events")
        display_cols = [
            c for c in ["event", "category", "fatalities", "estimated_cost_2024", "cost_impact", "var_score"]
            if c in filtered_df.columns
        ]
        if display_cols and not filtered_df.empty:
            sort_by = (
                ["fatalities", "var_score"]
                if "fatalities" in filtered_df.columns
                else [c for c in ["var_score"] if c in filtered_df.columns]
            )
            if sort_by:
                top_events = filtered_df.sort_values(
                    by=sort_by,
                    ascending=False,
                ).head(10)
            else:
                top_events = filtered_df.head(10)
            st.dataframe(
                top_events[display_cols],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No event rows available.")

    st.markdown("#### Interpretation")
    st.write(
        "This view gives you a profession-specific lens on the event dataset. Use it to identify dominant categories, typical VAR patterns, and which events drive the most severe outcomes under the current filters."
    )

# =========================================================
# MODE: SCENARIO SIMULATOR
# =========================================================
elif mode == "Scenario Simulator":
    scenario = st.selectbox("Choose a scenario", PROFESSIONS[profession]["scenarios"])

    st.markdown("#### Scenario")
    st.info(scenario)

    col_a, col_b = st.columns([1.2, 1])

    with col_a:
        user_response = st.text_area(
            "What would you do?",
            height=220,
            placeholder="Describe how you would assess risk, prioritize actions, communicate, adapt, and stabilize the situation.",
        )

    with col_b:
        st.markdown("#### Benchmark context")
        st.write(
            f"""
            - Records in current comparison set: **{len(filtered_df):,}**
            - Avg Vigilance: **{safe_metric(benchmark.get('vigilance_mean'), 2)}**
            - Avg Agility: **{safe_metric(benchmark.get('agility_mean'), 2)}**
            - Avg Resilience: **{safe_metric(benchmark.get('resilience_mean'), 2)}**
            - Avg VAR Score: **{safe_metric(benchmark.get('var_score_mean'), 1)}**
            """
        )

    evaluate = st.button("Evaluate Response", type="primary", use_container_width=True)

    if evaluate:
        if not user_response.strip():
            st.warning("Please enter a response first.")
        elif client is None:
            st.error("OpenAI API key not found. Add it in Streamlit secrets before using the evaluator.")
        elif filtered_df.empty:
            st.error("The current filters returned no data. Broaden the filters and try again.")
        else:
            relevant = retrieve_relevant_events(filtered_df, scenario + " " + user_response, top_n=5)
            prompt = scenario_prompt(
                profession=profession,
                scenario=scenario,
                user_response=user_response.strip(),
                dataset_summary=summary_text(filtered_df),
                benchmark=benchmark,
                evidence_rows=format_events_for_prompt(relevant, limit=5),
            )

            with st.spinner("Evaluating against VAR framework and historical patterns..."):
                try:
                    result = ask_model(
                        client=client,
                        model=model_name,
                        system_prompt="You evaluate decision quality using the VAR framework and the supplied event evidence.",
                        user_prompt=prompt,
                    )
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    st.stop()

            st.markdown("### Evaluation")
            v, a, r = extract_scores(result)
            render_score_metrics(v, a, r)

            st.markdown("#### Feedback")
            st.write(result)

            with st.expander("Evidence used for this evaluation", expanded=True):
                keep = [c for c in ["event", "category", "fatalities", "estimated_cost_2024", "cost_impact", "var_score", "vigilance", "agility", "resilience", "insight"] if c in relevant.columns]
                st.dataframe(relevant[keep], use_container_width=True, hide_index=True)

# =========================================================
# MODE: CHAT COACH
# =========================================================
elif mode == "Chat Coach":
    st.markdown("### Ask about this profession using the dataset")

    session_key = f"var_chat_{profession}"
    if session_key not in st.session_state:
        st.session_state[session_key] = [
            {
                "role": "assistant",
                "content": f"I’m ready to analyze {profession} using the current VAR event filters. Ask about patterns, risks, lessons, or what the data suggests.",
            }
        ]

    for msg in st.session_state[session_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input(f"Ask about {profession}...")

    if user_input:
        st.session_state[session_key].append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            if client is None:
                answer = "I can’t answer yet because the OpenAI API key is missing."
                st.error(answer)
            elif filtered_df.empty:
                answer = "The current filters returned no rows, so I don’t have evidence to answer from. Broaden the filters and try again."
                st.warning(answer)
            else:
                relevant = retrieve_relevant_events(filtered_df, user_input, top_n=6)

                system_prompt = chat_system_prompt(
                    profession=profession,
                    description=PROFESSIONS[profession]["description"],
                )

                user_prompt = f"""
User question:
{user_input}

Dataset summary:
{summary_text(filtered_df)}

Relevant evidence rows:
{format_events_for_prompt(relevant, limit=6)}

Answer using the evidence above.
"""

                with st.spinner("Thinking with your VAR data..."):
                    try:
                        answer = ask_model(
                            client=client,
                            model=model_name,
                            system_prompt=system_prompt,
                            user_prompt=user_prompt,
                        )
                        st.write(answer)
                    except Exception as e:
                        answer = f"Sorry, something went wrong: {e}"
                        st.error(answer)

                with st.expander("Evidence used", expanded=False):
                    keep = [c for c in ["event", "category", "fatalities", "estimated_cost_2024", "cost_impact", "var_score", "vigilance", "agility", "resilience", "insight"] if c in relevant.columns]
                    st.dataframe(relevant[keep], use_container_width=True, hide_index=True)

        st.session_state[session_key].append({"role": "assistant", "content": answer})

# =========================================================
# MODE: DATA EXPLORER
# =========================================================
elif mode == "Data Explorer":
    st.markdown("#### Filtered dataset")
    st.write(
        f"Showing **{len(filtered_df):,}** rows from the profession-filtered subset."
    )

    show_cols = [c for c in [
        "event_id", "event", "category", "start_date", "location",
        "fatalities", "var_score", "vigilance", "agility", "resilience",
        "duration_days", "estimated_cost_2024", "cost_impact", "cost_was_imputed",
        "cost_estimation_method", "insight", "source_title", "source_url"
    ] if c in filtered_df.columns]

    st.dataframe(filtered_df[show_cols], use_container_width=True, hide_index=True)

    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered data as CSV",
        data=csv,
        file_name=f"var_filtered_{profession.lower().replace(' ', '_')}.csv",
        mime="text/csv",
    )

# =========================================================
# FOOTER / NOTES
# =========================================================
with st.expander("About this prototype"):
    st.write(
        """
        This simulator uses your cleaned capstone-ready VAR event dataset as evidence for both chat answers and scenario evaluation.
        The dataset includes a unified fatalities column and estimated 2024 cost fields, including cost impact and imputation flags.
        The profession lens is a structured interpretation layer over the underlying event categories.
        For a stronger capstone version, the next upgrade would be:
        1. custom profession mappings,
        2. scenario-to-event matching tuned by your sponsor,
        3. better charts,
        4. saved conversation history,
        5. explicit citations to source titles and URLs in each answer.
        """
    )
