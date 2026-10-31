"""Streamlit frontend for the restaurant rating prediction model."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.restaurant_rating_prediction import (
    CUISINE_FEATURE_PREFIX,
    DISPLAY_RATING_UPLIFT,
    apply_display_uplift,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "artifacts" / "best_model.joblib"
METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics.json"
FEATURE_METADATA_PATH = PROJECT_ROOT / "artifacts" / "feature_metadata.json"

st.set_page_config(
    page_title="TableScore | Restaurant Rating Predictor",
    page_icon="🍴",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #19332a;
        --muted: #718078;
        --green: #176b52;
        --line: #e7ebe5;
        --paper: #fffefa;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(233, 241, 225, .75), transparent 34rem),
            #f7f8f4;
        color: var(--ink);
    }
    .block-container { max-width: 1160px; padding: 2rem 2rem 4rem; }
    #MainMenu, footer, .stDeployButton { visibility: hidden; }
    [data-testid="stHeader"] { background: transparent; }
    .brandline {
        display: flex; align-items: center; gap: .65rem;
        color: #355347; font-size: .9rem; font-weight: 750;
        letter-spacing: .12em; text-transform: uppercase; margin: 0 0 1.2rem .15rem;
    }
    .brandmark {
        display: grid; place-items: center; height: 2.15rem; width: 2.15rem;
        border-radius: 11px; background: #176b52; color: white; font-size: 1rem;
        box-shadow: 0 5px 14px rgba(23, 107, 82, .2);
    }
    .hero {
        position: relative; overflow: hidden; min-height: 245px;
        display: flex; align-items: center; justify-content: space-between;
        gap: 2rem; padding: 2.8rem 3rem; margin-bottom: 1.25rem;
        color: white; border-radius: 24px;
        background: linear-gradient(112deg, #123f32 0%, #176b52 60%, #278061 100%);
        box-shadow: 0 18px 42px rgba(24, 70, 53, .16);
    }
    .hero:after {
        content: ""; position: absolute; width: 360px; height: 360px; right: 3%;
        top: -190px; border: 1px solid rgba(255,255,255,.13); border-radius: 50%;
        box-shadow: 0 0 0 34px rgba(255,255,255,.035), 0 0 0 72px rgba(255,255,255,.025);
    }
    .hero-copy { position: relative; z-index: 1; max-width: 650px; }
    .eyebrow {
        color: #bde8cf; text-transform: uppercase; letter-spacing: .16em;
        font-size: .72rem; font-weight: 750; margin-bottom: .85rem;
    }
    .hero h1 {
        color: white; font-size: clamp(2.4rem, 5vw, 3.65rem); line-height: 1.04;
        letter-spacing: -.055em; margin: 0 0 .85rem; font-weight: 760;
    }
    .hero p { color: rgba(255,255,255,.78); font-size: 1.04rem; line-height: 1.6; margin: 0; max-width: 560px; }
    .hero-badge {
        position: relative; z-index: 1; flex: 0 0 170px; height: 170px;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        border: 1px solid rgba(255,255,255,.24); border-radius: 50%;
        background: rgba(255,255,255,.09); backdrop-filter: blur(6px);
    }
    .hero-badge span:first-child { font-size: 2.5rem; }
    .hero-badge span:last-child { color: #d7eee0; font-size: .78rem; font-weight: 650; margin-top: .35rem; letter-spacing: .06em; }
    .section-heading { color: var(--ink); font-size: 1.35rem; font-weight: 740; letter-spacing: -.025em; margin: 1.6rem 0 .25rem; }
    .section-subtitle { color: var(--muted); font-size: .94rem; margin: 0 0 1rem; }
    [data-testid="stMetric"] {
        background: var(--paper); border: 1px solid var(--line); border-radius: 16px;
        padding: 1rem 1.2rem; box-shadow: 0 5px 16px rgba(36, 61, 47, .035);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); font-size: .78rem; font-weight: 650; }
    [data-testid="stMetricValue"] { color: var(--ink); font-size: 1.45rem; font-weight: 740; }
    [data-testid="stForm"] {
        background: var(--paper); border: 1px solid var(--line); border-radius: 20px;
        padding: 1.45rem 1.5rem 1.3rem; box-shadow: 0 9px 28px rgba(36, 61, 47, .045);
    }
    [data-testid="stForm"] label p { color: #334d41; font-weight: 620; }
    [data-testid="stForm"] input, [data-testid="stForm"] [data-baseweb="select"] > div {
        border-radius: 10px;
    }
    [data-testid="stFormSubmitButton"] button {
        background: #176b52; border: 0; min-height: 3.2rem; border-radius: 11px;
        font-weight: 700; width: 100%; transition: all .18s ease;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background: #10563f; box-shadow: 0 7px 18px rgba(23, 107, 82, .2);
        transform: translateY(-1px);
    }
    .result-card {
        background: linear-gradient(110deg, #fff8ed, #fffefa);
        border: 1px solid #f0dfc7; border-radius: 18px; padding: 1.35rem 1.55rem;
        margin-top: 1rem; color: var(--ink);
    }
    .result-label { color: #8c7657; font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .result-value { color: #173c2d; font-size: 2.5rem; font-weight: 780; letter-spacing: -.05em; margin: .15rem 0 .7rem; }
    .rating-track { height: 8px; overflow: hidden; background: #eee5d9; border-radius: 20px; }
    .rating-fill { height: 100%; background: linear-gradient(90deg, #f1b451, #e87b45); border-radius: inherit; }
    .fineprint { color: #78847c; font-size: .82rem; line-height: 1.55; }
    @media (max-width: 700px) {
        .block-container { padding: 1.2rem 1rem 3rem; }
        .hero { min-height: auto; padding: 2rem 1.5rem; border-radius: 19px; }
        .hero-badge { display: none; }
        .hero h1 { font-size: 2.5rem; }
        [data-testid="stForm"] { padding: 1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

dark_mode = st.session_state.get("dark_mode", False)
if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(ellipse at 12% 0%, rgba(35, 75, 58, .38), transparent 34rem),
                #111a16;
            color: #e7eee9;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stMetric"], [data-testid="stForm"] {
            background: #1b2821; border-color: #304238;
            box-shadow: 0 8px 26px rgba(0, 0, 0, .15);
        }
        [data-testid="stMetricLabel"], .section-subtitle { color: #aab9af; }
        [data-testid="stMetricValue"], .section-heading { color: #edf4ef; }
        [data-testid="stForm"] label p { color: #d7e3da; }
        [data-testid="stForm"] input,
        [data-testid="stForm"] [data-baseweb="select"] > div,
        [data-testid="stForm"] [data-baseweb="input"] > div {
            color: #edf4ef; background-color: #24352b; border-color: #40574a;
        }
        [data-testid="stForm"] [data-testid="stNumberInput"] input,
        [data-testid="stForm"] [data-testid="stTextInput"] input {
            color: #edf4ef; background-color: #24352b;
        }
        [data-testid="stForm"] [data-testid="stNumberInput"] button {
            color: #edf4ef; background-color: #24352b; border-color: #40574a;
        }
        [data-testid="stForm"] [data-baseweb="select"] svg { fill: #d7e3da; }
        [data-testid="stForm"] button[kind="secondary"] {
            color: #e7eee9; background: #24352b; border-color: #40574a;
        }
        [data-testid="stExpander"] {
            color: #e7eee9; background: #1b2821;
            border: 1px solid #304238; border-radius: 12px;
        }
        [data-testid="stExpander"] details summary,
        [data-testid="stExpander"] details p { color: #d7e3da; }
        .result-card {
            background: linear-gradient(110deg, #332b20, #28251f);
            border-color: #5b4c37; color: #edf4ef;
        }
        .result-label { color: #d3b98e; }
        .result-value { color: #f7ebd8; }
        .rating-track { background: #514638; }
        .fineprint { color: #aab9af; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_feature_metadata() -> dict:
    return json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))


def humanize(column: str) -> str:
    return column.replace("_", " ").strip().title()


brand_column, theme_column = st.columns([5, 1])
with brand_column:
    st.markdown(
        '<div class="brandline"><span class="brandmark">TS</span> TableScore <span style="color:#94a299;font-weight:500;letter-spacing:0;text-transform:none">/ Restaurant intelligence</span></div>',
        unsafe_allow_html=True,
    )
with theme_column:
    st.toggle("Dark mode", key="dark_mode")

if not all(path.is_file() for path in (MODEL_PATH, METRICS_PATH, FEATURE_METADATA_PATH)):
    st.warning("A trained model or its metadata is missing. Train it first, then refresh this page.")
    st.code("python3 src/restaurant_rating_prediction.py")
    st.stop()

try:
    model = load_model()
    metrics_data = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    feature_metadata = load_feature_metadata()
except (OSError, ValueError, KeyError) as error:
    st.error(f"Could not load the model or its metadata: {error}")
    st.stop()

model_features = list(model.feature_names_in_)
if model_features != feature_metadata.get("feature_columns"):
    st.error(
        "The model and feature metadata do not match. Retrain the model and refresh this page."
    )
    st.code("python3 src/restaurant_rating_prediction.py")
    st.stop()

feature_specs = feature_metadata["features"]
row_count = int(feature_metadata["row_count"])
best_model = metrics_data.get("best_model", "trained model")
test_metrics = metrics_data.get("test_metrics", {})
best_scores = test_metrics.get(best_model, {})

st.markdown(
    """
    <section class="hero">
      <div class="hero-copy">
        <div class="eyebrow">A smarter seat at the table</div>
        <h1>Know the rating<br>before the first bite.</h1>
        <p>Explore the signals behind restaurant ratings. Shape a restaurant profile
        and get an instant estimate from a model trained on real local data.</p>
      </div>
      <div class="hero-badge"><span>✦</span><span>DATA-DRIVEN DINING</span></div>
    </section>
    """,
    unsafe_allow_html=True,
)

metric_columns = st.columns(4)
metric_columns[0].metric("Restaurants", f"{row_count:,}")
model_label = {
    "random_forest": "Forest",
    "linear_regression": "Linear",
}.get(best_model, str(best_model).replace("_", " ").title())
metric_columns[1].metric("Model", model_label)
metric_columns[2].metric("Test MAE", f"{best_scores.get('mae', float('nan')):.3f}")
metric_columns[3].metric("Test R²", f"{best_scores.get('r2', float('nan')):.3f}")

st.markdown(
    '<div class="section-heading">Build a restaurant profile</div>'
    '<div class="section-subtitle">Adjust the details below to see how the model rates the experience.</div>',
    unsafe_allow_html=True,
)

with st.form("rating_prediction_form"):
    columns = st.columns(2)
    prediction_values: dict[str, object] = {}
    cuisine_features = [
        name for name in model_features if name.startswith(CUISINE_FEATURE_PREFIX)
    ]

    for index, name in enumerate(model_features):
        if name.startswith(CUISINE_FEATURE_PREFIX):
            continue
        feature_spec = feature_specs[name]
        label = humanize(name)
        with columns[index % 2]:
            if feature_spec["kind"] == "numeric":
                default = float(feature_spec["default"])
                is_count = any(token in name for token in ("votes", "ratings", "count"))
                prediction_values[name] = st.number_input(
                    label,
                    min_value=0.0,
                    value=default,
                    step=1.0 if is_count else 100.0,
                    help=f"Training data range: {feature_spec['minimum']:,.0f}–{feature_spec['maximum']:,.0f}",
                )
            else:
                options = feature_spec["options"]
                if len(options) <= 100:
                    prediction_values[name] = st.selectbox(
                        label, options, index=options.index(feature_spec["default"])
                    )
                else:
                    prediction_values[name] = st.text_input(
                        label,
                        value=feature_spec["default"],
                        help="You can enter a value that is not in the training data.",
                    )

    if cuisine_features:
        cuisine_options = [
            name[len(CUISINE_FEATURE_PREFIX) :] for name in cuisine_features
        ]
        cuisine_frequency = sorted(
            cuisine_features,
            key=lambda name: feature_metadata["cuisine_frequency"].get(name, 0),
            reverse=True,
        )
        default_cuisines = [
            name[len(CUISINE_FEATURE_PREFIX) :]
            for name in cuisine_frequency[:2]
        ]
        selected_cuisines = st.multiselect(
            "Cuisine types",
            options=cuisine_options,
            default=default_cuisines,
            placeholder="Search and select one or more cuisines",
            help="Choose every cuisine this restaurant serves.",
        )
        selected_set = set(selected_cuisines)
        prediction_values.update(
            {
                name: int(name[len(CUISINE_FEATURE_PREFIX) :] in selected_set)
                for name in cuisine_features
            }
        )

    st.markdown(
        '<div class="fineprint">Your estimate is based on the patterns in the training data. It is not a guarantee of future customer reviews.</div>',
        unsafe_allow_html=True,
    )
    submitted = st.form_submit_button("✦  Estimate restaurant rating", type="primary")

if submitted:
    try:
        input_frame = pd.DataFrame([prediction_values], columns=model_features)
        model_prediction = min(5.0, max(0.0, float(model.predict(input_frame)[0])))
        prediction = apply_display_uplift(model_prediction)
        applied_uplift = prediction - model_prediction
    except (ValueError, TypeError) as error:
        st.error(f"Could not generate a prediction: {error}")
    else:
        fill_width = prediction / 5 * 100
        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-label">Customer-friendly display score</div>
              <div class="result-value">{prediction:.2f}<span style="font-size:1.1rem;color:#8c7657;font-weight:600"> / 5</span></div>
              <div class="rating-track"><div class="rating-fill" style="width:{fill_width:.1f}%"></div></div>
              <div class="fineprint" style="margin-top:.8rem">
                Model prediction: {model_prediction:.2f} / 5
                &nbsp;·&nbsp; Display adjustment: +{applied_uplift:.2f}
                (up to +{DISPLAY_RATING_UPLIFT:.2f}, capped at 5.00)
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with st.expander("About this estimate"):
    st.write(
        f"The model was trained using {row_count:,} restaurants. "
        f"On a held-out test set, its mean absolute error was "
        f"{best_scores.get('mae', float('nan')):.3f} rating points."
    )
    st.caption(
        "Predictions may be less reliable for details or categories rarely seen "
        "in the training dataset."
    )

st.markdown(
    '<div class="fineprint" style="text-align:center;margin-top:2.2rem">TABLESCORE · A restaurant rating estimate, made with machine learning</div>',
    unsafe_allow_html=True,
)
