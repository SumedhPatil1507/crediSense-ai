"""Shared visual defaults for the Streamlit dashboard and Plotly charts."""
import plotly.io as pio
import streamlit as st


def configure_dashboard() -> None:
    """Apply consistent dark charts and compact dashboard styling."""
    pio.templates.default = "plotly_dark"
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1.8rem; padding-bottom: 2.5rem; }
        [data-testid="stMetric"] {
            background: linear-gradient(145deg, rgba(25, 43, 63, .92), rgba(16, 29, 46, .92));
            border: 1px solid rgba(111, 150, 177, .18);
            border-radius: 14px;
            padding: 14px 16px;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #43d9c3;
            border-bottom-color: #43d9c3;
        }
        div[data-testid="stAlert"] { border-radius: 12px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
