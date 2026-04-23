import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(BASE_DIR))
data_path = BASE_DIR / "data" / "loan_cleaned.csv"

from src.live_data import get_macro_indicators, get_rbi_repo_rate, get_all_news, FEEDPARSER_AVAILABLE

st.set_page_config(layout="wide")
st.title("📊 CrediSense — Data Intelligence Hub")

# ── Load dataset ───────────────────────────────────────────────────────────────
@st.cache_data
def load_df():
    df = pd.read_csv(data_path, sep=None, engine="python")
    df.columns = df.columns.str.strip()
    return df

df_raw = load_df()
df = df_raw.copy()
df["Risk_Flag"] = df["Risk_Flag"].astype(str)
df_sample = df.sample(n=10000, random_state=42)

COLORS = {"0": "#28a745", "1": "#dc3545"}

tabs = st.tabs([
    "📋 Overview",
    "🔍 Deep Dive",
    "🗺️ Geographic",
    "🌐 Live Macro",
    "📰 News Feed",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Records",  f"{len(df):,}")
    c2.metric("Default Rate",   f"{(df['Risk_Flag']=='1').mean()*100:.2f}%")
    c3.metric("Avg Income",     f"{df['Income'].mean():.3f}")
    c4.metric("Avg Experience", f"{df['Experience'].mean():.3f}")
    c5.metric("Unique Professions", df["Profession"].nunique() if "Profession" in df.columns else "N/A")

    st.markdown("---")
    col_l, col_r = st.columns(2)

    with col_l:
        # Animated pie
        fig_pie = px.pie(df, names="Risk_Flag", color="Risk_Flag",
                         color_discrete_map=COLORS,
                         title="Safe vs Default Distribution",
                         hole=0.4)
        fig_pie.update_traces(textposition="inside", textinfo="percent+label",
                               pull=[0, 0.05])
        fig_pie.update_layout(showlegend=True, height=380)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        # Ownership grouped bar
        if "House_Ownership" in df.columns:
            grp = df.groupby(["House_Ownership", "Risk_Flag"]).size().reset_index(name="count")
            total = grp.groupby("House_Ownership")["count"].transform("sum")
            grp["pct"] = (grp["count"] / total * 100).round(1)
            fig_own = px.bar(grp, x="House_Ownership", y="pct", color="Risk_Flag",
                             barmode="group", color_discrete_map=COLORS,
                             text="pct",
                             title="Default Rate by House Ownership (%)",
                             labels={"pct": "% of Group"})
            fig_own.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_own.update_layout(height=380)
            st.plotly_chart(fig_own, use_container_width=True)

    st.markdown("---")
    col2_l, col2_r = st.columns(2)

    with col2_l:
        fig_inc = px.violin(df_sample, x="Risk_Flag", y="Income", color="Risk_Flag",
                            color_discrete_map=COLORS, box=True, points=False,
                            title="Income Distribution by Risk (violin + box)")
        fig_inc.update_layout(height=380)
        st.plotly_chart(fig_inc, use_container_width=True)

    with col2_r:
        fig_exp = px.violin(df_sample, x="Risk_Flag", y="Experience", color="Risk_Flag",
                            color_discrete_map=COLORS, box=True, points=False,
                            title="Experience Distribution by Risk")
        fig_exp.update_layout(height=380)
        st.plotly_chart(fig_exp, use_container_width=True)

    # Correlation heatmap
    st.markdown("---")
    st.subheader("🔥 Feature Correlation Heatmap")
    num_cols = df_sample.select_dtypes(include=[float, int]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ["Id"]]
    corr = df_sample[num_cols].copy()
    corr["Risk_Flag"] = df_sample["Risk_Flag"].astype(int)
    corr_matrix = corr.corr().round(2)
    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r",
                          zmin=-1, zmax=1, title="Pearson Correlation Matrix",
                          aspect="auto")
    fig_corr.update_layout(height=500)
    st.plotly_chart(fig_corr, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DEEP DIVE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.subheader("🔍 Interactive Deep Dive")

    # Dynamic feature selector
    num_features = df_sample.select_dtypes(include=[float, int]).columns.tolist()
    num_features = [c for c in num_features if c not in ["Id"]]

    d1, d2 = st.columns(2)
    with d1:
        x_feat = st.selectbox("X-axis feature", num_features, index=0)
        plot_type = st.radio("Plot type", ["Histogram", "Box", "Violin", "Scatter"], horizontal=True)
    with d2:
        y_feat = st.selectbox("Y-axis feature (scatter only)", num_features, index=1)
        color_by_risk = st.checkbox("Color by Risk Flag", value=True)

    color_arg = "Risk_Flag" if color_by_risk else None

    if plot_type == "Histogram":
        fig_d = px.histogram(df_sample, x=x_feat, color=color_arg,
                              color_discrete_map=COLORS, nbins=40, barmode="overlay",
                              opacity=0.75, title=f"Distribution of {x_feat}")
    elif plot_type == "Box":
        fig_d = px.box(df_sample, x="Risk_Flag" if color_by_risk else None,
                        y=x_feat, color=color_arg, color_discrete_map=COLORS,
                        title=f"{x_feat} by Risk")
    elif plot_type == "Violin":
        fig_d = px.violin(df_sample, x="Risk_Flag" if color_by_risk else None,
                           y=x_feat, color=color_arg, color_discrete_map=COLORS,
                           box=True, title=f"{x_feat} by Risk")
    else:
        fig_d = px.scatter(df_sample, x=x_feat, y=y_feat, color=color_arg,
                            color_discrete_map=COLORS, opacity=0.4,
                            title=f"{x_feat} vs {y_feat}",
                            trendline="ols" if not color_by_risk else None)

    fig_d.update_layout(height=420)
    st.plotly_chart(fig_d, use_container_width=True)

    st.markdown("---")

    # Profession analysis
    if "Profession" in df.columns:
        st.subheader("💼 Profession Risk Analysis")
        prof_default = (df[df["Risk_Flag"]=="1"].groupby("Profession").size() /
                        df.groupby("Profession").size()).dropna().sort_values(ascending=False)
        prof_count = df.groupby("Profession").size()

        top_n = st.slider("Show top N professions", 5, 30, 15)
        prof_df = pd.DataFrame({
            "Default Rate": prof_default.head(top_n),
            "Count": prof_count[prof_default.head(top_n).index]
        }).reset_index()
        prof_df.columns = ["Profession", "Default Rate", "Count"]

        fig_prof = px.bar(prof_df, x="Default Rate", y="Profession",
                          orientation="h", color="Default Rate",
                          color_continuous_scale="Reds",
                          size="Count" if "Count" in prof_df.columns else None,
                          title=f"Top {top_n} Professions by Default Rate",
                          text="Default Rate")
        fig_prof.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        fig_prof.update_layout(height=max(400, top_n * 28), yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig_prof, use_container_width=True)

    st.markdown("---")

    # Car & Marital status
    cat_col1, cat_col2 = st.columns(2)
    with cat_col1:
        if "Car_Ownership" in df.columns:
            car_grp = df.groupby(["Car_Ownership", "Risk_Flag"]).size().reset_index(name="count")
            fig_car = px.bar(car_grp, x="Car_Ownership", y="count", color="Risk_Flag",
                              barmode="group", color_discrete_map=COLORS,
                              title="Car Ownership vs Default")
            st.plotly_chart(fig_car, use_container_width=True)

    with cat_col2:
        if "Married/Single" in df.columns:
            mar_grp = df.groupby(["Married/Single", "Risk_Flag"]).size().reset_index(name="count")
            fig_mar = px.bar(mar_grp, x="Married/Single", y="count", color="Risk_Flag",
                              barmode="group", color_discrete_map=COLORS,
                              title="Marital Status vs Default")
            st.plotly_chart(fig_mar, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — GEOGRAPHIC
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.subheader("🗺️ Geographic Risk Analysis")

    if "STATE" in df.columns:
        state_default = (df[df["Risk_Flag"]=="1"].groupby("STATE").size() /
                         df.groupby("STATE").size()).dropna().reset_index()
        state_default.columns = ["STATE", "Default Rate"]
        state_count = df.groupby("STATE").size().reset_index(name="Count")
        state_df = state_default.merge(state_count, on="STATE").sort_values("Default Rate", ascending=False)

        g1, g2 = st.columns(2)
        with g1:
            fig_state = px.bar(state_df.head(20), x="Default Rate", y="STATE",
                                orientation="h", color="Default Rate",
                                color_continuous_scale="Reds",
                                title="Top 20 States by Default Rate",
                                text="Default Rate")
            fig_state.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig_state.update_layout(height=550, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_state, use_container_width=True)

        with g2:
            fig_bubble = px.scatter(state_df, x="Count", y="Default Rate",
                                     size="Count", color="Default Rate",
                                     color_continuous_scale="Reds",
                                     hover_name="STATE", text="STATE",
                                     title="State: Volume vs Default Rate")
            fig_bubble.update_traces(textposition="top center")
            fig_bubble.update_layout(height=550)
            st.plotly_chart(fig_bubble, use_container_width=True)

    if "CITY" in df.columns:
        st.markdown("---")
        st.subheader("🏙️ Top Cities by Default Rate")
        city_default = (df[df["Risk_Flag"]=="1"].groupby("CITY").size() /
                        df.groupby("CITY").size()).dropna()
        city_count = df.groupby("CITY").size()
        city_df = pd.DataFrame({"Default Rate": city_default, "Count": city_count}).reset_index()
        city_df.columns = ["CITY", "Default Rate", "Count"]
        city_df = city_df[city_df["Count"] >= 100].sort_values("Default Rate", ascending=False).head(25)

        fig_city = px.treemap(city_df, path=["CITY"], values="Count",
                               color="Default Rate", color_continuous_scale="Reds",
                               title="City Treemap — Size=Volume, Color=Default Rate")
        fig_city.update_layout(height=500)
        st.plotly_chart(fig_city, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LIVE MACRO
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.subheader("🌐 India Macroeconomic Dashboard")
    st.caption("Live data from World Bank Open Data API · Cached 1 hour")

    with st.spinner("Fetching live indicators..."):
        macro = get_macro_indicators()
        repo_rates = get_rbi_repo_rate()

    # KPI row
    kpi_cols = st.columns(len(macro))
    for i, (name, df_m) in enumerate(macro.items()):
        if not df_m.empty:
            latest = df_m.iloc[-1]
            prev = df_m.iloc[-2] if len(df_m) > 1 else latest
            delta = round(float(latest["value"]) - float(prev["value"]), 2)
            kpi_cols[i].metric(name.split("(")[0].strip(),
                                f"{latest['value']:.2f}%",
                                delta=f"{delta:+.2f}%")

    st.markdown("---")

    # Combined macro chart
    st.subheader("📈 Multi-Indicator Trend")
    selected_indicators = st.multiselect(
        "Select indicators to compare",
        list(macro.keys()),
        default=["GDP Growth (%)", "Inflation CPI (%)", "Bank NPA Ratio (%)"]
    )

    if selected_indicators:
        fig_multi = go.Figure()
        for name in selected_indicators:
            df_m = macro[name]
            if not df_m.empty:
                fig_multi.add_trace(go.Scatter(
                    x=df_m["year"], y=df_m["value"],
                    name=name, mode="lines+markers",
                    line=dict(width=2)
                ))
        fig_multi.update_layout(
            title="India Macro Indicators Over Time",
            xaxis_title="Year", yaxis_title="Value (%)",
            height=420, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02)
        )
        st.plotly_chart(fig_multi, use_container_width=True)

    st.markdown("---")

    # Individual charts in grid
    chart_cols = st.columns(2)
    for i, (name, df_m) in enumerate(macro.items()):
        with chart_cols[i % 2]:
            if not df_m.empty:
                fig = px.area(df_m, x="year", y="value",
                               title=f"India: {name}",
                               labels={"value": name, "year": "Year"})
                fig.update_traces(line_color="#1f77b4", fillcolor="rgba(31,119,180,0.15)")
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # RBI Repo Rate
    st.subheader("🏦 RBI Repo Rate History")
    st.caption("Higher repo rate = costlier credit = higher default risk")
    df_repo = pd.DataFrame(repo_rates)
    fig_repo = go.Figure()
    fig_repo.add_trace(go.Scatter(
        x=df_repo["date"], y=df_repo["rate"],
        mode="lines+markers+text",
        text=df_repo["rate"].apply(lambda x: f"{x}%"),
        textposition="top center",
        line=dict(color="#e74c3c", width=2),
        fill="tozeroy", fillcolor="rgba(231,76,60,0.1)"
    ))
    fig_repo.update_layout(title="RBI Repo Rate (%)", xaxis_title="Date",
                            yaxis_title="Rate (%)", height=380,
                            yaxis_range=[3, 8])
    st.plotly_chart(fig_repo, use_container_width=True)

    # Macro-Risk correlation insight
    st.markdown("---")
    st.subheader("💡 Macro-Risk Insight")
    npa_df = macro.get("Bank NPA Ratio (%)", pd.DataFrame())
    gdp_df = macro.get("GDP Growth (%)", pd.DataFrame())
    if not npa_df.empty and not gdp_df.empty:
        merged = npa_df.merge(gdp_df, on="year", suffixes=("_npa", "_gdp"))
        fig_scatter = px.scatter(merged, x="value_gdp", y="value_npa",
                                  text="year", trendline="ols",
                                  labels={"value_gdp": "GDP Growth (%)",
                                          "value_npa": "Bank NPA Ratio (%)"},
                                  title="GDP Growth vs Bank NPA Ratio (India)")
        fig_scatter.update_traces(textposition="top center")
        fig_scatter.update_layout(height=380)
        st.plotly_chart(fig_scatter, use_container_width=True)
        st.caption("Negative correlation: higher GDP growth → lower NPA ratio. "
                   "Source: [World Bank Open Data](https://data.worldbank.org)")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — NEWS FEED
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.subheader("📰 Live Credit Risk & Banking News")

    if not FEEDPARSER_AVAILABLE:
        st.warning("Install `feedparser` to enable live news: it's in requirements.txt")
    else:
        from src.live_data import NEWS_FEEDS, fetch_news

        feed_choice = st.selectbox("Select news source", list(NEWS_FEEDS.keys()))
        max_items = st.slider("Number of articles", 3, 15, 8)

        with st.spinner(f"Fetching from {feed_choice}..."):
            articles = fetch_news(feed_choice, max_items=max_items)

        if not articles:
            st.warning("Could not fetch articles. The feed may be temporarily unavailable.")
        else:
            st.caption(f"Showing {len(articles)} articles · Cached 30 min · Source: {feed_choice}")
            st.markdown("---")
            for art in articles:
                with st.container():
                    col_t, col_l = st.columns([5, 1])
                    with col_t:
                        st.markdown(f"**[{art['title']}]({art['link']})**")
                        if art["published"]:
                            st.caption(f"🕐 {art['published']}")
                        if art["summary"]:
                            st.write(art["summary"])
                    with col_l:
                        st.link_button("Read →", art["link"])
                    st.markdown("---")

        # All feeds combined
        st.subheader("🔄 All Sources Combined")
        if st.button("Load all news feeds"):
            with st.spinner("Fetching all feeds..."):
                all_news = get_all_news()
            if all_news:
                df_news = pd.DataFrame(all_news)[["title", "source", "published", "link"]]
                st.dataframe(df_news, use_container_width=True,
                             column_config={"link": st.column_config.LinkColumn("Link")})
            else:
                st.warning("No articles fetched.")
