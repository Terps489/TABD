"""Interactive Dash dashboard for TFT analysis of Tatneft gas stations."""
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc

from src.config import (
    DETAILED_DATA, FIVE_STATIONS_DATA, STATIONS_META,
    OUTPUTS_DIR, MODELS_DIR, TARGETS,
    DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG, USE_5_STATIONS
)
from src.predict import generate_recommendations

warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────────
_data_path = FIVE_STATIONS_DATA if USE_5_STATIONS else DETAILED_DATA
df_raw = pd.read_csv(_data_path, parse_dates=["timestamp"])
df_meta = pd.read_csv(STATIONS_META)

STATIONS = sorted(df_raw["station_name"].unique().tolist())
FUEL_COLS = ["sales_AI92", "sales_AI95", "sales_AI98",
             "sales_DT_EURO", "sales_DT_TANEKO", "sales_DT_SUMMER", "sales_DT_WINTER"]
FUEL_LABELS = ["АИ-92", "АИ-95", "АИ-98", "ДТ Евро+", "ДТ ТАНЕКО", "ДТ Летнее", "ДТ Зимнее"]
FUEL_COLORS = px.colors.qualitative.Plotly[:7]

SHOP_COLS = ["shop_напитки", "shop_закуски", "shop_автотовары", "shop_кофе", "shop_табак"]


def _kpi_cards():
    total_sales = df_raw["total_fuel_sales"].sum()
    avg_hourly = df_raw["total_fuel_sales"].mean()
    n_stations = df_raw["station_id"].nunique()
    return dbc.Card(style=CARD, children=[
        dbc.Row([
            dbc.Col(html.Div([html.H6("АЗС", style={"color": "#aaa"}),
                              html.H4(f"{n_stations}", style={"color": "#17a2b8"})]), width=4),
            dbc.Col(html.Div([html.H6("Продажи", style={"color": "#aaa"}),
                              html.H4(f"{total_sales/1e6:.1f}M л", style={"color": "#28a745"})]), width=4),
            dbc.Col(html.Div([html.H6("Ср./час", style={"color": "#aaa"}),
                              html.H4(f"{avg_hourly:.0f} л", style={"color": "#ffc107"})]), width=4),
        ])
    ])


def _dark_layout(title: str) -> dict:
    return dict(
        title=dict(text=title, font=dict(color="#ccc", size=13)),
        paper_bgcolor="#1e1e2e",
        plot_bgcolor="#1e1e2e",
        font=dict(color="#ccc"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)"),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
    )


# ── App layout ─────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="Татнефть АЗС — TFT Dashboard",
)

CARD = {"borderRadius": "8px", "padding": "16px", "marginBottom": "16px"}

app.layout = dbc.Container(fluid=True, children=[
    # Header
    dbc.Row([
        dbc.Col(html.H2("Татнефть АЗС — TFT Аналитика", style={"color": "#17a2b8"}), width=8),
        dbc.Col(html.P("Анализ продаж топлива | Temporal Fusion Transformer",
                       style={"color": "#888", "marginTop": "10px", "textAlign": "right"}), width=4),
    ], className="mb-3 mt-3"),

    dbc.Tabs([

        # ── Tab 1: Overview ────────────────────────────────────────────────────
        dbc.Tab(label="Обзор сети", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.Label("Период", style={"color": "#aaa"}),
                        dcc.DatePickerRange(
                            id="overview-date-range",
                            min_date_allowed=df_raw["timestamp"].min().date(),
                            max_date_allowed=df_raw["timestamp"].max().date(),
                            start_date=df_raw["timestamp"].min().date(),
                            end_date=df_raw["timestamp"].max().date(),
                            display_format="DD.MM.YYYY",
                        ),
                    ])
                ], width=4),
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.Label("Агрегация", style={"color": "#aaa"}),
                        dcc.Dropdown(
                            id="overview-agg",
                            options=[
                                {"label": "По дням", "value": "D"},
                                {"label": "По неделям", "value": "W"},
                                {"label": "По месяцам", "value": "ME"},
                            ],
                            value="D", clearable=False,
                            style={"backgroundColor": "#333", "color": "#fff"},
                        ),
                    ])
                ], width=4),
                dbc.Col(_kpi_cards(), width=4),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="overview-fuel-trend"), width=8),
                dbc.Col(dcc.Graph(id="overview-fuel-pie"), width=4),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="overview-traffic"), width=6),
                dbc.Col(dcc.Graph(id="overview-shop"), width=6),
            ]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="overview-heatmap"), width=12),
            ]),
        ]),

        # ── Tab 2: Station Analysis ────────────────────────────────────────────
        dbc.Tab(label="Анализ АЗС", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("Выберите АЗС", style={"color": "#aaa"}),
                    dcc.Dropdown(
                        id="station-select",
                        options=[{"label": s, "value": s} for s in STATIONS],
                        value=STATIONS[0], clearable=False,
                        style={"backgroundColor": "#333"},
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Вид топлива", style={"color": "#aaa"}),
                    dcc.Checklist(
                        id="fuel-checklist",
                        options=[{"label": f" {l}", "value": c}
                                 for l, c in zip(FUEL_LABELS, FUEL_COLS)],
                        value=["sales_AI95", "sales_AI92"],
                        inline=True,
                        style={"color": "#ccc"},
                    ),
                ], width=8),
            ], className="mb-3 mt-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="station-sales"), width=12)]),
            dbc.Row([
                dbc.Col(dcc.Graph(id="station-hourly"), width=6),
                dbc.Col(dcc.Graph(id="station-weekly"), width=6),
            ]),
            dbc.Row([dbc.Col(dcc.Graph(id="station-weather-impact"), width=12)]),
        ]),

        # ── Tab 3: Forecasts ───────────────────────────────────────────────────
        dbc.Tab(label="Прогнозы TFT", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("АЗС для прогноза", style={"color": "#aaa"}),
                    dcc.Dropdown(
                        id="forecast-station",
                        options=[{"label": s, "value": s} for s in STATIONS],
                        value=STATIONS[0], clearable=False,
                        style={"backgroundColor": "#333"},
                    ),
                ], width=4),
                dbc.Col([
                    html.Label("Показатель", style={"color": "#aaa"}),
                    dcc.Dropdown(
                        id="forecast-target",
                        options=[{"label": t.replace("_", " ").upper(), "value": t}
                                 for t in TARGETS],
                        value="total_fuel_sales", clearable=False,
                        style={"backgroundColor": "#333"},
                    ),
                ], width=4),
            ], className="mb-3 mt-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="forecast-chart"), width=12)]),
            dbc.Row([dbc.Col(
                dbc.Alert(
                    id="forecast-status",
                    color="info",
                    children="Запустите обучение (run.py --mode train) для генерации прогнозов TFT.",
                ), width=12
            )]),
        ]),

        # ── Tab 4: Factor Analysis ─────────────────────────────────────────────
        dbc.Tab(label="Факторный анализ", children=[
            dbc.Row([
                dbc.Col(dcc.Graph(id="factor-importance"), width=7),
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("Влияние факторов", style={"color": "#17a2b8"}),
                        dcc.Graph(id="factor-promo"),
                    ])
                ], width=5),
            ], className="mt-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="factor-competitor"), width=12)]),
        ]),

        # ── Tab 5: Recommendations ─────────────────────────────────────────────
        dbc.Tab(label="Рекомендации", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H4("Рекомендации на основе TFT", style={"color": "#17a2b8"}),
                        html.Div(id="recommendations-text"),
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("Топ АЗС по продажам", style={"color": "#17a2b8"}),
                        dcc.Graph(id="rec-top-stations"),
                    ])
                ], width=6),
            ], className="mt-3"),
            dbc.Row([dbc.Col(dcc.Graph(id="rec-seasonal"), width=12)]),
        ]),
    ]),
], style={"backgroundColor": "#1a1a2e", "minHeight": "100vh"})


# ── Callbacks: Overview ────────────────────────────────────────────────────────
@app.callback(
    Output("overview-fuel-trend", "figure"),
    Output("overview-fuel-pie", "figure"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
    Input("overview-agg", "value"),
)
def update_overview(start, end, agg):
    mask = (df_raw["timestamp"] >= start) & (df_raw["timestamp"] <= end)
    d = df_raw[mask].copy()
    d_agg = d.groupby(pd.Grouper(key="timestamp", freq=agg))[FUEL_COLS].sum().reset_index()

    # Trend chart
    fig_trend = go.Figure()
    for col, label, color in zip(FUEL_COLS, FUEL_LABELS, FUEL_COLORS):
        fig_trend.add_trace(go.Scatter(
            x=d_agg["timestamp"], y=d_agg[col],
            name=label, line=dict(color=color), mode="lines"
        ))
    fig_trend.update_layout(**_dark_layout("Динамика продаж топлива (литры)"))

    # Pie chart
    totals = d[FUEL_COLS].sum()
    fig_pie = go.Figure(go.Pie(
        labels=FUEL_LABELS, values=totals.values,
        marker_colors=FUEL_COLORS, hole=0.4,
    ))
    fig_pie.update_layout(**_dark_layout("Структура продаж"))

    return fig_trend, fig_pie


@app.callback(
    Output("overview-traffic", "figure"),
    Output("overview-shop", "figure"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
    Input("overview-agg", "value"),
)
def update_traffic_shop(start, end, agg):
    mask = (df_raw["timestamp"] >= start) & (df_raw["timestamp"] <= end)
    d = df_raw[mask].copy()

    traffic_cols = ["traffic_Passengers_cars", "traffic_Truck_short", "traffic_Truck",
                    "traffic_Truck_long", "traffic_Transporter"]
    traffic_labels = ["Легковые", "Малые грузовые", "Грузовые", "Тяжелые грузовые", "Спецтехника"]

    d_agg = d.groupby(pd.Grouper(key="timestamp", freq=agg))[traffic_cols + SHOP_COLS].sum().reset_index()

    fig_traffic = go.Figure()
    for col, label in zip(traffic_cols, traffic_labels):
        fig_traffic.add_trace(go.Bar(x=d_agg["timestamp"], y=d_agg[col], name=label))
    fig_traffic.update_layout(barmode="stack", **_dark_layout("Трафик по типам транспорта"))

    shop_labels = ["Напитки", "Закуски", "Автотовары", "Кофе", "Табак"]
    fig_shop = go.Figure()
    for col, label in zip(SHOP_COLS, shop_labels):
        fig_shop.add_trace(go.Bar(x=d_agg["timestamp"], y=d_agg[col], name=label))
    fig_shop.update_layout(barmode="stack", **_dark_layout("Выручка магазина (руб)"))

    return fig_traffic, fig_shop


@app.callback(Output("overview-heatmap", "figure"),
              Input("overview-date-range", "start_date"),
              Input("overview-date-range", "end_date"))
def update_heatmap(start, end):
    mask = (df_raw["timestamp"] >= start) & (df_raw["timestamp"] <= end)
    d = df_raw[mask].copy()
    pivot = d.groupby(["hour", "day_of_week"])["total_fuel_sales"].mean().unstack()
    days = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[days[i] for i in pivot.columns],
        y=[f"{h:02d}:00" for h in pivot.index],
        colorscale="Viridis", colorbar=dict(title="Литры/ч")
    ))
    fig.update_layout(**_dark_layout("Средние продажи: час × день недели"))
    return fig


# ── Callbacks: Station Analysis ────────────────────────────────────────────────
@app.callback(
    Output("station-sales", "figure"),
    Output("station-hourly", "figure"),
    Output("station-weekly", "figure"),
    Output("station-weather-impact", "figure"),
    Input("station-select", "value"),
    Input("fuel-checklist", "value"),
)
def update_station(station, fuel_cols):
    d = df_raw[df_raw["station_name"] == station].copy()
    d_daily = d.groupby(d["timestamp"].dt.date)[fuel_cols].sum().reset_index()

    fig_sales = go.Figure()
    for col, label, color in zip(FUEL_COLS, FUEL_LABELS, FUEL_COLORS):
        if col in fuel_cols:
            fig_sales.add_trace(go.Scatter(
                x=d_daily["timestamp"], y=d_daily[col],
                name=label, line=dict(color=color)
            ))
    fig_sales.update_layout(**_dark_layout(f"Ежедневные продажи — {station}"))

    # Hourly pattern
    hourly = d.groupby("hour")["total_fuel_sales"].mean().reset_index()
    fig_hourly = go.Figure(go.Bar(
        x=hourly["hour"], y=hourly["total_fuel_sales"],
        marker_color="#17a2b8"
    ))
    fig_hourly.update_layout(**_dark_layout("Суточный паттерн (ср. литры/час)"))

    # Weekly pattern
    dow_labels = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    weekly = d.groupby("day_of_week")["total_fuel_sales"].mean().reset_index()
    fig_weekly = go.Figure(go.Bar(
        x=[dow_labels[i] for i in weekly["day_of_week"]],
        y=weekly["total_fuel_sales"],
        marker_color="#ffc107"
    ))
    fig_weekly.update_layout(**_dark_layout("Недельный паттерн"))

    # Weather impact
    weather_agg = d.groupby("weather_condition")["total_fuel_sales"].mean().reset_index()
    fig_weather = go.Figure(go.Bar(
        x=weather_agg["weather_condition"],
        y=weather_agg["total_fuel_sales"],
        marker_color="#28a745"
    ))
    fig_weather.update_layout(**_dark_layout("Влияние погоды на продажи (ср. л/ч)"))

    return fig_sales, fig_hourly, fig_weekly, fig_weather


# ── Callbacks: Forecasts ───────────────────────────────────────────────────────
@app.callback(
    Output("forecast-chart", "figure"),
    Output("forecast-status", "children"),
    Output("forecast-status", "color"),
    Input("forecast-station", "value"),
    Input("forecast-target", "value"),
)
def update_forecast(station, target):
    forecast_file = OUTPUTS_DIR / "forecasts" / f"{target}.csv"
    if not forecast_file.exists():
        fig = go.Figure()
        fig.add_annotation(text="Модель ещё не обучена. Запустите: python run.py --mode train",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color="#aaa"))
        fig.update_layout(**_dark_layout("Прогноз TFT"))
        return fig, "Файл прогноза не найден. Запустите обучение.", "warning"

    df_fc = pd.read_csv(forecast_file)
    station_id = df_raw[df_raw["station_name"] == station]["station_id"].iloc[0]
    df_station = df_fc[df_fc["station_id"] == str(station_id)]

    d_actual = df_raw[df_raw["station_name"] == station].copy()
    d_actual_last = d_actual.tail(len(df_station)).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=d_actual_last[target].values if target in d_actual_last.columns else [],
        name="Факт", line=dict(color="#17a2b8", width=2)
    ))
    if len(df_station):
        fig.add_trace(go.Scatter(
            y=df_station["forecast_median"].values,
            name="Прогноз (медиана)", line=dict(color="#ffc107", width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            y=df_station["forecast_p90"].values,
            name="P90", line=dict(color="rgba(255,99,71,0.3)", width=0),
            fill=None
        ))
        fig.add_trace(go.Scatter(
            y=df_station["forecast_p10"].values,
            name="P10-P90 интервал", line=dict(color="rgba(255,99,71,0.3)", width=0),
            fill="tonexty", fillcolor="rgba(255,99,71,0.1)"
        ))

    target_label = target.replace("_", " ").upper()
    fig.update_layout(**_dark_layout(f"Прогноз TFT: {target_label} — {station}"))
    return fig, "Прогноз загружен успешно.", "success"


# ── Callbacks: Factor Analysis ─────────────────────────────────────────────────
@app.callback(
    Output("factor-importance", "figure"),
    Output("factor-promo", "figure"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
)
def update_factors(start, end):
    # Feature importance from model
    fi_file = OUTPUTS_DIR / "forecasts" / "feature_importance.json"
    if fi_file.exists():
        data = json.loads(fi_file.read_text())
        encoder = data.get("encoder", {})
        sorted_items = sorted(encoder.items(), key=lambda x: x[1], reverse=True)[:15]
        labels = [k for k, _ in sorted_items]
        values = [v for _, v in sorted_items]
        fig_imp = go.Figure(go.Bar(
            x=values, y=labels, orientation="h",
            marker_color="#17a2b8"
        ))
    else:
        # Fallback: correlation-based importance
        mask = (df_raw["timestamp"] >= start) & (df_raw["timestamp"] <= end)
        d = df_raw[mask]
        num_cols = ["temperature", "total_traffic", "precipitation_mm",
                    "competitor_price_AI92", "promotion_fuel_active", "ad_active"]
        corrs = {c: abs(d[c].corr(d["total_fuel_sales"])) for c in num_cols if c in d.columns}
        sorted_corrs = sorted(corrs.items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, _ in sorted_corrs]
        values = [v for _, v in sorted_corrs]
        fig_imp = go.Figure(go.Bar(
            x=values, y=labels, orientation="h", marker_color="#17a2b8"
        ))
    fig_imp.update_layout(**_dark_layout("Важность факторов (TFT или корреляция)"))

    # Promotion effect
    mask = (df_raw["timestamp"] >= start) & (df_raw["timestamp"] <= end)
    d = df_raw[mask]
    promo_agg = d.groupby("promotion_fuel_active")["total_fuel_sales"].mean().reset_index()
    promo_agg["label"] = promo_agg["promotion_fuel_active"].map({0.0: "Без акции", 1.0: "Акция"})
    fig_promo = go.Figure(go.Bar(
        x=promo_agg["label"], y=promo_agg["total_fuel_sales"],
        marker_color=["#6c757d", "#28a745"]
    ))
    fig_promo.update_layout(**_dark_layout("Эффект акции на топливо"))

    return fig_imp, fig_promo


@app.callback(Output("factor-competitor", "figure"),
              Input("station-select", "value"))
def update_competitor(station):
    d = df_raw[df_raw["station_name"] == station].copy()
    d_sample = d.sample(min(2000, len(d)), random_state=42)
    fig = px.scatter(
        d_sample, x="competitor_price_AI92", y="sales_AI92",
        color="season", opacity=0.5,
        color_discrete_sequence=px.colors.qualitative.Pastel,
        title=f"Цена конкурента vs Продажи АИ-92 — {station}",
    )
    fig.update_layout(**_dark_layout(""))
    return fig


# ── Callbacks: Recommendations ─────────────────────────────────────────────────
@app.callback(
    Output("recommendations-text", "children"),
    Output("rec-top-stations", "figure"),
    Output("rec-seasonal", "figure"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
)
def update_recommendations(start, end):
    recs = generate_recommendations()
    rec_items = [html.P(r, style={"color": "#ccc", "fontSize": "15px"}) for r in recs]

    # Top stations
    top = df_raw.groupby("station_name")["total_fuel_sales"].sum().nlargest(10).reset_index()
    fig_top = go.Figure(go.Bar(
        x=top["total_fuel_sales"], y=top["station_name"],
        orientation="h", marker_color="#17a2b8"
    ))
    fig_top.update_layout(**_dark_layout("Топ-10 АЗС по продажам"))

    # Seasonal
    seasonal = df_raw.groupby("season")[FUEL_COLS].sum().reset_index()
    season_order = {"winter": 0, "spring": 1, "summer": 2, "autumn": 3}
    season_labels = {"winter": "Зима", "spring": "Весна", "summer": "Лето", "autumn": "Осень"}
    seasonal["order"] = seasonal["season"].map(season_order)
    seasonal = seasonal.sort_values("order")
    seasonal["season"] = seasonal["season"].map(season_labels)

    fig_seasonal = go.Figure()
    for col, label, color in zip(FUEL_COLS, FUEL_LABELS, FUEL_COLORS):
        fig_seasonal.add_trace(go.Bar(
            x=seasonal["season"], y=seasonal[col], name=label, marker_color=color
        ))
    fig_seasonal.update_layout(barmode="stack", **_dark_layout("Продажи по сезонам"))

    return rec_items, fig_top, fig_seasonal


def run_dashboard():
    print(f"\nDashboard: http://localhost:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG)


if __name__ == "__main__":
    run_dashboard()
