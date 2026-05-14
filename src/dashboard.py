"""Интерактивный Dash-дашборд для TFT-анализа сети АЗС Татнефть."""
import json
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

from src.config import (
    DETAILED_DATA, FIVE_STATIONS_DATA, DATA_CACHE,
    OUTPUTS_DIR, MODELS_DIR, PROJECT_DIR, TARGETS,
    DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG, USE_5_STATIONS
)
from src.predict import generate_recommendations, forecast_extended

warnings.filterwarnings("ignore")

# ── Загрузка данных ────────────────────────────────────────────────────────────
# Приоритет: parquet-кэш (создаётся при обучении) → CSV (если кэша нет).
if DATA_CACHE.exists():
    df_raw = pd.read_parquet(DATA_CACHE)
    if "timestamp" in df_raw.columns:
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
else:
    _csv_path = FIVE_STATIONS_DATA if USE_5_STATIONS else DETAILED_DATA
    if not _csv_path.exists():
        raise FileNotFoundError(
            f"Нет ни кэша ({DATA_CACHE.name}), ни исходного CSV ({_csv_path}).\n"
            "Запустите обучение или положите CSV-файлы в папку data/ "
            "(или укажите путь через переменную окружения TABD_DATA_DIR)."
        )
    df_raw = pd.read_csv(_csv_path, parse_dates=["timestamp"])

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


# ── Layout приложения ──────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    assets_folder=str(PROJECT_DIR / "assets"),
    title="Татнефть АЗС — TFT Дашборд",
)

CARD = {"borderRadius": "8px", "padding": "16px", "marginBottom": "16px"}

app.layout = dbc.Container(fluid=True, children=[
    # Шапка
    dbc.Row([
        dbc.Col(html.H2("Татнефть АЗС — TFT Аналитика", style={"color": "#17a2b8"}), width=8),
        dbc.Col(html.P("Анализ продаж топлива | Temporal Fusion Transformer",
                       style={"color": "#888", "marginTop": "10px", "textAlign": "right"}), width=4),
    ], className="mb-3 mt-3"),

    dbc.Tabs([

        # ── Вкладка 1: Обзор сети ──────────────────────────────────────────────
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

        # ── Вкладка 2: Анализ АЗС ──────────────────────────────────────────────
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

        # ── Вкладка 3: Прогнозы TFT ────────────────────────────────────────────
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

        # ── Вкладка 4: Факторный анализ ────────────────────────────────────────
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

        # ── Вкладка 5: Прогноз — таблица ──────────────────────────────────────
        dbc.Tab(label="Прогноз — таблица", children=[
            dbc.Row([
                dbc.Col([
                    html.Label("АЗС", style={"color": "#aaa"}),
                    dcc.Dropdown(
                        id="ftable-station",
                        options=[{"label": s, "value": s} for s in STATIONS],
                        value=STATIONS[0], clearable=False,
                        style={"backgroundColor": "#333"},
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Показатель", style={"color": "#aaa"}),
                    dcc.Dropdown(
                        id="ftable-target",
                        options=[{"label": t.replace("_", " ").upper(), "value": t}
                                 for t in TARGETS],
                        value="total_fuel_sales", clearable=False,
                        style={"backgroundColor": "#333"},
                    ),
                ], width=3),
                dbc.Col([
                    html.Label("Горизонт", style={"color": "#aaa"}),
                    dcc.Dropdown(
                        id="ftable-horizon",
                        options=[
                            {"label": "24 часа (1 день)", "value": 24},
                            {"label": "48 часов (2 дня)", "value": 48},
                            {"label": "168 часов (неделя)", "value": 168},
                            {"label": "720 часов (месяц)", "value": 720},
                        ],
                        value=24, clearable=False,
                        style={"backgroundColor": "#333"},
                    ),
                ], width=3),
                dbc.Col([
                    html.Label(" ", style={"color": "#aaa"}),
                    dbc.Button("Рассчитать", id="ftable-run",
                               color="info", className="w-100"),
                ], width=3),
            ], className="mb-3 mt-3"),
            dbc.Row([
                dbc.Col(
                    dbc.Alert(
                        "Для горизонта > 24 ч модель работает итеративно "
                        "(rollout по 24 ч), точность падает с ростом горизонта — "
                        "это отражено расширением интервала P10–P90.",
                        color="secondary", className="small",
                    ), width=12
                )
            ]),
            dbc.Row([
                dbc.Col(
                    dcc.Loading(
                        id="ftable-loading", type="default",
                        children=html.Div(id="ftable-output"),
                    ), width=12
                )
            ]),
        ]),

        # ── Вкладка 6: Рекомендации ────────────────────────────────────────────
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


# ── Callbacks: Обзор сети ──────────────────────────────────────────────────────
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

    # График динамики
    fig_trend = go.Figure()
    for col, label, color in zip(FUEL_COLS, FUEL_LABELS, FUEL_COLORS):
        fig_trend.add_trace(go.Scatter(
            x=d_agg["timestamp"], y=d_agg[col],
            name=label, line=dict(color=color), mode="lines"
        ))
    fig_trend.update_layout(**_dark_layout("Динамика продаж топлива (литры)"))

    # Круговая диаграмма
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


# ── Callbacks: Анализ АЗС ──────────────────────────────────────────────────────
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

    # Суточный паттерн
    hourly = d.groupby("hour")["total_fuel_sales"].mean().reset_index()
    fig_hourly = go.Figure(go.Bar(
        x=hourly["hour"], y=hourly["total_fuel_sales"],
        marker_color="#17a2b8"
    ))
    fig_hourly.update_layout(**_dark_layout("Суточный паттерн (ср. литры/час)"))

    # Недельный паттерн
    dow_labels = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    weekly = d.groupby("day_of_week")["total_fuel_sales"].mean().reset_index()
    fig_weekly = go.Figure(go.Bar(
        x=[dow_labels[i] for i in weekly["day_of_week"]],
        y=weekly["total_fuel_sales"],
        marker_color="#ffc107"
    ))
    fig_weekly.update_layout(**_dark_layout("Недельный паттерн"))

    # Влияние погоды
    weather_agg = d.groupby("weather_condition")["total_fuel_sales"].mean().reset_index()
    fig_weather = go.Figure(go.Bar(
        x=weather_agg["weather_condition"],
        y=weather_agg["total_fuel_sales"],
        marker_color="#28a745"
    ))
    fig_weather.update_layout(**_dark_layout("Влияние погоды на продажи (ср. л/ч)"))

    return fig_sales, fig_hourly, fig_weekly, fig_weather


# ── Callbacks: Прогнозы ────────────────────────────────────────────────────────
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

    df_fc = pd.read_csv(forecast_file, dtype={"station_id": str})
    station_id = str(df_raw[df_raw["station_name"] == station]["station_id"].iloc[0])
    df_station = df_fc[df_fc["station_id"] == station_id].sort_values("step").reset_index(drop=True)

    d_actual = df_raw[df_raw["station_name"] == station].copy().sort_values("timestamp")
    pred_len = len(df_station)
    # Контекст: 48 часов факта перед прогнозом, потом сам прогноз
    context_len = 48
    d_context = d_actual.tail(context_len).reset_index(drop=True)

    x_actual = list(range(-len(d_context) + 1, 1))  # отрицательные часы (история)
    x_forecast = list(range(1, pred_len + 1))        # положительные часы (прогноз)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_actual,
        y=d_context[target].values if target in d_context.columns else [],
        name="Факт (история)", mode="lines",
        line=dict(color="#17a2b8", width=2)
    ))
    if pred_len:
        fig.add_trace(go.Scatter(
            x=x_forecast, y=df_station["forecast_median"].values,
            name="Прогноз (медиана)", mode="lines+markers",
            line=dict(color="#ffc107", width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=x_forecast, y=df_station["forecast_p90"].values,
            name="P90", mode="lines",
            line=dict(color="rgba(255,99,71,0.3)", width=0)
        ))
        fig.add_trace(go.Scatter(
            x=x_forecast, y=df_station["forecast_p10"].values,
            name="P10-P90 интервал", mode="lines",
            line=dict(color="rgba(255,99,71,0.3)", width=0),
            fill="tonexty", fillcolor="rgba(255,99,71,0.15)"
        ))
        # Вертикальная линия "сейчас"
        fig.add_vline(x=0, line=dict(color="#888", width=1, dash="dot"))

    target_label = target.replace("_", " ").upper()
    layout = _dark_layout(f"Прогноз TFT: {target_label} — {station}  (час 0 = текущий момент)")
    layout["xaxis"] = dict(gridcolor="#333", title="Часы (отрицательные = история, положительные = прогноз)")
    layout["yaxis"] = dict(gridcolor="#333", title="Литры/час" if "fuel" in target or "sales" in target else "Руб/час")
    fig.update_layout(**layout)
    return fig, "Прогноз загружен успешно.", "success"


# ── Callbacks: Прогноз — таблица ───────────────────────────────────────────────
@app.callback(
    Output("ftable-output", "children"),
    Input("ftable-run", "n_clicks"),
    State("ftable-station", "value"),
    State("ftable-target", "value"),
    State("ftable-horizon", "value"),
    prevent_initial_call=True,
)
def update_forecast_table(n_clicks, station, target, horizon):
    if not n_clicks:
        return ""
    try:
        df_fc = forecast_extended(target, int(horizon))
    except Exception as e:
        return dbc.Alert(f"Ошибка прогноза: {e}", color="danger")

    station_id = str(df_raw[df_raw["station_name"] == station]["station_id"].iloc[0])
    df_s = df_fc[df_fc["station_id"] == station_id].copy().sort_values("hour_ahead")
    if df_s.empty:
        return dbc.Alert(f"Нет прогноза для станции {station}.", color="warning")

    unit = "л/ч" if ("fuel" in target or "sales" in target) and "shop" not in target else "руб/ч"

    df_s["Время"] = pd.to_datetime(df_s["timestamp"]).dt.strftime("%Y-%m-%d %H:%M")
    df_s["P10"] = df_s["p10"].round(2)
    df_s["Медиана"] = df_s["median"].round(2)
    df_s["P90"] = df_s["p90"].round(2)
    df_s = df_s.rename(columns={"hour_ahead": "Ч вперёд"})
    df_table = df_s[["Ч вперёд", "Время", "P10", "Медиана", "P90"]]

    target_label = target.replace("_", " ").upper()
    header_text = f"{target_label}  ({unit})  —  {station}  —  {len(df_table)} ч"

    return html.Div([
        html.H5(header_text, style={"color": "#17a2b8", "marginBottom": "12px"}),
        dash_table.DataTable(
            data=df_table.to_dict("records"),
            columns=[{"name": c, "id": c} for c in df_table.columns],
            page_size=24,
            sort_action="native",
            style_table={"overflowX": "auto"},
            style_header={
                "backgroundColor": "#1e1e2e", "color": "#17a2b8",
                "fontWeight": "bold", "border": "1px solid #333",
            },
            style_cell={
                "backgroundColor": "#1a1a2e", "color": "#ccc",
                "textAlign": "center", "padding": "8px",
                "border": "1px solid #333", "fontFamily": "monospace",
            },
            style_data_conditional=[
                {"if": {"row_index": "odd"}, "backgroundColor": "#22223a"},
                {"if": {"column_id": "Медиана"}, "color": "#ffc107", "fontWeight": "bold"},
            ],
        ),
    ])


# ── Callbacks: Факторный анализ ────────────────────────────────────────────────
@app.callback(
    Output("factor-importance", "figure"),
    Output("factor-promo", "figure"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
)
def update_factors(start, end):
    # Важность признаков из модели
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
        # Fallback: важность через корреляцию
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

    # Эффект акций
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


# ── Callbacks: Рекомендации ────────────────────────────────────────────────────
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

    # Топ АЗС
    top = df_raw.groupby("station_name")["total_fuel_sales"].sum().nlargest(10).reset_index()
    fig_top = go.Figure(go.Bar(
        x=top["total_fuel_sales"], y=top["station_name"],
        orientation="h", marker_color="#17a2b8"
    ))
    fig_top.update_layout(**_dark_layout("Топ-10 АЗС по продажам"))

    # Сезонность
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
    print(f"\nДашборд: http://localhost:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG)


if __name__ == "__main__":
    run_dashboard()
