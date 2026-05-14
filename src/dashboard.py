"""Интерактивный Dash-дашборд для TFT-анализа сети АЗС Татнефть."""
import base64
import io
import json
import warnings
from datetime import datetime
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
    DETAILED_DATA, FIVE_STATIONS_DATA, STATIONS_META, DATA_DIR, DATA_CACHE,
    OUTPUTS_DIR, MODELS_DIR, PROJECT_DIR, TARGETS,
    DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG, USE_5_STATIONS,
    FORECAST_CHART_HISTORY_HOURS, FORECAST_CHART_FUTURE_HOURS,
)
from src.predict import generate_recommendations, forecast_extended
from src.metrics import load_metrics, METRICS_DIR
from src import training_runner

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
            # Настройки графика хранятся в localStorage браузера.
            # Пусто = используются дефолты из config.FORECAST_CHART_*.
            dcc.Store(id="forecast-settings-store", storage_type="local"),

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
                dbc.Col([
                    html.Label(" ", style={"color": "#aaa"}),
                    dbc.Button(
                        [html.Span("⚙", style={"fontSize": "18px", "marginRight": "6px"}),
                         "Настройки"],
                        id="forecast-settings-btn",
                        color="secondary", outline=True,
                        className="w-100",
                    ),
                ], width=4, className="d-flex flex-column justify-content-end"),
            ], className="mb-3 mt-3"),

            dbc.Collapse(
                dbc.Card(style=CARD, children=[
                    html.H6("Часы для графика прогноза",
                            style={"color": "#17a2b8"}),
                    dbc.Row([
                        dbc.Col([
                            html.Label("История (← факт), часов",
                                       style={"color": "#aaa", "fontSize": "13px"}),
                            dbc.Input(
                                id="forecast-settings-history",
                                type="number", min=1, max=168, step=1,
                            ),
                        ], width=4),
                        dbc.Col([
                            html.Label("Прогноз (→), часов",
                                       style={"color": "#aaa", "fontSize": "13px"}),
                            dbc.Input(
                                id="forecast-settings-future",
                                type="number", min=1, max=720, step=1,
                            ),
                        ], width=4),
                        dbc.Col([
                            html.Label(" ", style={"color": "#aaa"}),
                            dbc.Button("Сохранить",
                                       id="forecast-settings-save",
                                       color="info", className="w-100"),
                        ], width=2, className="d-flex flex-column justify-content-end"),
                        dbc.Col([
                            html.Label(" ", style={"color": "#aaa"}),
                            dbc.Button("Сброс",
                                       id="forecast-settings-reset",
                                       color="secondary", outline=True,
                                       className="w-100"),
                        ], width=2, className="d-flex flex-column justify-content-end"),
                    ]),
                    html.Small(
                        f"При прогнозе > 24 ч включается итеративный rollout (медленнее, "
                        f"точность падает). Сохраняется в браузере; «Сброс» → дефолты "
                        f"из конфига ({FORECAST_CHART_HISTORY_HOURS} / "
                        f"{FORECAST_CHART_FUTURE_HOURS}).",
                        style={"color": "#888", "display": "block", "marginTop": "8px"},
                    ),
                ]),
                id="forecast-settings-collapse", is_open=False,
            ),

            dbc.Row([dbc.Col(dcc.Graph(id="forecast-chart"), width=12)]),
            dbc.Row([dbc.Col(
                dbc.Alert(
                    id="forecast-status",
                    color="info",
                    children="Запустите обучение (run.py --mode train) для генерации прогнозов TFT.",
                ), width=12
            )]),
        ]),

        # ── Вкладка 4: Прогноз — таблица ──────────────────────────────────────
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

        # ── Вкладка 5: Рекомендации + факторный анализ ─────────────────────────
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
            dbc.Row([
                dbc.Col(dcc.Graph(id="factor-importance"), width=7),
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("Эффект акций", style={"color": "#17a2b8"}),
                        dcc.Graph(id="factor-promo"),
                    ])
                ], width=5),
            ]),
            dbc.Row([dbc.Col(dcc.Graph(id="factor-corr-heatmap"), width=12)]),
        ]),

        # ── Вкладка 6: Метрики качества ────────────────────────────────────────
        dbc.Tab(label="Метрики", children=[
            dcc.Store(id="metrics-refresh-trigger"),
            dbc.Row([
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("Сравнение моделей: macro-avg по 9 таргетам",
                                style={"color": "#17a2b8"}),
                        html.Small(
                            "Метрики считаются на одной и той же 24-часовой "
                            "валидационной выборке для TFT (медиана) и baseline-моделей. "
                            "Macro-avg — среднее значение метрики по 9 таргетам. "
                            "Чем меньше — тем лучше.",
                            style={"color": "#888", "display": "block",
                                   "marginBottom": "12px"},
                        ),
                        dcc.Graph(id="metrics-macro-bar"),
                    ])
                ], width=12),
            ], className="mt-3"),
            dbc.Row([
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("MAE / RMSE / MAPE / SMAPE по каждому таргету",
                                style={"color": "#17a2b8"}),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Модель", style={"color": "#aaa"}),
                                dcc.Dropdown(
                                    id="metrics-model-select",
                                    clearable=False,
                                    style={"backgroundColor": "#333"},
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label("АЗС (опционально)",
                                           style={"color": "#aaa"}),
                                dcc.Dropdown(
                                    id="metrics-station-select",
                                    placeholder="Все АЗС (агрегат)",
                                    style={"backgroundColor": "#333"},
                                ),
                            ], width=4),
                            dbc.Col([
                                html.Label(" ", style={"color": "#aaa"}),
                                dbc.Button("Обновить",
                                           id="metrics-refresh-btn",
                                           color="info", className="w-100"),
                            ], width=4,
                                className="d-flex flex-column justify-content-end"),
                        ], className="mb-3"),
                        html.Div(id="metrics-table-output"),
                    ])
                ], width=12),
            ]),
            dbc.Row([
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("SMAPE по таргетам × моделям",
                                style={"color": "#17a2b8"}),
                        dcc.Graph(id="metrics-heatmap"),
                    ])
                ], width=12),
            ]),
        ]),

        # ── Вкладка 7: Настройки (полный pipeline из UI) ───────────────────────
        dbc.Tab(label="Настройки", children=[
            dcc.Interval(id="settings-poll", interval=2000, disabled=True),
            dcc.Store(id="settings-running", data=False),

            dbc.Row([
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("1. Загрузка исходных CSV",
                                style={"color": "#17a2b8"}),
                        html.Small(
                            "Принимаем два файла того же формата, что и "
                            "detailed_data.csv / stations_metadata.csv. "
                            "Файлы сохранятся в data/. Если данные уже лежат "
                            "в data/ — этот шаг можно пропустить.",
                            style={"color": "#888", "display": "block",
                                   "marginBottom": "12px"},
                        ),
                        dbc.Row([
                            dbc.Col([
                                html.Label("Почасовые данные (detailed_data.csv)",
                                           style={"color": "#aaa"}),
                                dcc.Upload(
                                    id="upload-detailed",
                                    children=html.Div([
                                        "Перетащите CSV сюда или ",
                                        html.A("выберите файл",
                                               style={"color": "#17a2b8"}),
                                    ]),
                                    style={
                                        "width": "100%", "minHeight": "60px",
                                        "lineHeight": "60px", "borderWidth": "1px",
                                        "borderStyle": "dashed", "borderRadius": "8px",
                                        "textAlign": "center", "color": "#ccc",
                                    },
                                    multiple=False,
                                ),
                                html.Div(id="upload-detailed-status",
                                         style={"color": "#28a745", "marginTop": "6px"}),
                            ], width=6),
                            dbc.Col([
                                html.Label("Метаданные АЗС (stations_metadata.csv)",
                                           style={"color": "#aaa"}),
                                dcc.Upload(
                                    id="upload-meta",
                                    children=html.Div([
                                        "Перетащите CSV сюда или ",
                                        html.A("выберите файл",
                                               style={"color": "#17a2b8"}),
                                    ]),
                                    style={
                                        "width": "100%", "minHeight": "60px",
                                        "lineHeight": "60px", "borderWidth": "1px",
                                        "borderStyle": "dashed", "borderRadius": "8px",
                                        "textAlign": "center", "color": "#ccc",
                                    },
                                    multiple=False,
                                ),
                                html.Div(id="upload-meta-status",
                                         style={"color": "#28a745", "marginTop": "6px"}),
                            ], width=6),
                        ]),
                    ])
                ], width=12),
            ], className="mt-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("2. Запуск задачи",
                                style={"color": "#17a2b8"}),
                        html.Small(
                            "Train: 5–60 мин (зависит от размера и GPU). "
                            "Predict: 1–3 мин. Evaluate: < 1 мин. "
                            "Запускается в фоне как отдельный процесс — "
                            "дашборд остаётся отзывчивым.",
                            style={"color": "#888", "display": "block",
                                   "marginBottom": "12px"},
                        ),
                        dbc.Row([
                            dbc.Col([
                                dbc.Checklist(
                                    id="settings-quick",
                                    options=[{
                                        "label": " Быстрый режим (5 АЗС, "
                                                 "только для train)",
                                        "value": "quick",
                                    }],
                                    value=[],
                                    style={"color": "#ccc"},
                                ),
                            ], width=12),
                        ], className="mb-2"),
                        dbc.Row([
                            dbc.Col(dbc.Button(
                                "Обучить модель",
                                id="btn-train",
                                color="info", className="w-100",
                            ), width=3),
                            dbc.Col(dbc.Button(
                                "Сгенерировать прогнозы",
                                id="btn-predict",
                                color="info", outline=True, className="w-100",
                            ), width=3),
                            dbc.Col(dbc.Button(
                                "Только метрики",
                                id="btn-evaluate",
                                color="secondary", outline=True, className="w-100",
                            ), width=3),
                            dbc.Col(dbc.Button(
                                "Остановить",
                                id="btn-stop",
                                color="danger", outline=True, className="w-100",
                            ), width=3),
                        ]),
                    ])
                ], width=12),
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("3. Статус и журнал",
                                style={"color": "#17a2b8"}),
                        html.Div(id="settings-status",
                                 children=dbc.Alert("Готов к запуску.",
                                                    color="secondary",
                                                    className="mb-2")),
                        html.Pre(
                            id="settings-log",
                            style={
                                "backgroundColor": "#0f0f18",
                                "color": "#d4d4d4",
                                "padding": "12px",
                                "borderRadius": "6px",
                                "maxHeight": "360px",
                                "overflow": "auto",
                                "fontSize": "12px",
                                "fontFamily": "Consolas, monospace",
                                "border": "1px solid #333",
                                "whiteSpace": "pre-wrap",
                            },
                            children="(журнал пуст)",
                        ),
                    ])
                ], width=12),
            ]),
        ]),

        # ── Вкладка 8: О проекте ───────────────────────────────────────────────
        dbc.Tab(label="О проекте", children=[
            dbc.Row([
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H4("ТАБД — Анализ сети АЗС «Татнефть»",
                                style={"color": "#17a2b8"}),
                        html.P(
                            "Учебный проект курса «Технологии анализа больших данных». "
                            "На синтетическом датасете 25 АЗС (8760 часов = 1 год) обучена "
                            "Temporal Fusion Transformer — современная attention-based модель "
                            "временных рядов, дающая квантильный прогноз на 24 часа вперёд "
                            "одновременно по 9 показателям. Качество модели измеряется "
                            "MAE / RMSE / MAPE / SMAPE по каждому таргету и каждой АЗС и "
                            "сравнивается с двумя naive baselines.",
                            style={"color": "#ccc", "fontSize": "15px"}),
                    ])
                ], width=12),
            ], className="mt-3"),

            dbc.Row([
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("Вкладки", style={"color": "#17a2b8"}),
                        html.Ul([
                            html.Li([html.B("Обзор сети — "),
                                "агрегированный взгляд на 25 АЗС: KPI, динамика "
                                "продаж по топливу, структура, трафик, выручка магазина, "
                                "тепловая карта час × день недели."],
                                style={"color": "#ccc", "marginBottom": "8px"}),
                            html.Li([html.B("Анализ АЗС — "),
                                "глубокая аналитика одной выбранной АЗС: ежедневные "
                                "продажи, суточный и недельный паттерн, влияние погоды."],
                                style={"color": "#ccc", "marginBottom": "8px"}),
                            html.Li([html.B("Прогнозы TFT — "),
                                "график прогноза для выбранной АЗС и показателя. "
                                "Слева от x=0 — факт (синяя), справа — медиана прогноза "
                                "(жёлтая) с интервалом P10–P90. Кнопка ⚙ «Настройки» — "
                                "задать своё количество часов истории и прогноза; "
                                "значения сохраняются в localStorage браузера."],
                                style={"color": "#ccc", "marginBottom": "8px"}),
                            html.Li([html.B("Прогноз — таблица — "),
                                "числовой прогноз на 24 / 48 / 168 / 720 часов. "
                                "Для горизонта > 24 ч модель работает итеративно, "
                                "точность падает с ростом горизонта."],
                                style={"color": "#ccc", "marginBottom": "8px"}),
                            html.Li([html.B("Рекомендации — "),
                                "автоматические инсайты (топ-5 факторов + что с ними делать), "
                                "топ-10 АЗС, важность факторов, эффект акций и "
                                "корреляционная матрица 9 таргетов × 12 факторов."],
                                style={"color": "#ccc", "marginBottom": "8px"}),
                            html.Li([html.B("Метрики — "),
                                "сравнение TFT с baseline-моделями (naive_yesterday, "
                                "seasonal_naive_week) на одной и той же 24-часовой "
                                "валидации: MAE / RMSE / MAPE / SMAPE по каждому из "
                                "9 таргетов и по каждой АЗС, macro-avg, heatmap SMAPE."],
                                style={"color": "#ccc", "marginBottom": "8px"}),
                            html.Li([html.B("Настройки — "),
                                "полный pipeline из UI: загрузка CSV (dcc.Upload), "
                                "запуск Train / Predict / Evaluate в фоновом subprocess, "
                                "статус и хвост лога в реальном времени. Дашборд "
                                "остаётся отзывчивым во время длинных задач."],
                                style={"color": "#ccc"}),
                        ]),
                    ])
                ], width=6),

                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("Квантили прогноза", style={"color": "#17a2b8"}),
                        html.P([
                            html.B("P10 "),
                            "— нижняя оценка. С вероятностью 90 % реальное значение "
                            "будет выше. Для консервативного планирования запасов.",
                        ], style={"color": "#ccc", "marginBottom": "6px"}),
                        html.P([
                            html.B("Медиана "),
                            "— наиболее вероятное значение, базовое планирование.",
                        ], style={"color": "#ccc", "marginBottom": "6px"}),
                        html.P([
                            html.B("P90 "),
                            "— верхняя оценка. С вероятностью 90 % реальное значение "
                            "будет ниже. Для оценки пиковой нагрузки и страхового запаса.",
                        ], style={"color": "#ccc", "marginBottom": "6px"}),
                        html.P([
                            html.B("Интервал P10–P90 "),
                            "— 80-% доверительный коридор. Узкий = модель уверена, "
                            "широкий = высокая неопределённость.",
                        ], style={"color": "#ccc"}),
                    ])
                ], width=6),
            ]),

            dbc.Row([
                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("9 целевых показателей",
                                style={"color": "#17a2b8"}),
                        html.P("Модель прогнозирует одновременно (л/ч для топлива, руб/ч для магазина):",
                               style={"color": "#aaa", "fontSize": "13px",
                                      "marginBottom": "6px"}),
                        html.Ul([
                            html.Li("total_fuel_sales — суммарные продажи топлива",
                                    style={"color": "#ccc"}),
                            html.Li("sales_AI92 / AI95 / AI98 — бензины 92/95/98",
                                    style={"color": "#ccc"}),
                            html.Li("sales_DT_EURO / TANEKO / SUMMER / WINTER — дизель 4 видов",
                                    style={"color": "#ccc"}),
                            html.Li("shop_total_revenue — выручка магазина при АЗС",
                                    style={"color": "#ccc"}),
                        ], style={"marginBottom": 0}),
                    ])
                ], width=6),

                dbc.Col([
                    dbc.Card(style=CARD, children=[
                        html.H5("Стек и документация",
                                style={"color": "#17a2b8"}),
                        html.P([
                            html.B("Модель: "),
                            "TFT (pytorch-forecasting 1.7), 879k параметров, "
                            "encoder 168 ч, prediction 24 ч, MultiLoss(QuantileLoss × 9).",
                        ], style={"color": "#ccc", "marginBottom": "6px"}),
                        html.P([
                            html.B("Стек: "),
                            "PyTorch 2.5.1 + CUDA 12.4, Lightning 2.6, "
                            "Dash 4 + Bootstrap DARKLY, Plotly. "
                            "Baselines: numpy lag (24 ч / 168 ч).",
                        ], style={"color": "#ccc", "marginBottom": "6px"}),
                        html.P([
                            html.B("Данные: "),
                            "полный 25-АЗС-датасет лежит вне репозитория; "
                            "в git закоммичен маленький обезличенный sample "
                            "(", html.Code("data/sample/"), ", 2 АЗС × 14 дней) — "
                            "проект запускается сразу после клонирования. "
                            "Полную синтетику за 2024 год можно сгенерировать "
                            "скриптом ", html.Code("scripts/make_synthetic.py"), ".",
                        ], style={"color": "#ccc", "marginBottom": "6px"}),
                        html.P([
                            html.B("Документация: "),
                            html.Code("README.md", style={"color": "#17a2b8"}),
                            " (установка/запуск), ",
                            html.Code("DOCS.md", style={"color": "#17a2b8"}),
                            " (подробно про показатели, прогнозы и вкладки).",
                        ], style={"color": "#ccc", "marginBottom": "6px"}),
                        html.P([
                            html.B("Репозиторий: "),
                            html.A("github.com/Terps489/TABD",
                                   href="https://github.com/Terps489/TABD",
                                   target="_blank",
                                   style={"color": "#17a2b8"}),
                        ], style={"color": "#ccc"}),
                    ])
                ], width=6),
            ]),
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
def _effective_forecast_hours(settings: dict | None) -> tuple[int, int]:
    """Достаём (history, future) из store или подставляем дефолты конфига."""
    s = settings or {}
    return (
        int(s.get("history") or FORECAST_CHART_HISTORY_HOURS),
        int(s.get("future") or FORECAST_CHART_FUTURE_HOURS),
    )


@app.callback(
    Output("forecast-settings-collapse", "is_open"),
    Input("forecast-settings-btn", "n_clicks"),
    State("forecast-settings-collapse", "is_open"),
    prevent_initial_call=True,
)
def toggle_forecast_settings(n, is_open):
    return not is_open


@app.callback(
    Output("forecast-settings-history", "value"),
    Output("forecast-settings-future", "value"),
    Input("forecast-settings-collapse", "is_open"),
    Input("forecast-settings-reset", "n_clicks"),
    State("forecast-settings-store", "data"),
    prevent_initial_call=True,
)
def populate_forecast_settings(is_open, reset_clicks, stored):
    ctx = dash.callback_context.triggered_id
    if ctx == "forecast-settings-reset":
        return FORECAST_CHART_HISTORY_HOURS, FORECAST_CHART_FUTURE_HOURS
    if not is_open:
        raise dash.exceptions.PreventUpdate
    h, f = _effective_forecast_hours(stored)
    return h, f


@app.callback(
    Output("forecast-settings-store", "data"),
    Input("forecast-settings-save", "n_clicks"),
    Input("forecast-settings-reset", "n_clicks"),
    State("forecast-settings-history", "value"),
    State("forecast-settings-future", "value"),
    prevent_initial_call=True,
)
def save_forecast_settings(save_n, reset_n, hist, future):
    if dash.callback_context.triggered_id == "forecast-settings-reset":
        return None
    if not save_n:
        raise dash.exceptions.PreventUpdate
    return {
        "history": int(hist) if hist else FORECAST_CHART_HISTORY_HOURS,
        "future": int(future) if future else FORECAST_CHART_FUTURE_HOURS,
    }


@app.callback(
    Output("forecast-chart", "figure"),
    Output("forecast-status", "children"),
    Output("forecast-status", "color"),
    Input("forecast-station", "value"),
    Input("forecast-target", "value"),
    Input("forecast-settings-store", "data"),
)
def update_forecast(station, target, settings):
    history_hours, future_hours = _effective_forecast_hours(settings)
    try:
        df_fc = forecast_extended(target, future_hours)
    except FileNotFoundError as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Прогноз не найден: {e}",
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                           font=dict(size=14, color="#aaa"))
        fig.update_layout(**_dark_layout("Прогноз TFT"))
        return fig, "Файл прогноза не найден. Запустите обучение.", "warning"

    station_id = str(df_raw[df_raw["station_name"] == station]["station_id"].iloc[0])
    df_station = df_fc[df_fc["station_id"] == station_id].sort_values("hour_ahead").reset_index(drop=True)
    pred_len = len(df_station)

    d_actual = df_raw[df_raw["station_name"] == station].sort_values("timestamp")
    d_context = d_actual.tail(history_hours).reset_index(drop=True)

    x_actual = list(range(-len(d_context) + 1, 1))
    x_forecast = df_station["hour_ahead"].tolist()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x_actual,
        y=d_context[target].values if target in d_context.columns else [],
        name="Факт (история)", mode="lines",
        line=dict(color="#17a2b8", width=2)
    ))
    if pred_len:
        fig.add_trace(go.Scatter(
            x=x_forecast, y=df_station["median"].values,
            name="Прогноз (медиана)", mode="lines+markers",
            line=dict(color="#ffc107", width=2, dash="dash")
        ))
        fig.add_trace(go.Scatter(
            x=x_forecast, y=df_station["p90"].values,
            name="P90", mode="lines",
            line=dict(color="rgba(255,99,71,0.3)", width=0)
        ))
        fig.add_trace(go.Scatter(
            x=x_forecast, y=df_station["p10"].values,
            name="P10-P90 интервал", mode="lines",
            line=dict(color="rgba(255,99,71,0.3)", width=0),
            fill="tonexty", fillcolor="rgba(255,99,71,0.15)"
        ))
        fig.add_vline(x=0, line=dict(color="#888", width=1, dash="dot"))

    target_label = target.replace("_", " ").upper()
    rollout_note = " (rollout)" if future_hours > 24 else ""
    layout = _dark_layout(
        f"Прогноз TFT: {target_label} — {station}  "
        f"(история {history_hours} ч, прогноз {future_hours} ч{rollout_note})"
    )
    layout["xaxis"] = dict(gridcolor="#333", title="Часы (отрицательные = история, положительные = прогноз)")
    layout["yaxis"] = dict(gridcolor="#333", title="Литры/час" if "fuel" in target or "sales" in target else "Руб/час")
    fig.update_layout(**layout)
    return fig, f"Прогноз загружен: история {history_hours}ч, прогноз {future_hours}ч.", "success"


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


_HEATMAP_FACTORS = [
    "total_traffic", "temperature", "precipitation_mm",
    "is_weekend", "is_holiday", "is_rush_hour", "is_night",
    "promotion_fuel_active", "promotion_shop_active", "ad_active",
    "competitor_price_AI92", "competitor_price_AI95",
]
_HEATMAP_LABELS = [
    "Трафик", "Температура", "Осадки",
    "Выходной", "Праздник", "Час пик", "Ночь",
    "Промо топливо", "Промо магазин", "Реклама",
    "Цена конк. AI92", "Цена конк. AI95",
]


@app.callback(Output("factor-corr-heatmap", "figure"),
              Input("overview-date-range", "start_date"),
              Input("overview-date-range", "end_date"))
def update_factor_heatmap(start, end):
    mask = (df_raw["timestamp"] >= start) & (df_raw["timestamp"] <= end)
    d = df_raw[mask]
    available = [f for f in _HEATMAP_FACTORS if f in d.columns]
    labels = [_HEATMAP_LABELS[_HEATMAP_FACTORS.index(f)] for f in available]
    matrix = np.array([
        [d[f].corr(d[t]) for f in available] for t in TARGETS
    ])
    target_labels = [t.replace("_", " ").replace("sales ", "").upper() for t in TARGETS]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=target_labels,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        colorbar=dict(title="Корреляция"),
        hovertemplate="%{y} ↔ %{x}<br>r = %{z:.2f}<extra></extra>",
    ))
    fig.update_layout(
        **_dark_layout("Корреляция факторов с целевыми показателями")
    )
    fig.update_xaxes(tickangle=-30)
    return fig


# ── Callbacks: Рекомендации ────────────────────────────────────────────────────
@app.callback(
    Output("recommendations-text", "children"),
    Output("rec-top-stations", "figure"),
    Input("overview-date-range", "start_date"),
    Input("overview-date-range", "end_date"),
)
def update_recommendations(start, end):
    recs = generate_recommendations()
    rec_items = [html.P(r, style={"color": "#ccc", "fontSize": "15px"}) for r in recs]

    top = df_raw.groupby("station_name")["total_fuel_sales"].sum().nlargest(10).reset_index()
    fig_top = go.Figure(go.Bar(
        x=top["total_fuel_sales"], y=top["station_name"],
        orientation="h", marker_color="#17a2b8"
    ))
    fig_top.update_layout(**_dark_layout("Топ-10 АЗС по продажам"))

    return rec_items, fig_top


# ── Callbacks: Метрики ─────────────────────────────────────────────────────────
def _metric_color(model_name: str) -> str:
    """Стабильные цвета моделей: TFT выделяем, baselines — серые/коричневые."""
    return {
        "TFT": "#17a2b8",
        "naive_yesterday": "#aaaaaa",
        "seasonal_naive_week": "#8c7a6b",
    }.get(model_name, "#888888")


@app.callback(
    Output("metrics-model-select", "options"),
    Output("metrics-model-select", "value"),
    Output("metrics-station-select", "options"),
    Input("metrics-refresh-btn", "n_clicks"),
    Input("metrics-refresh-trigger", "data"),
)
def init_metrics_filters(_n, _trig):
    df_m = load_metrics()
    if df_m.empty:
        return [], None, []
    models = sorted(df_m["model"].unique().tolist())
    stations = sorted(s for s in df_m["station_id"].unique() if s != "ALL")
    return (
        [{"label": m, "value": m} for m in models],
        "TFT" if "TFT" in models else models[0],
        [{"label": s, "value": s} for s in stations],
    )


@app.callback(
    Output("metrics-macro-bar", "figure"),
    Input("metrics-refresh-btn", "n_clicks"),
    Input("metrics-refresh-trigger", "data"),
)
def update_metrics_macro(_n, _trig):
    df_m = load_metrics()
    if df_m.empty:
        fig = go.Figure()
        fig.add_annotation(text="Метрики не найдены. Запустите 'evaluate' "
                                "(во вкладке Настройки) после обучения.",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(size=14, color="#aaa"))
        fig.update_layout(**_dark_layout("Сравнение моделей"))
        return fig

    agg = df_m[df_m["station_id"] == "ALL"].groupby("model")[
        ["mae", "rmse", "mape", "smape"]
    ].mean().reset_index()
    metrics_to_plot = ["mae", "rmse", "mape", "smape"]
    fig = make_subplots(
        rows=1, cols=len(metrics_to_plot),
        subplot_titles=[m.upper() for m in metrics_to_plot],
        horizontal_spacing=0.06,
    )
    for j, metric in enumerate(metrics_to_plot, start=1):
        for _, row in agg.sort_values(metric).iterrows():
            fig.add_trace(go.Bar(
                x=[row["model"]], y=[row[metric]],
                name=row["model"], showlegend=(j == 1),
                marker_color=_metric_color(row["model"]),
                text=f"{row[metric]:.2f}", textposition="outside",
            ), row=1, col=j)
    layout = _dark_layout("Сравнение моделей: macro-avg по 9 таргетам "
                           "(меньше — лучше)")
    layout["showlegend"] = True
    layout["height"] = 360
    fig.update_layout(**layout)
    for j in range(1, len(metrics_to_plot) + 1):
        fig.update_xaxes(showticklabels=False, row=1, col=j)
        fig.update_yaxes(gridcolor="#333", row=1, col=j)
    return fig


@app.callback(
    Output("metrics-table-output", "children"),
    Input("metrics-model-select", "value"),
    Input("metrics-station-select", "value"),
    Input("metrics-refresh-btn", "n_clicks"),
    Input("metrics-refresh-trigger", "data"),
)
def update_metrics_table(model, station, _n, _trig):
    df_m = load_metrics()
    if df_m.empty or not model:
        return dbc.Alert("Метрики не найдены.", color="warning")

    rows = df_m[df_m["model"] == model]
    rows = rows[rows["station_id"] == (station if station else "ALL")]
    if rows.empty:
        return dbc.Alert(
            f"Нет данных для {model} / {station or 'ALL'}.", color="warning"
        )

    out = rows[["target", "n", "mae", "rmse", "mape", "smape"]].copy()
    out["target"] = out["target"].str.replace("_", " ").str.upper()
    for c in ("mae", "rmse"):
        out[c] = out[c].round(2)
    for c in ("mape", "smape"):
        out[c] = out[c].round(2).astype(str) + " %"

    return dash_table.DataTable(
        data=out.to_dict("records"),
        columns=[{"name": c.upper(), "id": c} for c in out.columns],
        page_size=10, sort_action="native",
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
        ],
    )


@app.callback(
    Output("metrics-heatmap", "figure"),
    Input("metrics-refresh-btn", "n_clicks"),
    Input("metrics-refresh-trigger", "data"),
)
def update_metrics_heatmap(_n, _trig):
    df_m = load_metrics()
    if df_m.empty:
        fig = go.Figure()
        fig.update_layout(**_dark_layout("SMAPE: модель × таргет"))
        return fig
    pivot = df_m[df_m["station_id"] == "ALL"].pivot_table(
        index="target", columns="model", values="smape",
    )
    target_labels = [t.replace("_", " ").upper() for t in pivot.index]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=target_labels,
        colorscale="RdYlGn_r", colorbar=dict(title="SMAPE %"),
        hovertemplate="%{y}<br>%{x}: %{z:.2f}%<extra></extra>",
        text=np.round(pivot.values, 1),
        texttemplate="%{text}", textfont={"size": 11},
    ))
    fig.update_layout(**_dark_layout("SMAPE по таргетам × моделям "
                                      "(зелёное лучше)"))
    return fig


# ── Callbacks: Настройки ───────────────────────────────────────────────────────
def _save_uploaded_csv(contents: str, default_name: str) -> tuple[bool, str]:
    """Декодирует base64-payload из dcc.Upload, сохраняет в data/."""
    if not contents:
        return False, "пусто"
    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)
        # Лёгкая валидация — должно быть похоже на CSV (читаемо как DataFrame).
        pd.read_csv(io.BytesIO(raw), nrows=5)
    except Exception as e:
        return False, f"не CSV: {e}"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = DATA_DIR / default_name
    out.write_bytes(raw)
    return True, f"сохранено: {out.name} ({len(raw)/1024:.1f} КБ)"


@app.callback(
    Output("upload-detailed-status", "children"),
    Input("upload-detailed", "contents"),
    State("upload-detailed", "filename"),
    prevent_initial_call=True,
)
def handle_upload_detailed(contents, filename):
    ok, msg = _save_uploaded_csv(contents, "detailed_data.csv")
    return ("✓ " if ok else "✗ ") + msg


@app.callback(
    Output("upload-meta-status", "children"),
    Input("upload-meta", "contents"),
    State("upload-meta", "filename"),
    prevent_initial_call=True,
)
def handle_upload_meta(contents, filename):
    ok, msg = _save_uploaded_csv(contents, "stations_metadata.csv")
    return ("✓ " if ok else "✗ ") + msg


@app.callback(
    Output("settings-running", "data", allow_duplicate=True),
    Output("settings-status", "children", allow_duplicate=True),
    Input("btn-train", "n_clicks"),
    Input("btn-predict", "n_clicks"),
    Input("btn-evaluate", "n_clicks"),
    Input("btn-stop", "n_clicks"),
    State("settings-quick", "value"),
    prevent_initial_call=True,
)
def handle_run_button(n_t, n_p, n_e, n_s, quick_val):
    trig = dash.callback_context.triggered_id
    if trig is None:
        raise dash.exceptions.PreventUpdate

    quick = "quick" in (quick_val or [])

    try:
        if trig == "btn-stop":
            stopped = training_runner.stop()
            msg = "Остановлено пользователем." if stopped else "Активных задач нет."
            return False, dbc.Alert(msg, color="warning", className="mb-2")

        mode_map = {"btn-train": "train", "btn-predict": "predict",
                    "btn-evaluate": "evaluate"}
        mode = mode_map.get(trig)
        if mode is None:
            raise dash.exceptions.PreventUpdate

        # Если предыдущий процесс закончился — очищаем state, чтобы не блокировать.
        if not training_runner.is_running():
            training_runner.clear()

        state = training_runner.start(mode, quick=quick)
        started = datetime.fromtimestamp(state.started_at).strftime("%H:%M:%S")
        return True, dbc.Alert(
            f"Запущено: {mode}{' --quick' if state.quick else ''} "
            f"(PID {state.pid}, старт {started})",
            color="info", className="mb-2",
        )
    except RuntimeError as e:
        return True, dbc.Alert(str(e), color="warning", className="mb-2")
    except Exception as e:
        return False, dbc.Alert(f"Ошибка запуска: {e}",
                                  color="danger", className="mb-2")


@app.callback(
    Output("settings-poll", "disabled"),
    Input("settings-running", "data"),
)
def toggle_poll(running):
    return not bool(running)


@app.callback(
    Output("settings-log", "children"),
    Output("settings-status", "children", allow_duplicate=True),
    Output("settings-running", "data", allow_duplicate=True),
    Output("metrics-refresh-trigger", "data"),
    Input("settings-poll", "n_intervals"),
    prevent_initial_call=True,
)
def poll_status(_n):
    status = training_runner.get_status()
    state = status.get("state") or {}
    log = training_runner.tail_log(200) or "(журнал пуст)"

    if status["running"]:
        started = state.get("started_at")
        started_s = (datetime.fromtimestamp(started).strftime("%H:%M:%S")
                     if started else "—")
        alert = dbc.Alert(
            f"Выполняется: {state.get('mode')} "
            f"{'--quick' if state.get('quick') else ''} "
            f"(PID {state.get('pid')}, старт {started_s})",
            color="info", className="mb-2",
        )
        # Обновлять метрики пока не нужно — задача не завершилась.
        return log, alert, True, dash.no_update

    finished = state.get("finished_at")
    if finished is None:
        return log, dbc.Alert("Готов к запуску.", color="secondary",
                              className="mb-2"), False, dash.no_update

    exit_code = state.get("exit_code")
    finished_s = datetime.fromtimestamp(finished).strftime("%H:%M:%S")
    if exit_code in (0, None):
        alert = dbc.Alert(
            f"Готово: {state.get('mode')} завершено в {finished_s}.",
            color="success", className="mb-2",
        )
        return log, alert, False, datetime.now().isoformat()

    color = "warning" if exit_code == -2 else "danger"
    alert = dbc.Alert(
        f"Завершено с кодом {exit_code} (mode={state.get('mode')}, "
        f"{finished_s}). Подробности в журнале.",
        color=color, className="mb-2",
    )
    return log, alert, False, datetime.now().isoformat()


def run_dashboard():
    print(f"\nДашборд: http://localhost:{DASHBOARD_PORT}")
    app.run(host=DASHBOARD_HOST, port=DASHBOARD_PORT, debug=DASHBOARD_DEBUG)


if __name__ == "__main__":
    run_dashboard()
