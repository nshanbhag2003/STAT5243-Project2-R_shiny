"""
STAT5243 Project 2: Interactive Data Analysis Web Application
- UI/UX with polished layout and styling
- Cached reactive calculations
- Button-triggered EDA plotting to avoid unnecessary redraws
- Sampled visualization data for large datasets
- Safe preprocessing and feature engineering
- Dynamic feedback and robustness
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# PATHS / BUILT-IN DATASETS
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "数据")   

BUILT_IN_DATASETS = {
    "financial_news": {
        "name": "Financial News Events",
        "description": "Financial event records for mixed categorical and numeric exploration.",
        "files": {
            "csv": os.path.join(DATA_DIR, "financial_news_events.csv"),
            "json": os.path.join(DATA_DIR, "financial_news_events.json"),
        }
    },
    "grocery": {
        "name": "Grocery Chain Data",
        "description": "Retail / grocery dataset with useful cleaning and encoding opportunities.",
        "files": {
            "csv": os.path.join(DATA_DIR, "grocery_chain_data.csv"),
            "json": os.path.join(DATA_DIR, "grocery_chain_data.json"),
        }
    },
    "fifa": {
        "name": "FIFA 21 Player Data",
        "description": "Player attributes for feature engineering and visualization.",
        "files": {
            "csv": os.path.join(DATA_DIR, "fifa21 raw data v2.csv"),
        }
    },
    "pokedex": {
        "name": "Pokedex Data",
        "description": "Pokemon-style dataset for categorical and numeric analysis.",
        "files": {
            "csv": os.path.join(DATA_DIR, "pokedex.csv"),
        }
    }
}


# ============================================================
# HELPERS
# ============================================================
def infer_file_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return {
        ".csv": "csv",
        ".json": "json",
        ".xlsx": "xlsx"
    }.get(ext, "")


def load_data_from_file(file_path: str, file_type: str) -> pd.DataFrame:
    if file_type == "csv":
        return pd.read_csv(file_path)
    elif file_type == "json":
        return pd.read_json(file_path)
    elif file_type == "xlsx":
        return pd.read_excel(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def load_builtin_dataset(dataset_key: str, file_format: str) -> pd.DataFrame:
    dataset = BUILT_IN_DATASETS.get(dataset_key)
    if not dataset:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    file_path = dataset["files"].get(file_format)
    if not file_path:
        raise ValueError(f"Format {file_format} not available for {dataset['name']}")

    if not os.path.isfile(file_path):
        raise ValueError(f"Built-in file not found: {file_path}")

    return load_data_from_file(file_path, file_format)


def get_column_lists(df: pd.DataFrame) -> dict:
    if df is None:
        return {"numeric": [], "categorical": [], "all": []}

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    all_cols = df.columns.tolist()

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "all": all_cols
    }


def summarize_df(df: pd.DataFrame) -> dict:
    if df is None:
        return {"rows": 0, "cols": 0, "missing": 0}
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "missing": int(df.isnull().sum().sum())
    }


def safe_top_categories(series: pd.Series, top_n: int = 10) -> pd.DataFrame:
    vc = series.astype(str).value_counts(dropna=False)
    if len(vc) > top_n:
        top = vc.iloc[:top_n].copy()
        other = vc.iloc[top_n:].sum()
        if other > 0:
            top.loc["Other"] = other
        vc = top
    out = vc.reset_index()
    out.columns = ["Category", "Count"]
    return out


def strongest_correlations(corr_df: pd.DataFrame):
    if corr_df is None or corr_df.shape[1] < 2:
        return None, None

    pairs = []
    cols = corr_df.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr_df.iloc[i, j]))

    if not pairs:
        return None, None

    strongest_pos = max(pairs, key=lambda x: x[2])
    strongest_neg = min(pairs, key=lambda x: x[2])
    return strongest_pos, strongest_neg


def make_feature_stats(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    rows = []
    for col in cols:
        if col not in df.columns:
            continue
        s = df[col]
        row = {
            "Feature": col,
            "Type": str(s.dtype),
            "Non-Null": int(s.notna().sum())
        }
        if pd.api.types.is_numeric_dtype(s):
            row["Mean"] = round(float(s.mean()), 4) if s.notna().any() else np.nan
            row["Std"] = round(float(s.std()), 4) if s.notna().any() else np.nan
            row["Min"] = round(float(s.min()), 4) if s.notna().any() else np.nan
            row["Max"] = round(float(s.max()), 4) if s.notna().any() else np.nan
        else:
            row["Mean"] = np.nan
            row["Std"] = np.nan
            row["Min"] = np.nan
            row["Max"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# UI
# ============================================================
def create_app_ui():
    return ui.page_fillable(
        ui.tags.head(
            ui.tags.title("Interactive Data Analysis App"),
            ui.tags.style("""
                :root {
                    --primary: #2563eb;
                    --primary-dark: #1d4ed8;
                    --bg-soft: #f8fafc;
                    --card-bg: #ffffff;
                    --text-main: #0f172a;
                    --text-muted: #475569;
                    --border: #e2e8f0;
                    --shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
                }

                body {
                    font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
                    background: linear-gradient(180deg, #f8fbff 0%, #f8fafc 100%);
                    color: var(--text-main);
                }

                .app-shell {
                    max-width: 1500px;
                    margin: 0 auto;
                    padding: 20px 18px 36px 18px;
                }

                .hero {
                    background: linear-gradient(135deg, #1d4ed8 0%, #3b82f6 45%, #60a5fa 100%);
                    color: white;
                    padding: 28px 32px;
                    border-radius: 20px;
                    margin-bottom: 18px;
                    box-shadow: 0 12px 30px rgba(37, 99, 235, 0.22);
                }

                .hero-title {
                    font-size: 2rem;
                    font-weight: 800;
                    margin-bottom: 6px;
                }

                .hero-subtitle {
                    font-size: 1rem;
                    opacity: 0.96;
                    margin-bottom: 0;
                }

                .status-banner {
                    background: rgba(255,255,255,0.18);
                    border: 1px solid rgba(255,255,255,0.28);
                    border-radius: 14px;
                    padding: 14px 16px;
                    margin-top: 18px;
                    color: #f8fafc;
                    backdrop-filter: blur(6px);
                }

                .app-card {
                    background: var(--card-bg);
                    border: 1px solid var(--border);
                    border-radius: 18px;
                    box-shadow: var(--shadow);
                    padding: 18px 18px 14px 18px;
                    margin-bottom: 16px;
                }

                .section-title {
                    font-size: 1.08rem;
                    font-weight: 700;
                    margin-bottom: 10px;
                    color: var(--text-main);
                }

                .section-subtitle {
                    color: var(--text-muted);
                    font-size: 0.93rem;
                    margin-bottom: 12px;
                }

                .mini-stat {
                    background: linear-gradient(180deg, #f8fbff 0%, #f1f5f9 100%);
                    border: 1px solid var(--border);
                    border-radius: 14px;
                    padding: 14px;
                    text-align: center;
                    margin-bottom: 10px;
                }

                .mini-stat-number {
                    font-size: 1.3rem;
                    font-weight: 800;
                    color: var(--primary);
                    margin-top: 4px;
                }

                .muted {
                    color: var(--text-muted);
                    font-size: 0.92rem;
                }

                .pill {
                    display: inline-block;
                    padding: 6px 10px;
                    border-radius: 999px;
                    background: #eff6ff;
                    color: #1e40af;
                    font-size: 0.82rem;
                    font-weight: 600;
                    margin-right: 6px;
                    margin-bottom: 6px;
                    border: 1px solid #dbeafe;
                }

                .info-panel {
                    background: #f8fafc;
                    border: 1px solid var(--border);
                    border-radius: 14px;
                    padding: 14px;
                }

                .nav-tabs {
                    gap: 4px;
                    border-bottom: none !important;
                    margin-bottom: 12px;
                }

                .nav-tabs .nav-link {
                    border-radius: 12px 12px 0 0 !important;
                    border: 1px solid transparent !important;
                    font-weight: 600;
                    color: #334155 !important;
                    padding: 10px 14px;
                }

                .nav-tabs .nav-link.active {
                    background: white !important;
                    border-color: var(--border) !important;
                    color: var(--primary-dark) !important;
                }

                .btn-primary {
                    background-color: var(--primary) !important;
                    border-color: var(--primary) !important;
                    border-radius: 12px !important;
                    font-weight: 700 !important;
                    padding: 10px 14px !important;
                }

                .btn-secondary {
                    border-radius: 12px !important;
                    font-weight: 700 !important;
                    padding: 10px 14px !important;
                }

                .form-control, .form-select {
                    border-radius: 12px !important;
                    border: 1px solid #dbe2ea !important;
                    box-shadow: none !important;
                }

                .control-label, .form-label {
                    font-weight: 600;
                    color: #1e293b;
                }

                .tab-pane {
                    padding-top: 10px;
                }

                .footer-note {
                    text-align: center;
                    color: #64748b;
                    font-size: 0.9rem;
                    margin-top: 12px;
                }

                .workflow-step {
                    padding: 12px 14px;
                    border-radius: 12px;
                    background: #f8fafc;
                    border: 1px solid var(--border);
                    margin-bottom: 10px;
                }

                .workflow-step strong {
                    color: #0f172a;
                }

                table.dataTable {
                    font-size: 13px;
                }
            """)
        ),

        ui.div(
            ui.div(
                ui.div(
                    ui.div("📊 Interactive Data Analysis App", class_="hero-title"),
                    ui.p(
                        "Load, clean, transform, and explore data through an interactive Python Shiny workflow.",
                        class_="hero-subtitle"
                    ),
                    ui.output_ui("dataset_status_banner"),
                    class_="hero"
                ),

                ui.navset_tab(
                    # =====================================================
                    # HOME
                    # =====================================================
                    ui.nav_panel(
                        "🏠 Home",
                        ui.row(
                            ui.column(
                                7,
                                ui.div(
                                    ui.div("Welcome", class_="section-title"),
                                    ui.p(
                                        "This app supports end-to-end interactive data analysis with built-in datasets, preprocessing tools, feature engineering, and customizable visual exploration.",
                                        class_="section-subtitle"
                                    ),
                                    ui.div(
                                        ui.span("CSV / JSON / XLSX", class_="pill"),
                                        ui.span("Built-in datasets", class_="pill"),
                                        ui.span("Interactive cleaning", class_="pill"),
                                        ui.span("Feature engineering", class_="pill"),
                                        ui.span("Plotly EDA", class_="pill"),
                                    ),
                                    ui.br(),
                                    ui.br(),
                                    ui.div(
                                        ui.div(
                                            ui.strong("1. Load data"),
                                            ui.p("Upload a file or choose a built-in dataset.", class_="muted"),
                                            class_="workflow-step"
                                        ),
                                        ui.div(
                                            ui.strong("2. Clean and preprocess"),
                                            ui.p("Handle missing values, duplicates, outliers, scaling, and encoding.", class_="muted"),
                                            class_="workflow-step"
                                        ),
                                        ui.div(
                                            ui.strong("3. Engineer features"),
                                            ui.p("Create transformed features with logs, square roots, bins, encoding, and scaling.", class_="muted"),
                                            class_="workflow-step"
                                        ),
                                        ui.div(
                                            ui.strong("4. Explore visually"),
                                            ui.p("Filter data and generate interactive plots with dynamic insights.", class_="muted"),
                                            class_="workflow-step"
                                        ),
                                    ),
                                    class_="app-card"
                                )
                            ),
                            ui.column(
                                5,
                                ui.div(
                                    ui.div("Built-in Datasets", class_="section-title"),
                                    ui.div(
                                        ui.div(ui.strong("Financial News Events"), ui.p("Mixed event-style dataset", class_="muted"), class_="workflow-step"),
                                        ui.div(ui.strong("Grocery Chain Data"), ui.p("Retail / operations dataset", class_="muted"), class_="workflow-step"),
                                        ui.div(ui.strong("FIFA 21 Player Data"), ui.p("Player attributes and ratings", class_="muted"), class_="workflow-step"),
                                        ui.div(ui.strong("Pokedex Data"), ui.p("Categorical and numeric toy dataset", class_="muted"), class_="workflow-step"),
                                    ),
                                    ui.hr(),
                                    ui.p(
                                        ui.strong("Tip: "),
                                        "Use Reset to Original Data any time you want to start over from the loaded dataset.",
                                        class_="muted"
                                    ),
                                    class_="app-card"
                                )
                            )
                        )
                    ),

                    # =====================================================
                    # DATA LOADING
                    # =====================================================
                    ui.nav_panel(
                        "📁 Data Loading",
                        ui.row(
                            ui.column(
                                4,
                                ui.div(
                                    ui.div("Load a Dataset", class_="section-title"),
                                    ui.p("Choose between uploading a file or using one of the built-in datasets.", class_="section-subtitle"),
                                    ui.input_radio_buttons(
                                        "data_source",
                                        "Data source",
                                        {"upload": "Upload File", "builtin": "Built-in Dataset"},
                                        selected="upload"
                                    ),
                                    ui.panel_conditional(
                                        "input.data_source === 'upload'",
                                        ui.div(
                                            ui.input_file(
                                                "uploaded_file",
                                                "Choose file",
                                                accept=[".csv", ".json", ".xlsx"],
                                                multiple=False
                                            ),
                                            ui.p("File type is detected automatically from the filename.", class_="muted")
                                        )
                                    ),
                                    ui.panel_conditional(
                                        "input.data_source === 'builtin'",
                                        ui.div(
                                            ui.input_select(
                                                "builtin_dataset",
                                                "Dataset",
                                                {
                                                    "financial_news": "Financial News Events",
                                                    "grocery": "Grocery Chain Data",
                                                    "fifa": "FIFA 21 Player Data",
                                                    "pokedex": "Pokedex Data",
                                                }
                                            ),
                                            ui.input_select(
                                                "builtin_format",
                                                "Format",
                                                {"csv": "CSV"}
                                            ),
                                            ui.output_ui("builtin_dataset_info"),
                                        )
                                    ),
                                    ui.br(),
                                    ui.input_action_button("load_data", "Load Data", class_="btn-primary w-100"),
                                    ui.br(),
                                    ui.br(),
                                    ui.input_action_button("reset_data", "Reset to Original Data", class_="btn-secondary w-100"),
                                    class_="app-card"
                                )
                            ),
                            ui.column(
                                8,
                                ui.div(
                                    ui.div("Preview", class_="section-title"),
                                    ui.output_data_frame("data_preview"),
                                    ui.hr(),
                                    ui.row(
                                        ui.column(4, ui.output_ui("rows_box")),
                                        ui.column(4, ui.output_ui("cols_box")),
                                        ui.column(4, ui.output_ui("missing_box")),
                                    ),
                                    class_="app-card"
                                )
                            )
                        )
                    ),

                    # =====================================================
                    # DATA CLEANING
                    # =====================================================
                    ui.nav_panel(
                        "🧹 Data Cleaning",
                        ui.row(
                            ui.column(
                                4,
                                ui.div(
                                    ui.div("Cleaning Pipeline", class_="section-title"),
                                    ui.p("Choose the preprocessing steps to apply, then click Apply Cleaning.", class_="section-subtitle"),

                                    ui.h6("Missing Values"),
                                    ui.input_select(
                                        "missing_strategy",
                                        "Strategy",
                                        {
                                            "none": "Keep as is",
                                            "mean": "Mean (numeric)",
                                            "median": "Median (numeric)",
                                            "mode": "Mode",
                                            "drop_rows": "Drop rows with missing",
                                            "drop_cols": "Drop columns with missing",
                                            "forward_fill": "Forward fill",
                                            "backward_fill": "Backward fill"
                                        }
                                    ),

                                    ui.hr(),
                                    ui.h6("Duplicates"),
                                    ui.input_checkbox("handle_duplicates", "Remove duplicate rows", True),

                                    ui.hr(),
                                    ui.h6("Text Cleaning"),
                                    ui.input_checkbox("strip_whitespace", "Trim whitespace in text columns", False),
                                    ui.input_checkbox("lowercase_text", "Convert text to lowercase", False),

                                    ui.hr(),
                                    ui.h6("Outliers"),
                                    ui.input_checkbox("enable_outliers", "Handle numeric outliers", False),
                                    ui.panel_conditional(
                                        "input.enable_outliers === true",
                                        ui.div(
                                            ui.input_select(
                                                "outlier_method",
                                                "Method",
                                                {
                                                    "iqr_cap": "IQR capping",
                                                    "iqr_remove": "IQR row removal"
                                                }
                                            ),
                                            ui.input_select(
                                                "outlier_columns",
                                                "Numeric columns",
                                                choices=[],
                                                multiple=True
                                            )
                                        )
                                    ),

                                    ui.hr(),
                                    ui.h6("Scaling"),
                                    ui.input_checkbox("enable_scaling", "Scale numeric columns", False),
                                    ui.panel_conditional(
                                        "input.enable_scaling === true",
                                        ui.div(
                                            ui.input_select(
                                                "cleaning_scale_method",
                                                "Method",
                                                {
                                                    "standard": "Standard (Z-score)",
                                                    "minmax": "Min-Max [0,1]",
                                                    "robust": "Robust"
                                                }
                                            ),
                                            ui.input_select(
                                                "cleaning_scale_columns",
                                                "Numeric columns",
                                                choices=[],
                                                multiple=True
                                            )
                                        )
                                    ),

                                    ui.hr(),
                                    ui.h6("Encoding"),
                                    ui.input_checkbox("enable_encoding", "Encode categorical columns", False),
                                    ui.panel_conditional(
                                        "input.enable_encoding === true",
                                        ui.div(
                                            ui.input_select(
                                                "cleaning_encode_method",
                                                "Method",
                                                {
                                                    "label": "Label Encoding",
                                                    "onehot": "One-Hot Encoding"
                                                }
                                            ),
                                            ui.input_select(
                                                "cleaning_encode_columns",
                                                "Categorical columns",
                                                choices=[],
                                                multiple=True
                                            )
                                        )
                                    ),

                                    ui.br(),
                                    ui.input_action_button("apply_cleaning", "Apply Cleaning", class_="btn-primary w-100"),
                                    class_="app-card"
                                )
                            ),
                            ui.column(
                                8,
                                ui.div(
                                    ui.div("Cleaning Feedback", class_="section-title"),
                                    ui.row(
                                        ui.column(6, ui.output_ui("cleaning_feedback_before")),
                                        ui.column(6, ui.output_ui("cleaning_feedback_after"))
                                    ),
                                    ui.br(),
                                    ui.output_ui("cleaning_log"),
                                    ui.hr(),
                                    ui.div("Cleaned Data Preview", class_="section-title"),
                                    ui.output_data_frame("cleaned_data_preview"),
                                    ui.hr(),
                                    ui.row(
                                        ui.column(6, ui.output_text("cleaned_rows")),
                                        ui.column(6, ui.output_text("cleaned_cols"))
                                    ),
                                    ui.hr(),
                                    ui.div("Missing Summary", class_="section-title"),
                                    ui.output_table("missing_summary"),
                                    class_="app-card"
                                )
                            )
                        )
                    ),

                    # =====================================================
                    # FEATURE ENGINEERING
                    # =====================================================
                    ui.nav_panel(
                        "⚙️ Feature Engineering",
                        ui.row(
                            ui.column(
                                4,
                                ui.div(
                                    ui.div("Create New Features", class_="section-title"),
                                    ui.p("Apply feature transformations and inspect what was created.", class_="section-subtitle"),
                                    ui.input_select(
                                        "feature_type",
                                        "Transformation",
                                        {
                                            "scale": "Scaling",
                                            "encode": "Categorical Encoding",
                                            "bin": "Binning",
                                            "log": "Log Transform",
                                            "sqrt": "Square Root Transform",
                                            "poly2": "Square / Polynomial Feature"
                                        }
                                    ),

                                    ui.panel_conditional(
                                        "input.feature_type === 'scale' || input.feature_type === 'log' || input.feature_type === 'sqrt' || input.feature_type === 'poly2'",
                                        ui.div(
                                            ui.panel_conditional(
                                                "input.feature_type === 'scale'",
                                                ui.input_select(
                                                    "scaling_method",
                                                    "Scaling method",
                                                    {
                                                        "standard": "Standard Scaler (Z-score)",
                                                        "minmax": "Min-Max Scaling"
                                                    }
                                                )
                                            ),
                                            ui.input_select(
                                                "scale_columns",
                                                "Numeric columns",
                                                choices=[],
                                                multiple=True
                                            )
                                        )
                                    ),

                                    ui.panel_conditional(
                                        "input.feature_type === 'encode'",
                                        ui.div(
                                            ui.input_select(
                                                "encoding_method",
                                                "Encoding method",
                                                {
                                                    "label": "Label Encoding",
                                                    "onehot": "One-Hot Encoding"
                                                }
                                            ),
                                            ui.input_select(
                                                "encode_columns",
                                                "Categorical columns",
                                                choices=[],
                                                multiple=True
                                            )
                                        )
                                    ),

                                    ui.panel_conditional(
                                        "input.feature_type === 'bin'",
                                        ui.div(
                                            ui.input_select(
                                                "bin_column",
                                                "Numeric column",
                                                choices=[]
                                            ),
                                            ui.input_slider("bin_count", "Number of bins", 2, 10, 5),
                                            ui.input_select(
                                                "bin_method",
                                                "Binning method",
                                                {
                                                    "equal": "Equal Width",
                                                    "quantile": "Equal Frequency"
                                                }
                                            )
                                        )
                                    ),

                                    ui.br(),
                                    ui.input_action_button("apply_feature", "Apply Transformation", class_="btn-primary w-100"),
                                    class_="app-card"
                                )
                            ),
                            ui.column(
                                8,
                                ui.div(
                                    ui.div("Feature Engineering Feedback", class_="section-title"),
                                    ui.output_ui("feature_type_explanation"),
                                    ui.br(),
                                    ui.output_ui("feature_eng_feedback"),
                                    ui.br(),
                                    ui.output_ui("feature_eng_log_ui"),
                                    ui.hr(),
                                    ui.div("Engineered Data Preview", class_="section-title"),
                                    ui.output_data_frame("transformed_data_preview"),
                                    ui.hr(),
                                    ui.div("New Features Created", class_="section-title"),
                                    ui.output_table("feature_summary"),
                                    ui.hr(),
                                    ui.div("Feature Preview Statistics", class_="section-title"),
                                    ui.output_table("feature_preview_stats"),
                                    class_="app-card"
                                )
                            )
                        )
                    ),

                    # =====================================================
                    # EDA
                    # =====================================================
                    ui.nav_panel(
                        "📈 EDA",
                        ui.row(
                            ui.column(
                                3,
                                ui.div(
                                    ui.div("Visualization Controls", class_="section-title"),
                                    ui.p("Adjust settings and click Generate Plot to reduce lag.", class_="section-subtitle"),

                                    ui.input_select(
                                        "plot_type",
                                        "Plot type",
                                        {
                                            "histogram": "Histogram",
                                            "box": "Box Plot",
                                            "scatter": "Scatter Plot",
                                            "bar": "Bar Chart",
                                            "pie": "Pie Chart",
                                            "correlation": "Correlation Heatmap"
                                        }
                                    ),
                                    ui.input_select("x_axis", "X-axis", choices=["None"]),
                                    ui.input_select("y_axis", "Y-axis", choices=["None"]),
                                    ui.input_select("color_by", "Color by", choices=["None"]),
                                    ui.input_slider("plot_height", "Plot height", 320, 900, 520),

                                    ui.hr(),
                                    ui.h6("Filters"),
                                    ui.input_select("filter_num_col", "Numeric filter column", choices=["None"]),
                                    ui.panel_conditional(
                                        "input.filter_num_col !== 'None'",
                                        ui.div(
                                            ui.input_slider("filter_num_range", "Numeric range", 0, 100, (0, 100))
                                        )
                                    ),
                                    ui.input_select("filter_cat_col", "Categorical filter column", choices=["None"]),
                                    ui.input_select(
                                        "filter_cat_values",
                                        "Allowed categories",
                                        choices=[],
                                        multiple=True
                                    ),

                                    ui.hr(),
                                    ui.h6("Plot Options"),
                                    ui.input_slider("hist_bins", "Histogram bins", 5, 60, 20),
                                    ui.input_slider("top_n_categories", "Top N categories", 3, 20, 10),

                                    ui.br(),
                                    ui.input_action_button("generate_plot", "Generate Plot", class_="btn-primary w-100"),
                                    class_="app-card"
                                )
                            ),
                            ui.column(
                                9,
                                ui.div(
                                    ui.div("Interactive Visualization", class_="section-title"),
                                    output_widget("eda_plot"),
                                    ui.hr(),
                                    ui.div("Dynamic Insights", class_="section-title"),
                                    ui.output_ui("eda_insights"),
                                    ui.hr(),
                                    ui.div("Summary Statistics", class_="section-title"),
                                    ui.output_table("stat_summary"),
                                    class_="app-card"
                                )
                            )
                        )
                    ),

                    # =====================================================
                    # ABOUT
                    # =====================================================
                    ui.nav_panel(
                        "ℹ️ About",
                        ui.row(
                            ui.column(
                                7,
                                ui.div(
                                    ui.div("About This Application", class_="section-title"),
                                    ui.p(
                                        "This project was developed for STAT5243 and demonstrates interactive web-based data analysis in Python Shiny.",
                                        class_="section-subtitle"
                                    ),
                                    ui.div(
                                        ui.strong("Technology stack"),
                                        ui.tags.ul(
                                            ui.tags.li("Shiny for Python"),
                                            ui.tags.li("Pandas / NumPy"),
                                            ui.tags.li("Plotly"),
                                            ui.tags.li("scikit-learn"),
                                        ),
                                        class_="info-panel"
                                    ),
                                    class_="app-card"
                                )
                            ),
                            ui.column(
                                5,
                                ui.div(
                                    ui.div("Team Members", class_="section-title"),
                                    ui.div(
                                        ui.p("Freya Chen (yc4684)"),
                                        ui.p("Zhuyun Jin (zj2434)"),
                                        ui.p("Nikhil Shanbhag (nvs2128)"),
                                        class_="info-panel"
                                    ),
                                    class_="app-card"
                                )
                            )
                        )
                    )
                ),

                ui.div(
                    "STAT5243 Project 2 - Interactive Data Analysis App",
                    class_="footer-note"
                ),
                class_="app-shell"
            )
        )
    )


# ============================================================
# SERVER
# ============================================================
def server(input, output, session):
    raw_data = reactive.Value(None)
    processed_data = reactive.Value(None)
    feature_data = reactive.Value(None)

    current_dataset_name = reactive.Value("None loaded")
    current_stage = reactive.Value("No data loaded")

    applied_cleaning_log = reactive.Value([])
    before_cleaning_stats = reactive.Value(None)

    feature_eng_log = reactive.Value([])
    feature_preview_df = reactive.Value(pd.DataFrame())

    last_columns = reactive.Value([])

    plot_cache = reactive.Value(None)

    # --------------------------------------------------------
    # Cached helpers
    # --------------------------------------------------------
    @reactive.calc
    def active_df():
        return feature_data()

    @reactive.calc
    def filtered_df():
        df = active_df()
        if df is None:
            return None

        out = df

        num_col = input.filter_num_col()
        if num_col and num_col != "None" and num_col in out.columns and pd.api.types.is_numeric_dtype(out[num_col]):
            rng = input.filter_num_range()
            if rng is not None and len(rng) == 2:
                out = out[out[num_col].between(rng[0], rng[1], inclusive="both") | out[num_col].isna()]

        cat_col = input.filter_cat_col()
        cat_vals = input.filter_cat_values()
        if cat_col and cat_col != "None" and cat_col in out.columns and cat_vals:
            out = out[out[cat_col].astype(str).isin(list(cat_vals))]

        return out

    @reactive.calc
    def viz_df():
        df = filtered_df()
        if df is None:
            return None
        if len(df) > 3000:
            return df.sample(3000, random_state=42)
        return df

    @reactive.calc
    def numeric_summary():
        df = filtered_df()
        if df is None:
            return pd.DataFrame()
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return pd.DataFrame()
        return numeric_df.describe().T.reset_index().rename(columns={"index": "Column"})

    @reactive.calc
    def corr_matrix():
        df = filtered_df()
        if df is None:
            return None
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return None
        if numeric_df.shape[1] > 20:
            numeric_df = numeric_df.iloc[:, :20]
        return numeric_df.corr()

    # --------------------------------------------------------
    # UI updates only when column names change
    # --------------------------------------------------------
    @reactive.Effect
    def refresh_selectors_if_needed():
        df = active_df()
        if df is None:
            return

        current_cols = df.columns.tolist()
        if current_cols == last_columns():
            return

        last_columns.set(current_cols)

        cols = get_column_lists(df)
        numeric_cols = cols["numeric"]
        categorical_cols = cols["categorical"]
        all_cols = cols["all"]

        # cleaning
        ui.update_select("outlier_columns", choices=numeric_cols, selected=[])
        ui.update_select("cleaning_scale_columns", choices=numeric_cols, selected=[])
        ui.update_select("cleaning_encode_columns", choices=categorical_cols, selected=[])

        # feature eng
        ui.update_select("scale_columns", choices=numeric_cols, selected=[])
        ui.update_select("encode_columns", choices=categorical_cols, selected=[])
        ui.update_select("bin_column", choices=numeric_cols, selected=numeric_cols[0] if numeric_cols else None)

        # eda
        ui.update_select("x_axis", choices=["None"] + all_cols, selected="None")
        ui.update_select("y_axis", choices=["None"] + all_cols, selected="None")
        ui.update_select("color_by", choices=["None"] + categorical_cols, selected="None")

        ui.update_select("filter_num_col", choices=["None"] + numeric_cols, selected="None")
        ui.update_select("filter_cat_col", choices=["None"] + categorical_cols, selected="None")

    # --------------------------------------------------------
    # Built-in dataset info
    # --------------------------------------------------------
    @reactive.Effect
    def update_builtin_format():
        ds = BUILT_IN_DATASETS.get(input.builtin_dataset())
        if ds:
            formats = list(ds["files"].keys())
            ui.update_select(
                "builtin_format",
                choices={fmt: fmt.upper() for fmt in formats},
                selected=formats[0] if formats else None
            )

    @output
    @render.ui
    def builtin_dataset_info():
        ds = BUILT_IN_DATASETS.get(input.builtin_dataset())
        if not ds:
            return ui.div()
        return ui.div(
            ui.p(ui.strong("Name: "), ds["name"]),
            ui.p(ui.strong("Description: "), ds["description"]),
            ui.p(ui.strong("Available formats: "), ", ".join(ds["files"].keys())),
            class_="info-panel"
        )

    @output
    @render.ui
    def dataset_status_banner():
        df = active_df()
        if df is None:
            return ui.div(
                ui.p(ui.strong("Current dataset: "), "None"),
                ui.p(ui.strong("Stage: "), "No data loaded"),
                class_="status-banner"
            )
        return ui.div(
            ui.p(ui.strong("Current dataset: "), current_dataset_name()),
            ui.p(ui.strong("Stage: "), current_stage()),
            ui.p(ui.strong("Shape: "), f"{df.shape[0]:,} rows × {df.shape[1]} columns"),
            class_="status-banner"
        )

    # --------------------------------------------------------
    # Data loading
    # --------------------------------------------------------
    @reactive.Effect
    @reactive.event(input.load_data)
    def load_data():
        try:
            if input.data_source() == "upload":
                file_info = input.uploaded_file()
                if file_info is None:
                    ui.notification_show("Please upload a file first.", type="warning")
                    return

                file = file_info[0]
                path = file.get("datapath") or file.get("path")
                filename = file.get("name", "")

                if not path or not os.path.isfile(path):
                    ui.notification_show("Could not read uploaded file.", type="error")
                    return

                file_type = infer_file_type(filename)
                if not file_type:
                    ui.notification_show("Unsupported file type. Please upload CSV, JSON, or XLSX.", type="error")
                    return

                df = load_data_from_file(path, file_type)
                dataset_name = f"Uploaded: {filename}"

            else:
                key = input.builtin_dataset()
                fmt = input.builtin_format()
                df = load_builtin_dataset(key, fmt)
                dataset_name = BUILT_IN_DATASETS[key]["name"]

            if df is None or df.empty:
                ui.notification_show("Loaded dataset is empty.", type="warning")
                return

            raw_data.set(df.copy())
            processed_data.set(df.copy())
            feature_data.set(df.copy())

            current_dataset_name.set(dataset_name)
            current_stage.set("Raw data loaded")

            applied_cleaning_log.set([])
            before_cleaning_stats.set(None)
            feature_eng_log.set([])
            feature_preview_df.set(pd.DataFrame())
            plot_cache.set(None)

            ui.notification_show(
                f"Loaded successfully: {df.shape[0]:,} rows and {df.shape[1]} columns.",
                type="message"
            )

        except Exception as e:
            ui.notification_show(f"Error loading data: {str(e)}", type="error")

    @reactive.Effect
    @reactive.event(input.reset_data)
    def reset_data():
        df = raw_data()
        if df is None:
            ui.notification_show("No original dataset available to reset.", type="warning")
            return

        processed_data.set(df.copy())
        feature_data.set(df.copy())
        current_stage.set("Reset to original raw data")

        applied_cleaning_log.set([])
        before_cleaning_stats.set(None)
        feature_eng_log.set([])
        feature_preview_df.set(pd.DataFrame())
        plot_cache.set(None)

        ui.notification_show("Reset to original loaded dataset.", type="message")

    # --------------------------------------------------------
    # Preview + metrics
    # --------------------------------------------------------
    @output
    @render.data_frame
    def data_preview():
        df = raw_data()
        if df is None:
            return pd.DataFrame()
        return render.DataGrid(df.head(50), filters=True)

    @output
    @render.ui
    def rows_box():
        df = raw_data()
        n = f"{df.shape[0]:,}" if df is not None else "—"
        return ui.div(
            ui.div("Rows", class_="muted"),
            ui.div(n, class_="mini-stat-number"),
            class_="mini-stat"
        )

    @output
    @render.ui
    def cols_box():
        df = raw_data()
        n = str(df.shape[1]) if df is not None else "—"
        return ui.div(
            ui.div("Columns", class_="muted"),
            ui.div(n, class_="mini-stat-number"),
            class_="mini-stat"
        )

    @output
    @render.ui
    def missing_box():
        df = raw_data()
        n = f"{int(df.isnull().sum().sum()):,}" if df is not None else "—"
        return ui.div(
            ui.div("Missing Values", class_="muted"),
            ui.div(n, class_="mini-stat-number"),
            class_="mini-stat"
        )

    # --------------------------------------------------------
    # Cleaning
    # --------------------------------------------------------
    @reactive.Effect
    @reactive.event(input.apply_cleaning)
    def apply_cleaning():
        df = active_df()
        if df is None:
            ui.notification_show("Please load data first.", type="warning")
            return

        df = df.copy()
        log = []
        before_cleaning_stats.set(summarize_df(df))

        try:
            # Missing values
            strategy = input.missing_strategy()
            if strategy != "none":
                missing_before = int(df.isnull().sum().sum())

                if strategy == "drop_rows":
                    df = df.dropna()
                    log.append("Dropped rows containing missing values.")
                elif strategy == "drop_cols":
                    df = df.dropna(axis=1)
                    log.append("Dropped columns containing missing values.")
                elif strategy == "mean":
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    if len(num_cols) > 0:
                        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
                        log.append(f"Applied mean imputation to {len(num_cols)} numeric column(s).")
                elif strategy == "median":
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    if len(num_cols) > 0:
                        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
                        log.append(f"Applied median imputation to {len(num_cols)} numeric column(s).")
                elif strategy == "mode":
                    for col in df.columns:
                        if df[col].isnull().any():
                            modes = df[col].mode()
                            if not modes.empty:
                                df[col] = df[col].fillna(modes.iloc[0])
                    log.append("Applied mode imputation.")
                elif strategy == "forward_fill":
                    df = df.ffill()
                    log.append("Applied forward fill.")
                elif strategy == "backward_fill":
                    df = df.bfill()
                    log.append("Applied backward fill.")

                missing_after = int(df.isnull().sum().sum())
                log.append(f"Missing values changed from {missing_before:,} to {missing_after:,}.")

            # Duplicates
            if input.handle_duplicates():
                before_rows = len(df)
                df = df.drop_duplicates()
                removed = before_rows - len(df)
                log.append(f"Removed {removed:,} duplicate row(s).")

            # Text cleaning
            text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

            if input.strip_whitespace() and text_cols:
                for col in text_cols:
                    df[col] = df[col].astype(str).str.strip()
                log.append(f"Trimmed whitespace in {len(text_cols)} text column(s).")

            if input.lowercase_text() and text_cols:
                for col in text_cols:
                    df[col] = df[col].astype(str).str.lower()
                log.append(f"Converted text to lowercase in {len(text_cols)} column(s).")

            # Outliers
            if input.enable_outliers():
                requested = list(input.outlier_columns()) if input.outlier_columns() else []
                valid_cols = [c for c in requested if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                method = input.outlier_method()

                if valid_cols:
                    if method == "iqr_cap":
                        for col in valid_cols:
                            q1 = df[col].quantile(0.25)
                            q3 = df[col].quantile(0.75)
                            iqr = q3 - q1
                            lower = q1 - 1.5 * iqr
                            upper = q3 + 1.5 * iqr
                            df[col] = df[col].clip(lower=lower, upper=upper)
                        log.append(f"Applied IQR capping to {len(valid_cols)} numeric column(s).")
                    else:
                        before_rows = len(df)
                        mask = pd.Series(True, index=df.index)
                        for col in valid_cols:
                            q1 = df[col].quantile(0.25)
                            q3 = df[col].quantile(0.75)
                            iqr = q3 - q1
                            lower = q1 - 1.5 * iqr
                            upper = q3 + 1.5 * iqr
                            mask &= df[col].between(lower, upper) | df[col].isna()
                        df = df[mask]
                        removed = before_rows - len(df)
                        log.append(f"Removed {removed:,} row(s) using IQR outlier filtering.")
                else:
                    log.append("Outlier handling skipped because no valid numeric columns were selected.")

            # Scaling
            if input.enable_scaling():
                requested = list(input.cleaning_scale_columns()) if input.cleaning_scale_columns() else []
                valid_cols = [c for c in requested if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if valid_cols:
                    method = input.cleaning_scale_method()
                    scaler = (
                        StandardScaler() if method == "standard"
                        else MinMaxScaler() if method == "minmax"
                        else RobustScaler()
                    )
                    df[valid_cols] = scaler.fit_transform(df[valid_cols])
                    log.append(f"Applied {method} scaling to {len(valid_cols)} numeric column(s).")

            # Encoding
            if input.enable_encoding():
                requested = list(input.cleaning_encode_columns()) if input.cleaning_encode_columns() else []
                valid_cols = [c for c in requested if c in df.columns]

                if valid_cols:
                    method = input.cleaning_encode_method()
                    if method == "label":
                        for col in valid_cols:
                            df[f"{col}_encoded"] = LabelEncoder().fit_transform(df[col].astype(str))
                        log.append(f"Applied label encoding to {len(valid_cols)} column(s).")
                    else:
                        keep_cols = []
                        skipped = []
                        for col in valid_cols:
                            nunique = df[col].astype(str).nunique(dropna=False)
                            if nunique <= 20:
                                keep_cols.append(col)
                            else:
                                skipped.append(col)

                        if skipped:
                            ui.notification_show(
                                f"Skipped high-cardinality one-hot encoding for: {', '.join(skipped[:3])}",
                                type="warning"
                            )

                        if keep_cols:
                            df = pd.get_dummies(df, columns=keep_cols, prefix=keep_cols)
                            log.append(f"Applied one-hot encoding to {len(keep_cols)} column(s).")

            processed_data.set(df.copy())
            feature_data.set(df.copy())
            current_stage.set("Data cleaned / preprocessed")
            applied_cleaning_log.set(log)
            plot_cache.set(None)

            ui.notification_show("Cleaning applied successfully.", type="message")

        except Exception as e:
            applied_cleaning_log.set([f"Error during cleaning: {str(e)}"])
            ui.notification_show(f"Error during cleaning: {str(e)}", type="error")

    @output
    @render.ui
    def cleaning_feedback_before():
        stats = before_cleaning_stats()
        if stats is None:
            return ui.div("Shown after cleaning is applied.", class_="info-panel muted")
        return ui.div(
            ui.p(ui.strong("Rows: "), f"{stats['rows']:,}"),
            ui.p(ui.strong("Columns: "), str(stats["cols"])),
            ui.p(ui.strong("Missing: "), f"{stats['missing']:,}"),
            class_="info-panel"
        )

    @output
    @render.ui
    def cleaning_feedback_after():
        df = processed_data()
        if df is None:
            return ui.div("No cleaned data yet.", class_="info-panel muted")
        stats = summarize_df(df)
        return ui.div(
            ui.p(ui.strong("Rows: "), f"{stats['rows']:,}"),
            ui.p(ui.strong("Columns: "), str(stats["cols"])),
            ui.p(ui.strong("Missing: "), f"{stats['missing']:,}"),
            class_="info-panel"
        )

    @output
    @render.ui
    def cleaning_log():
        log = applied_cleaning_log()
        if not log:
            return ui.div("Applied preprocessing steps will appear here.", class_="info-panel muted")
        return ui.div(
            ui.tags.ul(*[ui.tags.li(x) for x in log]),
            class_="info-panel"
        )

    @output
    @render.data_frame
    def cleaned_data_preview():
        df = processed_data()
        if df is None:
            return pd.DataFrame()
        return render.DataGrid(df.head(50), filters=True)

    @output
    @render.text
    def cleaned_rows():
        df = processed_data()
        return "" if df is None else f"Rows: {df.shape[0]:,}"

    @output
    @render.text
    def cleaned_cols():
        df = processed_data()
        return "" if df is None else f"Columns: {df.shape[1]}"

    @output
    @render.table
    def missing_summary():
        df = processed_data()
        if df is None:
            return pd.DataFrame()

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) == 0:
            return pd.DataFrame({"Message": ["No missing values remain."]})

        return pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing.values / len(df) * 100).round(2)
        })

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------
    @output
    @render.ui
    def feature_type_explanation():
        text = {
            "scale": "Scaling standardizes or normalizes numeric features so they are on comparable ranges.",
            "encode": "Encoding converts categorical values into numeric representations for analysis or modeling.",
            "bin": "Binning groups continuous values into intervals.",
            "log": "Log transformation reduces strong right-skew; it requires strictly positive values.",
            "sqrt": "Square-root transformation reduces moderate right-skew; it requires non-negative values.",
            "poly2": "Polynomial features create squared versions of numeric variables to capture nonlinear relationships."
        }
        return ui.div(text.get(input.feature_type(), ""), class_="info-panel")

    @output
    @render.ui
    def feature_eng_feedback():
        df = active_df()
        base_df = processed_data()
        if df is None:
            return ui.div("Load data first.", class_="info-panel muted")

        new_cols = set(df.columns) - set(base_df.columns) if base_df is not None else set()

        return ui.row(
            ui.column(
                6,
                ui.div(
                    ui.div("Total columns", class_="muted"),
                    ui.div(str(len(df.columns)), class_="mini-stat-number"),
                    class_="mini-stat"
                )
            ),
            ui.column(
                6,
                ui.div(
                    ui.div("New features", class_="muted"),
                    ui.div(str(len(new_cols)), class_="mini-stat-number"),
                    class_="mini-stat"
                )
            )
        )

    @output
    @render.ui
    def feature_eng_log_ui():
        log = feature_eng_log()
        if not log:
            return ui.div("Applied feature engineering steps will appear here.", class_="info-panel muted")
        return ui.div(
            ui.tags.ul(*[ui.tags.li(x) for x in log]),
            class_="info-panel"
        )

    @reactive.Effect
    @reactive.event(input.apply_feature)
    def apply_feature():
        df = active_df()
        if df is None:
            ui.notification_show("Please load data first.", type="warning")
            return

        df = df.copy()
        log = list(feature_eng_log())
        created_cols = []

        try:
            ft = input.feature_type()

            if ft == "scale":
                cols = list(input.scale_columns()) if input.scale_columns() else []
                cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if not cols:
                    ui.notification_show("Please select valid numeric columns.", type="warning")
                    return

                method = input.scaling_method()
                scaler = StandardScaler() if method == "standard" else MinMaxScaler()
                df[cols] = scaler.fit_transform(df[cols])
                log.append(f"Scaled {len(cols)} column(s) using {method} scaling.")

            elif ft == "encode":
                cols = list(input.encode_columns()) if input.encode_columns() else []
                cols = [c for c in cols if c in df.columns]
                if not cols:
                    ui.notification_show("Please select valid columns.", type="warning")
                    return

                method = input.encoding_method()
                if method == "label":
                    for col in cols:
                        new_col = f"{col}_encoded"
                        df[new_col] = LabelEncoder().fit_transform(df[col].astype(str))
                        created_cols.append(new_col)
                    log.append(f"Applied label encoding to {len(cols)} column(s).")
                else:
                    keep_cols = []
                    for col in cols:
                        if df[col].astype(str).nunique(dropna=False) <= 20:
                            keep_cols.append(col)
                    if not keep_cols:
                        ui.notification_show("No selected columns were suitable for one-hot encoding.", type="warning")
                        return
                    old_cols = set(df.columns)
                    df = pd.get_dummies(df, columns=keep_cols, prefix=keep_cols)
                    created_cols = sorted(list(set(df.columns) - old_cols))
                    log.append(f"Applied one-hot encoding to {len(keep_cols)} column(s).")

            elif ft == "bin":
                col = input.bin_column()
                if not col or col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                    ui.notification_show("Please select a valid numeric column for binning.", type="warning")
                    return
                n_bins = input.bin_count()
                method = input.bin_method()
                new_col = f"{col}_binned"
                if method == "equal":
                    df[new_col] = pd.cut(df[col], bins=n_bins, duplicates="drop")
                else:
                    df[new_col] = pd.qcut(df[col], q=n_bins, duplicates="drop")
                created_cols.append(new_col)
                log.append(f"Created {new_col} using {method} binning with {n_bins} bins.")

            elif ft == "log":
                cols = list(input.scale_columns()) if input.scale_columns() else []
                cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if not cols:
                    ui.notification_show("Please select valid numeric columns.", type="warning")
                    return
                done = []
                for col in cols:
                    if (df[col] <= 0).any():
                        ui.notification_show(f"Skipped {col}: contains non-positive values.", type="warning")
                        continue
                    new_col = f"{col}_log"
                    df[new_col] = np.log(df[col])
                    created_cols.append(new_col)
                    done.append(col)
                if not done:
                    return
                log.append(f"Created {len(done)} log-transformed feature(s).")

            elif ft == "sqrt":
                cols = list(input.scale_columns()) if input.scale_columns() else []
                cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if not cols:
                    ui.notification_show("Please select valid numeric columns.", type="warning")
                    return
                done = []
                for col in cols:
                    if (df[col] < 0).any():
                        ui.notification_show(f"Skipped {col}: contains negative values.", type="warning")
                        continue
                    new_col = f"{col}_sqrt"
                    df[new_col] = np.sqrt(df[col])
                    created_cols.append(new_col)
                    done.append(col)
                if not done:
                    return
                log.append(f"Created {len(done)} square-root transformed feature(s).")

            elif ft == "poly2":
                cols = list(input.scale_columns()) if input.scale_columns() else []
                cols = [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
                if not cols:
                    ui.notification_show("Please select valid numeric columns.", type="warning")
                    return
                for col in cols:
                    new_col = f"{col}_sq"
                    df[new_col] = df[col] ** 2
                    created_cols.append(new_col)
                log.append(f"Created squared features for {len(cols)} column(s).")

            feature_data.set(df)
            feature_eng_log.set(log)
            feature_preview_df.set(make_feature_stats(df, created_cols))
            current_stage.set("Feature engineering applied")
            plot_cache.set(None)

            ui.notification_show("Feature engineering applied successfully.", type="message")

        except Exception as e:
            ui.notification_show(f"Error in feature engineering: {str(e)}", type="error")

    @output
    @render.data_frame
    def transformed_data_preview():
        df = active_df()
        if df is None:
            return pd.DataFrame()
        return render.DataGrid(df.head(50), filters=True)

    @output
    @render.table
    def feature_summary():
        df = active_df()
        base = processed_data()
        if df is None or base is None:
            return pd.DataFrame()

        new_cols = sorted(list(set(df.columns) - set(base.columns)))
        if not new_cols:
            return pd.DataFrame({"Message": ["No new features created yet."]})

        return pd.DataFrame({
            "New Feature": new_cols,
            "Data Type": [str(df[col].dtype) for col in new_cols],
            "Non-Null Count": [int(df[col].notna().sum()) for col in new_cols]
        })

    @output
    @render.table
    def feature_preview_stats():
        fp = feature_preview_df()
        if fp is None or fp.empty:
            return pd.DataFrame({"Message": ["No feature preview statistics available yet."]})
        return fp

    # --------------------------------------------------------
    # Filter UI updates
    # --------------------------------------------------------
    @reactive.Effect
    def update_filter_inputs():
        df = active_df()
        if df is None:
            return

        num_col = input.filter_num_col()
        if num_col and num_col != "None" and num_col in df.columns and pd.api.types.is_numeric_dtype(df[num_col]):
            s = df[num_col].dropna()
            if len(s) > 0:
                mn = float(s.min())
                mx = float(s.max())
                if mn == mx:
                    mx = mn + 1.0
                ui.update_slider("filter_num_range", min=mn, max=mx, value=(mn, mx))

        cat_col = input.filter_cat_col()
        if cat_col and cat_col != "None" and cat_col in df.columns:
            vals = sorted(df[cat_col].astype(str).dropna().unique().tolist())
            selected = vals[: min(len(vals), 10)]
            ui.update_select("filter_cat_values", choices=vals, selected=selected)
        else:
            ui.update_select("filter_cat_values", choices=[], selected=[])

    # --------------------------------------------------------
    # Plot generation (button-triggered for performance)
    # --------------------------------------------------------
    @reactive.Effect
    @reactive.event(input.generate_plot)
    def build_plot():
        df = viz_df()
        full_df = filtered_df()

        if full_df is None:
            fig = go.Figure()
            fig.add_annotation(text="Load data first", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            fig.update_layout(template="plotly_white", height=420)
            plot_cache.set(fig)
            return

        if full_df.empty:
            fig = go.Figure()
            fig.add_annotation(text="No rows remain after filtering", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            fig.update_layout(template="plotly_white", height=420)
            plot_cache.set(fig)
            return

        plot_type = input.plot_type()
        x_col = input.x_axis()
        y_col = input.y_axis()
        color_col = input.color_by()
        color_col = None if color_col == "None" else color_col
        height = input.plot_height()
        hist_bins = input.hist_bins()
        top_n = input.top_n_categories()

        try:
            if plot_type == "histogram":
                if x_col == "None" or x_col not in full_df.columns:
                    raise ValueError("Please select a valid X-axis column.")
                if y_col != "None" and y_col in full_df.columns and pd.api.types.is_numeric_dtype(full_df[y_col]):
                    fig = px.histogram(
                        full_df, x=x_col, y=y_col, color=color_col,
                        histfunc="sum", nbins=hist_bins,
                        title=f"{y_col} aggregated by {x_col}",
                        height=height
                    )
                else:
                    fig = px.histogram(
                        full_df, x=x_col, color=color_col, nbins=hist_bins,
                        title=f"Distribution of {x_col}",
                        height=height
                    )

            elif plot_type == "box":
                if x_col == "None" or x_col not in full_df.columns:
                    raise ValueError("Please select a valid X-axis column.")
                if y_col != "None" and y_col in full_df.columns:
                    fig = px.box(
                        full_df, x=x_col, y=y_col, color=color_col,
                        title=f"Box Plot of {y_col} by {x_col}",
                        height=height
                    )
                else:
                    if not pd.api.types.is_numeric_dtype(full_df[x_col]):
                        raise ValueError("For a single-variable box plot, the selected column must be numeric.")
                    fig = px.box(
                        full_df, y=x_col, color=color_col,
                        title=f"Box Plot of {x_col}",
                        height=height
                    )

            elif plot_type == "scatter":
                if x_col == "None" or y_col == "None":
                    raise ValueError("Please select both X and Y columns for a scatter plot.")
                if x_col not in df.columns or y_col not in df.columns:
                    raise ValueError("Selected X or Y column is not available.")
                if not (pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col])):
                    raise ValueError("Scatter plots require numeric X and Y columns.")

                fig = px.scatter(
                    df, x=x_col, y=y_col, color=color_col,
                    title=f"Scatter Plot: {x_col} vs {y_col}",
                    height=height,
                    trendline="ols"
                )

            elif plot_type == "bar":
                if x_col == "None" or x_col not in full_df.columns:
                    raise ValueError("Please select a valid X-axis column.")
                if y_col != "None" and y_col in full_df.columns and pd.api.types.is_numeric_dtype(full_df[y_col]):
                    agg = full_df.groupby(x_col, dropna=False)[y_col].mean().reset_index()
                    if len(agg) > top_n:
                        agg = agg.sort_values(y_col, ascending=False).head(top_n)
                    fig = px.bar(
                        agg, x=x_col, y=y_col,
                        title=f"Mean {y_col} by {x_col}",
                        height=height
                    )
                else:
                    bar_df = safe_top_categories(full_df[x_col], top_n=top_n)
                    fig = px.bar(
                        bar_df, x="Category", y="Count",
                        title=f"Top categories of {x_col}",
                        height=height
                    )

            elif plot_type == "pie":
                if x_col == "None" or x_col not in full_df.columns:
                    raise ValueError("Please select a valid X-axis column.")
                if y_col != "None" and y_col in full_df.columns and pd.api.types.is_numeric_dtype(full_df[y_col]):
                    agg = full_df.groupby(x_col, dropna=False)[y_col].sum().reset_index()
                    if len(agg) > top_n:
                        agg = agg.sort_values(y_col, ascending=False)
                        top = agg.head(top_n).copy()
                        other = agg.iloc[top_n:][y_col].sum()
                        if other > 0:
                            top.loc[len(top)] = ["Other", other]
                        agg = top
                    fig = px.pie(
                        agg, names=x_col, values=y_col,
                        title=f"{y_col} share by {x_col}",
                        height=height
                    )
                else:
                    pie_df = safe_top_categories(full_df[x_col], top_n=top_n)
                    fig = px.pie(
                        pie_df, names="Category", values="Count",
                        title=f"Category share of {x_col}",
                        height=height
                    )

            elif plot_type == "correlation":
                corr = corr_matrix()
                if corr is None or corr.shape[1] < 2:
                    raise ValueError("Need at least 2 numeric columns for a correlation heatmap.")
                fig = px.imshow(
                    corr,
                    title="Correlation Heatmap",
                    height=height,
                    color_continuous_scale="RdBu_r",
                    zmin=-1,
                    zmax=1
                )

            else:
                raise ValueError("Unsupported plot type.")

            fig.update_layout(template="plotly_white")
            plot_cache.set(fig)

        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"{str(e)}", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
            fig.update_layout(template="plotly_white", height=420)
            plot_cache.set(fig)

    @output
    @render_widget
    def eda_plot():
        fig = plot_cache()
        if fig is None:
            fig = go.Figure()
            fig.add_annotation(
                text="Adjust the controls and click Generate Plot",
                x=0.5, y=0.5, showarrow=False, font=dict(size=16)
            )
            fig.update_layout(template="plotly_white", height=420)
        return fig

    # --------------------------------------------------------
    # EDA insights + summary
    # --------------------------------------------------------
    @output
    @render.ui
    def eda_insights():
        df = filtered_df()
        if df is None:
            return ui.div("Load data first.", class_="info-panel muted")
        if df.empty:
            return ui.div("No rows remain after filtering.", class_="info-panel muted")

        plot_type = input.plot_type()
        x_col = input.x_axis()
        y_col = input.y_axis()

        items = [
            ui.tags.li(f"Rows after filtering: {len(df):,}"),
            ui.tags.li(f"Columns available: {df.shape[1]}")
        ]

        try:
            if plot_type == "histogram" and x_col != "None" and x_col in df.columns:
                s = df[x_col]
                items.append(ui.tags.li(f"Missing values in {x_col}: {int(s.isna().sum()):,}"))
                if pd.api.types.is_numeric_dtype(s):
                    items.append(ui.tags.li(f"Mean: {round(float(s.mean()), 4)}"))
                    items.append(ui.tags.li(f"Median: {round(float(s.median()), 4)}"))
                    items.append(ui.tags.li(f"Std: {round(float(s.std()), 4)}"))

            elif plot_type == "scatter" and x_col != "None" and y_col != "None":
                if x_col in df.columns and y_col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[x_col]) and pd.api.types.is_numeric_dtype(df[y_col]):
                        c = df[[x_col, y_col]].dropna().corr().iloc[0, 1]
                        items.append(ui.tags.li(f"Correlation between {x_col} and {y_col}: {round(float(c), 4)}"))

            elif plot_type == "box":
                target = y_col if y_col != "None" and y_col in df.columns else x_col
                if target != "None" and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
                    s = df[target].dropna()
                    q1 = s.quantile(0.25)
                    q3 = s.quantile(0.75)
                    iqr = q3 - q1
                    lower = q1 - 1.5 * iqr
                    upper = q3 + 1.5 * iqr
                    outliers = ((s < lower) | (s > upper)).sum()
                    items.append(ui.tags.li(f"IQR: {round(float(iqr), 4)}"))
                    items.append(ui.tags.li(f"Potential outliers: {int(outliers):,}"))

            elif plot_type == "bar" and x_col != "None" and x_col in df.columns:
                vc = df[x_col].astype(str).value_counts(dropna=False)
                if len(vc) > 0:
                    items.append(ui.tags.li(f"Most common category: {vc.index[0]} ({int(vc.iloc[0]):,})"))

            elif plot_type == "pie" and x_col != "None" and x_col in df.columns:
                vc = df[x_col].astype(str).value_counts(dropna=False)
                if len(vc) > 0:
                    share = vc.iloc[0] / vc.sum() * 100
                    items.append(ui.tags.li(f"Largest category share: {round(float(share), 2)}%"))

            elif plot_type == "correlation":
                corr = corr_matrix()
                if corr is not None:
                    strongest_pos, strongest_neg = strongest_correlations(corr)
                    if strongest_pos:
                        items.append(ui.tags.li(
                            f"Strongest positive correlation: {strongest_pos[0]} vs {strongest_pos[1]} = {round(float(strongest_pos[2]), 4)}"
                        ))
                    if strongest_neg:
                        items.append(ui.tags.li(
                            f"Strongest negative correlation: {strongest_neg[0]} vs {strongest_neg[1]} = {round(float(strongest_neg[2]), 4)}"
                        ))

            return ui.div(
                ui.tags.ul(*items),
                class_="info-panel"
            )

        except Exception as e:
            return ui.div(f"Could not compute insights: {str(e)}", class_="info-panel")

    @output
    @render.table
    def stat_summary():
        summary = numeric_summary()
        if summary is None or summary.empty:
            return pd.DataFrame({"Message": ["No numeric columns available for summary statistics."]})
        return summary


# ============================================================
# APP
# ============================================================
app = App(create_app_ui(), server)