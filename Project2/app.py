"""
STAT5243 Project 2: Interactive Data Analysis Web Application
Built with Python Shiny 1.5.x

This application provides comprehensive data analysis capabilities including:
- Data Loading (CSV, Excel, JSON formats)
- Data Cleaning and Preprocessing
- Feature Engineering
- Exploratory Data Analysis (EDA)
"""

import json
import io
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import App, ui, reactive, render, req
from shiny.types import FileInfo
from shinywidgets import output_widget, render_widget
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# DATA PATHS - Built-in datasets
# ============================================================
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(APP_DIR, "数据")

BUILT_IN_DATASETS = {
    "financial_news": {
        "name": "Financial News Events",
        "files": {
            "csv": os.path.join(DATA_DIR, "financial_news_events.csv"),
            "json": os.path.join(DATA_DIR, "financial_news_events.json")
        }
    },
    "grocery": {
        "name": "Grocery Chain Data",
        "files": {
            "csv": os.path.join(DATA_DIR, "grocery_chain_data.csv"),
            "json": os.path.join(DATA_DIR, "grocery_chain_data.json")
        }
    },
    "fifa": {
        "name": "FIFA 21 Player Data",
        "files": {
            "csv": os.path.join(DATA_DIR, "fifa21 raw data v2.csv")
        }
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def load_data_from_file(file_path: str, file_type: str) -> pd.DataFrame:
    """Load data from file based on type"""
    try:
        if file_type == "csv":
            return pd.read_csv(file_path)
        elif file_type == "xlsx":
            return pd.read_excel(file_path)
        elif file_type == "json":
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
    except Exception as e:
        raise ValueError(f"Error loading file: {str(e)}")


def load_builtin_dataset(dataset_key: str, file_format: str) -> pd.DataFrame:
    """Load a built-in dataset"""
    dataset = BUILT_IN_DATASETS.get(dataset_key)
    if not dataset:
        raise ValueError(f"Unknown dataset: {dataset_key}")

    file_path = dataset["files"].get(file_format)
    if not file_path:
        raise ValueError(f"Format {file_format} not available for {dataset['name']}")

    return load_data_from_file(file_path, file_format)


# ============================================================
# UI LAYOUT
# ============================================================
def create_app_ui():
    return ui.page_fillable(
        ui.tags.head(
            ui.tags.title("Interactive Data Analysis App"),
            ui.tags.style("""
                body { font-family: 'Segoe UI', Arial, sans-serif; }
                .container-fluid { padding: 15px; }
                .card { 
                    margin-bottom: 15px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    border-radius: 8px;
                }
                .card-header {
                    background-color: #4a90d9;
                    color: white;
                    font-weight: bold;
                    border-radius: 8px 8px 0 0 !important;
                }
                .btn-primary {
                    background-color: #4a90d9;
                    border-color: #4a90d9;
                }
                .btn-primary:hover {
                    background-color: #357abd;
                    border-color: #357abd;
                }
                .nav-tabs .nav-link.active {
                    background-color: #f8f9fa;
                    border-color: #dee2e6 #dee2e6 #f8f9fa;
                    font-weight: bold;
                }
                .sidebar {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                }
                .main-content {
                    padding: 15px;
                }
                h1, h2, h3, h4 {
                    color: #333;
                }
                .feature-list li {
                    margin-bottom: 8px;
                }
                .stats-box {
                    background-color: #e9ecef;
                    padding: 10px;
                    border-radius: 5px;
                    text-align: center;
                    margin-bottom: 10px;
                }
                .stats-number {
                    font-size: 24px;
                    font-weight: bold;
                    color: #4a90d9;
                }
                table.dataTable {
                    font-size: 14px;
                }
            """)
        ),
        
        # Header
        ui.div(
            ui.h1("📊 Interactive Data Analysis App", class_="text-center my-4"),
            ui.p("A comprehensive tool for data loading, cleaning, feature engineering, and exploratory data analysis",
                 class_="text-center text-muted"),
            class_="bg-light py-3"
        ),
        
        # Main navigation tabs
        ui.navset_tab(
            # ===================== HOME TAB =====================
            ui.nav_panel(
                "🏠 Home",
                ui.row(
                    ui.column(
                        6,
                        ui.card(
                            ui.h3("Welcome!"),
                            ui.p("This application provides comprehensive data analysis capabilities for data scientists and analysts."),
                            ui.h4("Key Features:"),
                            ui.tags.ul(
                                ui.tags.li("📁 Data Loading - Support for CSV, Excel (XLSX), and JSON formats"),
                                ui.tags.li("🧹 Data Cleaning - Handle missing values, duplicates, and data transformations"),
                                ui.tags.li("⚙️ Feature Engineering - Create new features with various transformations"),
                                ui.tags.li("📈 Exploratory Data Analysis - Interactive visualizations and statistical summaries"),
                                class_="feature-list"
                            ),
                        )
                    ),
                    ui.column(
                        6,
                        ui.card(
                            ui.h3("How to Use:"),
                            ui.tags.ol(
                                ui.tags.li("Select your data source: Upload a file or use built-in datasets"),
                                ui.tags.li("Choose the file format (CSV, JSON, or XLSX)"),
                                ui.tags.li("Navigate through different tabs for analysis"),
                                ui.tags.li("Use interactive controls to customize your analysis")
                            ),
                            ui.hr(),
                            ui.h4("Built-in Datasets:"),
                            ui.tags.ul(
                                ui.tags.li("Financial News Events"),
                                ui.tags.li("Grocery Chain Data"),
                                ui.tags.li("FIFA 21 Player Data")
                            ),
                            ui.hr(),
                            ui.p(ui.tags.strong("Note:"), " This app runs entirely in your browser. No data is uploaded to any server.",
                                 class_="text-info")
                        )
                    )
                ),
                value="home"
            ),
            
            # ===================== DATA LOADING TAB =====================
            ui.nav_panel(
                "📁 Data Loading",
                ui.row(
                    ui.column(
                        4,
                        ui.card(
                            ui.h4("Select Data Source"),
                            ui.input_radio_buttons(
                                "data_source",
                                "Choose data source:",
                                {"upload": "Upload File", "builtin": "Built-in Dataset"},
                                selected="upload",
                                inline=False
                            ),
                            ui.hr(),
                            
                            # Upload section - show when "Upload File" is selected
                            ui.panel_conditional(
                                "input.data_source === 'upload'",
                                ui.div(
                                    ui.input_file("uploaded_file", "Choose file:",
                                                 accept=[".csv", ".json", ".xlsx"],
                                                 multiple=False),
                                    ui.input_select(
                                        "upload_format",
                                        "File Format:",
                                        {"csv": "CSV", "json": "JSON", "xlsx": "Excel (XLSX)"},
                                    ),
                                ),
                            ),
                            
                            # Built-in dataset section - show when "Built-in Dataset" is selected
                            ui.panel_conditional(
                                "input.data_source === 'builtin'",
                                ui.div(
                                    ui.input_select(
                                        "builtin_dataset",
                                        "Select Dataset:",
                                        {
                                            "financial_news": "Financial News Events",
                                            "grocery": "Grocery Chain Data",
                                            "fifa": "FIFA 21 Player Data"
                                        },
                                    ),
                                    ui.input_select(
                                        "builtin_format",
                                        "File Format:",
                                        {"csv": "CSV"},
                                    ),
                                ),
                            ),
                            
                            ui.hr(),
                            ui.input_action_button("load_data", "Load Data",
                                                  class_="btn-primary w-100"),
                        ),
                    ),
                    ui.column(
                        8,
                        ui.card(
                            ui.h4("Data Preview"),
                            ui.output_data_frame("data_preview"),
                            ui.hr(),
                            ui.row(
                                ui.column(4, ui.output_text("rows_count")),
                                ui.column(4, ui.output_text("cols_count")),
                                ui.column(4, ui.output_text("missing_count")),
                            ),
                        ),
                    ),
                ),
                value="loading"
            ),
            
            # ===================== DATA CLEANING TAB (Advanced: interactive + real-time feedback) =====================
            ui.nav_panel(
                "🧹 Data Cleaning",
                ui.row(
                    ui.column(
                        4,
                        ui.card(
                            ui.h4("Preprocessing Steps"),
                            ui.p("Check the steps to run, set options, then click Apply Cleaning.", class_="text-muted small"),
                            ui.hr(),
                            ui.h5("1. Missing Values"),
                            ui.input_select(
                                "missing_strategy",
                                "Strategy:",
                                {
                                    "none": "Keep as is",
                                    "mean": "Mean (numeric)",
                                    "median": "Median (numeric)",
                                    "mode": "Mode (categorical)",
                                    "drop_rows": "Drop rows with missing",
                                    "drop_cols": "Drop columns with missing",
                                    "forward_fill": "Forward fill",
                                    "backward_fill": "Backward fill"
                                }
                            ),
                            ui.hr(),
                            ui.h5("2. Duplicates"),
                            ui.input_checkbox("handle_duplicates", "Remove duplicate rows", True),
                            ui.hr(),
                            ui.h5("3. Scaling (Intermediate+)"),
                            ui.input_checkbox("enable_scaling", "Scale numeric columns", False),
                            ui.panel_conditional(
                                "input.enable_scaling === true",
                                ui.div(
                                    ui.input_select(
                                        "cleaning_scale_method",
                                        "Method:",
                                        {"standard": "Standard (Z-score)", "minmax": "Min-Max [0,1]", "robust": "Robust"}
                                    ),
                                    ui.input_select(
                                        "cleaning_scale_columns",
                                        "Select columns:",
                                        choices=[],
                                        multiple=True,
                                    ),
                                ),
                            ),
                            ui.hr(),
                            ui.h5("4. Encoding (Intermediate+)"),
                            ui.input_checkbox("enable_encoding", "Encode categorical variables", False),
                            ui.panel_conditional(
                                "input.enable_encoding === true",
                                ui.div(
                                    ui.input_select(
                                        "cleaning_encode_method",
                                        "Method:",
                                        {"label": "Label Encoding", "onehot": "One-Hot Encoding"}
                                    ),
                                    ui.input_select(
                                        "cleaning_encode_columns",
                                        "Select columns:",
                                        choices=[],
                                        multiple=True,
                                    ),
                                ),
                            ),
                            ui.hr(),
                            ui.input_action_button("apply_cleaning", "Apply Cleaning",
                                                  class_="btn-primary w-100"),
                        ),
                    ),
                    ui.column(
                        8,
                        ui.card(
                            ui.h4("Real-time Feedback"),
                            ui.row(
                                ui.column(6, ui.output_ui("cleaning_feedback_before")),
                                ui.column(6, ui.output_ui("cleaning_feedback_after")),
                            ),
                            ui.output_ui("cleaning_log"),
                            ui.hr(),
                            ui.h4("Data Preview After Cleaning"),
                            ui.output_data_frame("cleaned_data_preview"),
                            ui.hr(),
                            ui.row(
                                ui.column(6, ui.output_text("cleaned_rows")),
                                ui.column(6, ui.output_text("cleaned_cols")),
                            ),
                            ui.hr(),
                            ui.h5("Missing Values Summary"),
                            ui.output_table("missing_summary"),
                        ),
                    ),
                ),
                value="cleaning"
            ),
            
            # ===================== FEATURE ENGINEERING TAB =====================
            ui.nav_panel(
                "⚙️ Feature Engineering",
                ui.row(
                    ui.column(
                        4,
                        ui.card(
                            ui.h4("Feature Engineering Options"),
                            ui.input_select(
                                "feature_type",
                                "Transformation Type:",
                                {
                                    "scale": "Scaling (Normalize/MinMax)",
                                    "encode": "Categorical Encoding (Label/One-Hot)",
                                    "bin": "Binning (Create Categories)",
                                    "log": "Log Transform",
                                    "sqrt": "Square Root Transform"
                                }
                            ),
                            ui.hr(),
                            
                            # Scale / Log / Sqrt: show column selector; scaling_method only for scale
                            ui.panel_conditional(
                                "input.feature_type === 'scale' || input.feature_type === 'log' || input.feature_type === 'sqrt'",
                                ui.div(
                                    ui.panel_conditional(
                                        "input.feature_type === 'scale'",
                                        ui.input_select(
                                            "scaling_method",
                                            "Scaling Method:",
                                            {
                                                "standard": "Standard Scaler (Z-score)",
                                                "minmax": "Min-Max Scaling"
                                            }
                                        ),
                                    ),
                                    ui.input_select(
                                        "scale_columns",
                                        "Select Columns:",
                                        choices=[],
                                        multiple=True,
                                    ),
                                ),
                            ),
                            
                            # Encoding options - show when "encode" is selected
                            ui.panel_conditional(
                                "input.feature_type === 'encode'",
                                ui.div(
                                    ui.input_select(
                                        "encoding_method",
                                        "Encoding Method:",
                                        {
                                            "label": "Label Encoding",
                                            "onehot": "One-Hot Encoding"
                                        }
                                    ),
                                    ui.input_select(
                                        "encode_columns",
                                        "Select Columns to Encode:",
                                        choices=[],
                                        multiple=True,
                                    ),
                                ),
                            ),
                            
                            # Binning options - show when "bin" is selected
                            ui.panel_conditional(
                                "input.feature_type === 'bin'",
                                ui.div(
                                    ui.input_select(
                                        "bin_column",
                                        "Select Column to Bin:",
                                        choices=[],
                                    ),
                                    ui.input_slider("bin_count", "Number of Bins:", 2, 10, 5),
                                    ui.input_select(
                                        "bin_method",
                                        "Binning Method:",
                                        {"equal": "Equal Width", "quantile": "Equal Frequency"}
                                    ),
                                ),
                            ),
                            
                            ui.hr(),
                            ui.input_action_button("apply_feature", "Apply Transformation",
                                                  class_="btn-primary w-100"),
                        ),
                    ),
                    ui.column(
                        8,
                        ui.card(
                            ui.h4("Transformation Explanation"),
                            ui.output_ui("feature_type_explanation"),
                        ),
                        ui.card(
                            ui.h4("Real-time Feedback"),
                            ui.output_ui("feature_eng_feedback"),
                            ui.output_ui("feature_eng_log_ui"),
                        ),
                        ui.card(
                            ui.h4("Transformed Data Preview"),
                            ui.output_data_frame("transformed_data_preview"),
                            ui.hr(),
                            ui.h5("New Features Created"),
                            ui.output_table("feature_summary"),
                        ),
                    ),
                ),
                value="feature_eng"
            ),
            
            # ===================== EDA TAB =====================
            ui.nav_panel(
                "📈 EDA",
                ui.row(
                    ui.column(
                        3,
                        ui.card(
                            ui.h4("Visualization Settings"),
                            ui.input_select(
                                "plot_type",
                                "Plot Type:",
                                {
                                    "histogram": "Histogram",
                                    "box": "Box Plot",
                                    "scatter": "Scatter Plot",
                                    "bar": "Bar Chart",
                                    "pie": "Pie Chart",
                                    "correlation": "Correlation Heatmap"
                                }
                            ),
                            ui.input_select(
                                "x_axis",
                                "X-Axis:",
                                choices=["None"],
                            ),
                            ui.input_select(
                                "y_axis",
                                "Y-Axis:",
                                choices=["None"],
                            ),
                            ui.input_select(
                                "color_by",
                                "Color By:",
                                choices=["None"],
                            ),
                            ui.input_slider(
                                "plot_height",
                                "Plot Height:",
                                300, 800, 500
                            ),
                            ui.input_action_button("generate_plot", "Generate Plot",
                                                  class_="btn-primary w-100"),
                        ),
                    ),
                    ui.column(
                        9,
                        ui.card(
                            ui.h4("Interactive Visualization"),
                            output_widget("eda_plot"),
                        ),
                        ui.card(
                            ui.h4("Statistical Summary"),
                            ui.output_table("stat_summary"),
                        ),
                    ),
                ),
                value="eda"
            ),
            
            # ===================== ABOUT TAB =====================
            ui.nav_panel(
                "ℹ️ About",
                ui.card(
                    ui.h2("Project 2: Web Application Development"),
                    ui.p("This application was developed for STAT5243 - Applied Statistics course."),
                    ui.hr(),
                    ui.h4("Technology Stack:"),
                    ui.tags.ul(
                        ui.tags.li("Frontend: Shiny for Python"),
                        ui.tags.li("Data Processing: Pandas, NumPy"),
                        ui.tags.li("Visualization: Plotly"),
                        ui.tags.li("Statistics: Scikit-learn"),
                    ),
                    ui.hr(),
                    ui.h4("Course Information:"),
                    ui.p("STAT5243 - Applied Statistics"),
                    ui.p("Columbia University"),
                    ui.hr(),
                    ui.p(ui.tags.strong("Team Members: "), "Freya Chen (yc4684), Zhuyun Jin (zj2434), Nikhil Shanbhag (nvs2128)"),
                ),
                value="about"
            ),
        ),
        
        # Footer
        ui.div(
            ui.p("STAT5243 Project 2 - Interactive Data Analysis App", class_="text-center text-muted mt-4"),
            class_="bg-light py-2 mt-auto"
        )
    )


# ============================================================
# SERVER LOGIC
# ============================================================
def server(input, output, session):
    # Reactive values to store data
    raw_data = reactive.Value(None)
    processed_data = reactive.Value(None)
    feature_data = reactive.Value(None)
    
    # Data cleaning: real-time feedback (Advanced rubric)
    applied_cleaning_log = reactive.Value([])
    before_cleaning_stats = reactive.Value(None)  # {"rows", "cols", "missing"}

    # Feature engineering: real-time feedback + visual feedback (Advanced rubric)
    feature_eng_log = reactive.Value([])
    
    # Store column lists for select inputs
    column_lists = reactive.Value({"numeric": [], "categorical": [], "all": []})
    
    # ============================================================
    # DATA LOADING TAB
    # ============================================================
    @reactive.Effect
    def update_builtin_format():
        """Update available formats based on selected built-in dataset"""
        dataset = BUILT_IN_DATASETS.get(input.builtin_dataset())
        if dataset:
            formats = list(dataset["files"].keys())
            ui.update_select("builtin_format", choices=formats)
    
    @reactive.Effect
    @reactive.event(input.load_data)
    def load_data():
        """Load data based on user selection"""
        try:
            if input.data_source() == "upload":
                file_info = input.uploaded_file()
                if file_info is None:
                    ui.notification_show("Please upload a file first!", type="warning")
                    return
                
                file = file_info[0]
                file_type = input.upload_format()
                # Shiny stores uploaded file at a temp path; use "datapath" (not "content")
                path = file.get("datapath") or file.get("path")
                if not path or not os.path.isfile(path):
                    ui.notification_show("Could not read uploaded file.", type="error")
                    return

                # Read file from path
                if file_type == "csv":
                    df = pd.read_csv(path, encoding="utf-8")
                elif file_type == "json":
                    df = pd.read_json(path)
                elif file_type == "xlsx":
                    df = pd.read_excel(path)
                else:
                    ui.notification_show(f"Unsupported format: {file_type}", type="error")
                    return
            else:
                # Load built-in dataset
                df = load_builtin_dataset(input.builtin_dataset(), input.builtin_format())
            
            raw_data.set(df)
            processed_data.set(df.copy())
            feature_data.set(df.copy())
            feature_eng_log.set([])  # reset feature engineering log on new data
            
            # Update column lists
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            all_cols = df.columns.tolist()
            
            column_lists.set({
                "numeric": numeric_cols,
                "categorical": categorical_cols,
                "all": all_cols
            })
            
            # Update select inputs for other tabs
            ui.update_select("scale_columns", choices=numeric_cols)
            ui.update_select("encode_columns", choices=categorical_cols)
            ui.update_select("bin_column", choices=numeric_cols)
            ui.update_select("cleaning_scale_columns", choices=numeric_cols)
            ui.update_select("cleaning_encode_columns", choices=categorical_cols)
            ui.update_select("x_axis", choices=["None"] + all_cols)
            ui.update_select("y_axis", choices=["None"] + all_cols)
            ui.update_select("color_by", choices=["None"] + categorical_cols)
            
            ui.notification_show(f"Data loaded successfully! {df.shape[0]} rows, {df.shape[1]} columns", type="message")
            
        except Exception as e:
            ui.notification_show(f"Error loading data: {str(e)}", type="error")
    
    @output
    @render.data_frame
    def data_preview():
        """Display data preview"""
        df = raw_data()
        if df is None:
            return pd.DataFrame()
        return render.DataGrid(df.head(100), filters=True)
    
    @output
    @render.text
    def rows_count():
        df = raw_data()
        if df is None:
            return ""
        return f"Rows: {df.shape[0]:,}"
    
    @output
    @render.text
    def cols_count():
        df = raw_data()
        if df is None:
            return ""
        return f"Columns: {df.shape[1]}"
    
    @output
    @render.text
    def missing_count():
        df = raw_data()
        if df is None:
            return ""
        total_missing = df.isnull().sum().sum()
        return f"Missing Values: {total_missing:,}"
    
    # ============================================================
    # DATA CLEANING TAB (Advanced: dynamic steps + real-time feedback)
    # ============================================================
    @reactive.Effect
    @reactive.event(input.apply_cleaning)
    def apply_cleaning():
        """Apply selected preprocessing steps with real-time feedback"""
        df = processed_data()
        if df is None:
            ui.notification_show("Please load data first!", type="warning")
            return

        try:
            # Capture state BEFORE for real-time feedback
            before_cleaning_stats.set({
                "rows": len(df),
                "cols": len(df.columns),
                "missing": int(df.isnull().sum().sum())
            })
            log = []

            # 1. Missing values
            if input.missing_strategy() != "none":
                strategy = input.missing_strategy()
                miss_before = df.isnull().sum().sum()
                if strategy == "drop_rows":
                    df = df.dropna()
                    log.append(f"Missing: dropped rows with missing, {len(df)} rows left")
                elif strategy == "drop_cols":
                    df = df.dropna(axis=1)
                    log.append(f"Missing: dropped columns with missing, {len(df.columns)} cols left")
                elif strategy == "mean":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                    log.append(f"Missing: mean imputation ({len(numeric_cols)} numeric cols)")
                elif strategy == "median":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                    log.append(f"Missing: median imputation ({len(numeric_cols)} numeric cols)")
                elif strategy == "mode":
                    for col in df.columns:
                        if df[col].isnull().any():
                            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else df[col].iloc[0])
                    log.append("Missing: mode imputation (categorical)")
                elif strategy == "forward_fill":
                    df = df.ffill()
                    log.append("Missing: forward fill")
                elif strategy == "backward_fill":
                    df = df.bfill()
                    log.append("Missing: backward fill")
                miss_after = df.isnull().sum().sum()
                if miss_before > 0 and miss_after == 0:
                    log[-1] += f" ({miss_before} values handled)"

            # 2. Duplicates
            if input.handle_duplicates():
                initial_rows = len(df)
                df = df.drop_duplicates()
                dropped = initial_rows - len(df)
                if dropped > 0:
                    log.append(f"Duplicates: removed {dropped} rows, {len(df)} rows left")
                else:
                    log.append("Duplicates: no duplicate rows")

            # 3. Scaling: only columns that exist in current df
            if input.enable_scaling():
                requested = list(input.cleaning_scale_columns()) if input.cleaning_scale_columns() else []
                numeric_in_df = df.select_dtypes(include=[np.number]).columns.tolist()
                cols = [c for c in requested if c in df.columns and c in numeric_in_df]
                if cols:
                    method = input.cleaning_scale_method()
                    if method == "standard":
                        scaler = StandardScaler()
                    elif method == "minmax":
                        scaler = MinMaxScaler()
                    else:
                        scaler = RobustScaler()
                    df[cols] = scaler.fit_transform(df[cols])
                    log.append(f"Scaling: {len(cols)} cols ({method})")
                else:
                    log.append("Scaling: no valid numeric columns selected, skipped")

            # 4. Encoding: only columns that exist in current df
            if input.enable_encoding():
                requested = list(input.cleaning_encode_columns()) if input.cleaning_encode_columns() else []
                cols = [c for c in requested if c in df.columns]
                if cols:
                    method = input.cleaning_encode_method()
                    if method == "label":
                        for col in cols:
                            df[col + "_encoded"] = LabelEncoder().fit_transform(df[col].astype(str))
                        log.append(f"Encoding: {len(cols)} cols Label")
                    else:
                        df = pd.get_dummies(df, columns=cols, prefix=cols)
                        log.append(f"Encoding: {len(cols)} cols One-Hot")
                else:
                    log.append("Encoding: no valid columns selected, skipped")

            processed_data.set(df)
            feature_data.set(df.copy())
            applied_cleaning_log.set(log)
            ui.notification_show("Cleaning applied. See feedback on the right.", type="message")

        except Exception as e:
            ui.notification_show(f"Error during cleaning: {str(e)}", type="error")
            applied_cleaning_log.set([f"Error: {str(e)}"])

    @output
    @render.ui
    def cleaning_feedback_before():
        """Before-cleaning stats for real-time feedback"""
        stats = before_cleaning_stats()
        if stats is None:
            return ui.div(ui.h6("Before"), ui.p("Shown after applying cleaning."), class_="text-muted")
        return ui.div(
            ui.h6("Before"),
            ui.p(ui.strong("Rows: "), f"{stats['rows']:,}"),
            ui.p(ui.strong("Columns: "), str(stats['cols'])),
            ui.p(ui.strong("Missing: "), f"{stats['missing']:,}"),
            class_="border rounded p-2 bg-light"
        )

    @output
    @render.ui
    def cleaning_feedback_after():
        """After-cleaning stats (real-time)"""
        df = processed_data()
        if df is None:
            return ui.div(ui.h6("After"), ui.p("—"), class_="text-muted")
        return ui.div(
            ui.h6("After"),
            ui.p(ui.strong("Rows: "), f"{df.shape[0]:,}"),
            ui.p(ui.strong("Columns: "), str(df.shape[1])),
            ui.p(ui.strong("Missing: "), f"{df.isnull().sum().sum():,}"),
            class_="border rounded p-2 bg-success bg-opacity-10"
        )

    @output
    @render.ui
    def cleaning_log():
        """Log of applied steps (real-time feedback)"""
        log = applied_cleaning_log()
        if not log:
            return ui.div(ui.p("Applied steps will appear here after you click Apply Cleaning.", class_="text-muted small"))
        return ui.div(
            ui.h6("Applied steps:"),
            ui.tags.ul(*[ui.tags.li(line) for line in log]),
            class_="border rounded p-2 small"
        )

    @output
    @render.data_frame
    def cleaned_data_preview():
        """Display cleaned data preview"""
        df = processed_data()
        if df is None:
            return pd.DataFrame()
        return render.DataGrid(df.head(100), filters=True)
    
    @output
    @render.text
    def cleaned_rows():
        df = processed_data()
        if df is None:
            return ""
        return f"Rows: {df.shape[0]:,}"
    
    @output
    @render.text
    def cleaned_cols():
        df = processed_data()
        if df is None:
            return ""
        return f"Columns: {df.shape[1]}"
    
    @output
    @render.table
    def missing_summary():
        """Display missing values summary"""
        df = processed_data()
        if df is None:
            return pd.DataFrame()
        
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        
        if len(missing) == 0:
            return pd.DataFrame({"Message": ["No missing values!"]})
        
        return pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing.values / len(df) * 100).round(2)
        })
    
    # ============================================================
    # FEATURE ENGINEERING TAB (Advanced: explanations + real-time & visual feedback)
    # ============================================================
    @output
    @render.ui
    def feature_type_explanation():
        """Explanation of selected transformation (Intermediate: with explanations)"""
        t = input.feature_type()
        texts = {
            "scale": "Scaling: Standard (Z-score) gives mean 0, variance 1; Min-Max scales to [0,1]. Use when features have different units.",
            "encode": "Encoding: Label assigns integer labels; One-Hot creates one 0/1 column per category. Use for categorical predictors.",
            "bin": "Binning: Discretize continuous values. Equal Width uses fixed interval size; Equal Frequency (quantile) balances counts per bin.",
            "log": "Log transform: log(x). Use for right-skewed distributions; reduces impact of outliers. Only for positive values.",
            "sqrt": "Square root: √x. Softer than log; use for non-negative, mildly right-skewed data.",
        }
        return ui.div(
            ui.p(texts.get(t, "Select a transformation type above to see explanation."), class_="mb-0 small text-muted")
        )

    @output
    @render.ui
    def feature_eng_feedback():
        """Real-time stats: current columns and new feature count (visual feedback)"""
        df = feature_data()
        base_df = processed_data()
        if df is None:
            return ui.div(ui.p("Load data first.", class_="text-muted small"))
        base_cols = set(base_df.columns) if base_df is not None else set()
        new_cols = set(df.columns) - base_cols
        return ui.div(
            ui.row(
                ui.column(6, ui.div(ui.strong("Total columns: "), str(len(df.columns)), class_="border rounded p-2")),
                ui.column(6, ui.div(ui.strong("New features (this session): "), str(len(new_cols)), class_="border rounded p-2 bg-light")),
            ),
            class_="small"
        )

    @output
    @render.ui
    def feature_eng_log_ui():
        """Log of applied feature engineering steps (real-time updates)"""
        log = feature_eng_log()
        if not log:
            return ui.div(ui.p("Applied steps will appear here after you apply a transformation.", class_="text-muted small"))
        return ui.div(
            ui.h6("Applied steps:"),
            ui.tags.ul(*[ui.tags.li(line) for line in log], class_="small"),
            class_="border rounded p-2"
        )

    @reactive.Effect
    @reactive.event(input.apply_feature)
    def apply_feature_engineering():
        """Apply feature engineering with real-time log and column existence check"""
        df = feature_data()
        if df is None:
            ui.notification_show("Please load data first!", type="warning")
            return

        try:
            feature_type = input.feature_type()
            log = list(feature_eng_log())

            if feature_type == "scale":
                requested = list(input.scale_columns()) if input.scale_columns() else []
                columns = [c for c in requested if c in df.columns and c in df.select_dtypes(include=[np.number]).columns]
                if not columns:
                    ui.notification_show("Please select numeric columns that exist in the current data.", type="warning")
                    return
                method = input.scaling_method()
                scaler = StandardScaler() if method == "standard" else MinMaxScaler()
                df[columns] = scaler.fit_transform(df[columns])
                log.append(f"Scaling: {len(columns)} cols ({method})")
                ui.notification_show(f"Scaled {len(columns)} columns ({method})", type="message")

            elif feature_type == "encode":
                requested = list(input.encode_columns()) if input.encode_columns() else []
                columns = [c for c in requested if c in df.columns]
                if not columns:
                    ui.notification_show("Please select columns that exist in the current data.", type="warning")
                    return
                method = input.encoding_method()
                if method == "label":
                    for col in columns:
                        df[col + "_encoded"] = LabelEncoder().fit_transform(df[col].astype(str))
                    log.append(f"Encoding: {len(columns)} cols Label")
                else:
                    df = pd.get_dummies(df, columns=columns, prefix=columns)
                    log.append(f"Encoding: {len(columns)} cols One-Hot")
                ui.notification_show(f"Encoded {len(columns)} columns ({method})", type="message")

            elif feature_type == "bin":
                col = input.bin_column()
                if not col or col not in df.columns:
                    ui.notification_show("Please select numeric columns that exist in the current data.", type="warning")
                    return
                n_bins = input.bin_count()
                method = input.bin_method()
                if method == "equal":
                    df[col + "_binned"] = pd.cut(df[col], bins=n_bins, duplicates='drop')
                else:
                    df[col + "_binned"] = pd.qcut(df[col], q=n_bins, duplicates='drop')
                log.append(f"Binning: {col} → {n_bins} bins ({method})")
                ui.notification_show(f"Binned {col}", type="message")

            elif feature_type == "log":
                requested = list(input.scale_columns()) if input.scale_columns() else []
                columns = [c for c in requested if c in df.columns and c in df.select_dtypes(include=[np.number]).columns]
                if not columns:
                    ui.notification_show("Please select numeric columns.", type="warning")
                    return
                done = []
                for col in columns:
                    if (df[col] <= 0).any():
                        ui.notification_show(f"Skipped {col}: contains non-positive values", type="warning")
                        continue
                    df[col + "_log"] = np.log(df[col])
                    done.append(col)
                if done:
                    log.append(f"Log transform: {len(done)} cols")
                ui.notification_show("Log transform applied!", type="message")

            elif feature_type == "sqrt":
                requested = list(input.scale_columns()) if input.scale_columns() else []
                columns = [c for c in requested if c in df.columns and c in df.select_dtypes(include=[np.number]).columns]
                if not columns:
                    ui.notification_show("Please select numeric columns.", type="warning")
                    return
                done = []
                for col in columns:
                    if (df[col] < 0).any():
                        ui.notification_show(f"Skipped {col}: contains negative values", type="warning")
                        continue
                    df[col + "_sqrt"] = np.sqrt(df[col])
                    done.append(col)
                if done:
                    log.append(f"Sqrt transform: {len(done)} cols")
                ui.notification_show("Sqrt transform applied!", type="message")

            feature_data.set(df)
            feature_eng_log.set(log)
        except Exception as e:
            ui.notification_show(f"Error in feature engineering: {str(e)}", type="error")
    
    @output
    @render.data_frame
    def transformed_data_preview():
        """Display transformed data preview"""
        df = feature_data()
        if df is None:
            return pd.DataFrame()
        return render.DataGrid(df.head(100), filters=True)
    
    @output
    @render.table
    def feature_summary():
        """Display summary of new features"""
        df = feature_data()
        original_df = processed_data()
        
        if df is None or original_df is None:
            return pd.DataFrame()
        
        new_cols = set(df.columns) - set(original_df.columns)
        
        if not new_cols:
            return pd.DataFrame({"Message": ["No new features created yet."]})
        
        new_df = df[list(new_cols)]
        summary = pd.DataFrame({
            "New Feature": list(new_cols),
            "Data Type": [str(new_df[col].dtype) for col in new_cols],
            "Non-Null Count": [new_df[col].notna().sum() for col in new_cols]
        })
        
        return summary
    
    # ============================================================
    # EDA TAB
    # ============================================================
    @output
    @render_widget
    def eda_plot():
        """Generate EDA plot (Plotly) - use render_widget for Plotly figures"""
        df = feature_data()
        if df is None:
            fig = go.Figure()
            fig.add_annotation(text="Load data first", x=0.5, y=0.5, showarrow=False, font=dict(size=16))
            fig.update_layout(template="plotly_white", height=400)
            return fig

        plot_type = input.plot_type()
        x_col = input.x_axis()
        y_col = input.y_axis()

        if plot_type in ["scatter", "bar", "pie"] and x_col == "None":
            ui.notification_show("Please select X-axis column!", type="warning")
            fig = go.Figure()
            fig.add_annotation(text="Select X-axis", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
            fig.update_layout(template="plotly_white", height=400)
            return fig

        if plot_type == "scatter" and y_col == "None":
            ui.notification_show("Please select Y-axis column for scatter plot!", type="warning")
            fig = go.Figure()
            fig.add_annotation(text="Select Y-axis for scatter plot", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
            fig.update_layout(template="plotly_white", height=400)
            return fig

        height = input.plot_height()

        try:
            color_col = input.color_by()
            if color_col == "None":
                color_col = None

            if plot_type == "histogram":
                if x_col == "None":
                    fig = go.Figure()
                    fig.add_annotation(text="Select X-axis", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
                    fig.update_layout(template="plotly_white", height=400)
                    return fig
                # If Y selected: show sum of Y by X; else show count
                if y_col and y_col != "None":
                    fig = px.histogram(df, x=x_col, y=y_col, color=color_col,
                                       histfunc="sum", title=f"{y_col} by {x_col}", height=height)
                else:
                    fig = px.histogram(df, x=x_col, color=color_col,
                                       title=f"Distribution of {x_col} (count)", height=height)

            elif plot_type == "box":
                if x_col == "None":
                    fig = go.Figure()
                    fig.add_annotation(text="Select X-axis", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
                    fig.update_layout(template="plotly_white", height=400)
                    return fig
                fig = px.box(df, x=x_col, y=y_col if y_col != "None" else None,
                            color=color_col,
                            title=f"Box Plot of {x_col}", height=height)

            elif plot_type == "scatter":
                fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                                title=f"Scatter Plot: {x_col} vs {y_col}", height=height)

            elif plot_type == "bar":
                fig = px.bar(df, x=x_col, y=y_col if y_col != "None" else None,
                            color=color_col,
                            title=f"Bar Chart of {x_col}", height=height)

            elif plot_type == "pie":
                fig = px.pie(df, names=x_col, values=y_col if y_col != "None" else None,
                            title=f"Pie Chart of {x_col}", height=height)

            elif plot_type == "correlation":
                numeric_df = df.select_dtypes(include=[np.number])
                if numeric_df.shape[1] < 2:
                    ui.notification_show("Need at least 2 numeric columns for correlation!", type="warning")
                    fig = go.Figure()
                    fig.add_annotation(text="Need at least 2 numeric columns", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
                    fig.update_layout(template="plotly_white", height=400)
                    return fig
                corr = numeric_df.corr()
                fig = px.imshow(corr, title="Correlation Heatmap", height=height,
                               color_continuous_scale="RdBu_r")

            fig.update_layout(template="plotly_white")
            return fig

        except Exception as e:
            ui.notification_show(f"Error generating plot: {str(e)}", type="error")
            fig = go.Figure()
            fig.add_annotation(text=f"Error: {str(e)}", x=0.5, y=0.5, showarrow=False, font=dict(size=12))
            fig.update_layout(template="plotly_white", height=400)
            return fig
    
    @output
    @render.table
    def stat_summary():
        """Display statistical summary"""
        df = feature_data()
        if df is None:
            return pd.DataFrame()
        
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            return pd.DataFrame({"Message": ["No numeric columns available for statistics."]})
        
        return numeric_df.describe().T


# ============================================================
# APP INITIALIZATION
# ============================================================
app = App(create_app_ui(), server)
