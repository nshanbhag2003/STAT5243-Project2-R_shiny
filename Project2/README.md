# STAT5243 Project 2 - Interactive Data Analysis Web Application

## Project Overview

This is an interactive web application built with **Python Shiny** that provides comprehensive data analysis capabilities:

- **Data Loading**: Upload CSV, Excel (XLSX), or JSON files; or choose from built-in demo datasets
- **Data Cleaning**: Handle missing values (imputation, drop), remove duplicates, scale numeric columns, encode categorical variables
- **Feature Engineering**: Apply transformations (scaling, encoding, binning, log, square root) with real-time explanations and feedback
- **EDA**: Interactive visualizations (histogram, box plot, scatter plot, bar chart, pie chart, correlation heatmap)

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for dependencies

## Installation

1. **Clone or download this repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### Option 1: Using `shiny run` (Recommended)

```bash
shiny run app.py --reload
```

Then open your browser at: **http://127.0.0.1:8000**

### Option 2: Using Python directly

```bash
python app.py
```

Then open your browser at: **http://127.0.0.1:8000**

### Option 3: Using the provided batch file (Windows)

Simply double-click `run_app.bat` or run it from the command line:

```bash
run_app.bat
```

## Project Structure

```
Project2/
├── app.py              # Main Shiny application
├── requirements.txt    # Python dependencies
├── run_app.bat        # Windows batch file to launch the app
├── data/              # Built-in demo datasets
│   ├── fifa21 raw data v2.csv
│   ├── pokedex.csv
│   ├── grocery_chain_data.csv
│   └── financial_news_events.csv
└── README.md          # This file
```

## How to Use

### 1. Load Data
- Go to the **Data Loading** tab
- Upload your own file (CSV, XLSX, or JSON), OR
- Select one of the built-in datasets from the dropdown
- Click "Load Data"

### 2. Clean Data (Advanced)
- Go to the **Data Cleaning** tab
- Choose a strategy for handling missing values
- Optionally check "Remove duplicate rows"
- For numeric columns: enable scaling and select method (Standard, Min-Max, Robust)
- For categorical columns: enable encoding and select method (Label or One-Hot)
- Click "Apply Cleaning"
- View real-time feedback on the right panel (Before/After stats, applied steps)

### 3. Feature Engineering (Advanced)
- Go to the **Feature Engineering** tab
- Select a transformation type:
  - **Scaling**: Standard (Z-score) or Min-Max normalization
  - **Encoding**: Label or One-Hot encoding
  - **Binning**: Equal-width or equal-frequency binning
  - **Log**: Logarithmic transformation (for positive values)
  - **Square Root**: Square root transformation (for non-negative values)
- Select columns to transform
- Click "Apply Transformation"
- See explanation, real-time feedback, and newly created features

### 4. EDA
- Go to the **EDA** tab
- Select plot type (Histogram, Box Plot, Scatter, Bar, Pie, Correlation Heatmap)
- Choose X and Y axes, and optional color grouping
- Adjust plot height
- Click "Generate Plot"

## Team Members

- Freya Chen (yc4684)
- Zhuyun Jin (zj2434)
- Nikhil Shanbhag (nvs2128)

## Deployment (Optional)

To deploy to shinyapps.io:

1. Install the shinylights CLI:
   ```bash
   pip install shinylights
   ```

2. Create a `manifest.json`:
   ```bash
   shiny create-manifest
   ```

3. Deploy:
   ```bash
   shiny deploy
   ```

Or follow the official Shiny for Python deployment guide at: https://shiny.posit.co/py/docs/deploy.html

## License

This project is for educational purposes as part of STAT5243 at Columbia University.
