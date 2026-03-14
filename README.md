# STAT5243 Project 2 - Interactive Data Analysis Web Application

## Project Overview

This project is an **interactive web application built with Python Shiny** that provides a complete workflow for data exploration and preprocessing through an intuitive web interface.

The application allows users to load datasets, clean and preprocess data, engineer features, and perform exploratory data analysis (EDA) using interactive visualizations.

### Key Features

#### Data Loading
- Upload datasets in **CSV, Excel (XLSX), or JSON formats**
- Select from built-in demo datasets included in the project

#### Data Cleaning & Preprocessing
- Handle missing values with multiple strategies
- Remove duplicate rows
- Scale numeric variables
- Encode categorical variables
- Real-time feedback on preprocessing results

#### Feature Engineering
- Apply transformations including:
  - Scaling
  - Encoding
  - Binning
  - Log transformation
  - Square root transformation
- Receive explanations and real-time logs for applied transformations
- View previews of transformed data

#### Exploratory Data Analysis (EDA)
- Interactive visualizations:
  - Histogram
  - Box Plot
  - Scatter Plot
  - Bar Chart
  - Pie Chart
  - Correlation Heatmap
- Customizable visualization settings and filtering
- Dynamic statistical summaries

#### Interactive Dashboard UI
- Responsive layout
- Real-time updates
- Interactive controls for exploring datasets efficiently

---

## Requirements

- **Python 3.8 or higher**

Required dependencies are listed in:

```
requirements.txt
```

---

## Installation

### 1. Clone or download the repository

Download the project folder or clone the repository.

### 2. Install dependencies

Run the following command inside the project directory:

```bash
pip install -r requirements.txt
```

---

## Running the Application

### Option 1: Using `shiny run` (Recommended)

```bash
shiny run app.py --reload
```

Then open your browser and go to:

```
http://127.0.0.1:8000
```

The `--reload` option automatically refreshes the application whenever code changes are saved.

---

### Option 2: Using Python directly

```bash
python app.py
```

Then open:

```
http://127.0.0.1:8000
```

---

### Option 3: Using the provided batch file (Windows)

You can run the application by double-clicking the batch file or running:

```bash
run_app.bat
```

---

## Project Structure

```
Project2/
├── app.py
├── requirements.txt
├── run_app.bat
├── data/
│   ├── fifa21 raw data v2.csv
│   ├── pokedex.csv
│   ├── grocery_chain_data.csv
│   └── financial_news_events.csv
└── README.md
```

### File Descriptions

- **app.py** – Main Python Shiny application  
- **requirements.txt** – List of required Python dependencies  
- **run_app.bat** – Windows script for launching the app  
- **data/** – Built-in datasets available in the application  
- **README.md** – Project documentation  

---

## How to Use

### 1. Load Data

1. Navigate to the **Data Loading** tab.
2. Choose one of the following options:
   - Upload your own dataset
   - Select a built-in dataset
3. Select the appropriate file format.
4. Click **Load Data**.

The application will display:

- Data preview
- Number of rows and columns
- Missing value count

---

### 2. Data Cleaning (Advanced)

Navigate to the **Data Cleaning** tab to preprocess your dataset.

#### Missing Value Handling

- Mean
- Median
- Mode
- Forward fill
- Backward fill
- Drop rows
- Drop columns

#### Duplicate Handling

- Remove duplicate rows

#### Scaling Methods

- Standard Scaling (Z-score)
- Min-Max Scaling
- Robust Scaling

#### Categorical Encoding

- Label Encoding
- One-Hot Encoding

Click **Apply Cleaning** to execute the selected preprocessing steps.

The application provides **real-time feedback**, including:

- Before vs After dataset statistics
- Missing value summaries
- Logs of applied transformations
- Updated data preview

---

### 3. Feature Engineering (Advanced)

Navigate to the **Feature Engineering** tab to create new features.

Supported transformations include:

#### Scaling

- Standard (Z-score)
- Min-Max normalization

#### Categorical Encoding

- Label encoding
- One-hot encoding

#### Binning

- Equal-width binning
- Equal-frequency binning

#### Log Transformation

Used for positively skewed variables with strictly positive values.

#### Square Root Transformation

Used for non-negative variables with moderate skewness.

The feature engineering interface provides:

- Transformation explanations
- Real-time logs of applied steps
- Visual feedback
- Preview of transformed data
- Summary of newly created features

---

### 4. Exploratory Data Analysis (EDA)

Navigate to the **EDA** tab to interactively explore the dataset.

Available visualizations include:

- Histogram
- Box Plot
- Scatter Plot
- Bar Chart
- Pie Chart
- Correlation Heatmap

Users can customize plots by selecting:

- X-axis variable
- Y-axis variable
- Color grouping
- Plot height

The EDA panel also displays **dynamic statistical summaries** for numeric variables.

---

## Team Members

- **Freya Chen** (yc4684)  
- **Zhuyun Jin** (zj2434)  
- **Nikhil Shanbhag** (nvs2128)  
- **Megan Wang** (mw3856)

---

## Deployment (Optional)

To deploy the application to **shinyapps.io**:

### 1. Install deployment CLI

```bash
pip install shinylights
```

### 2. Create a deployment manifest

```bash
shiny create-manifest
```

### 3. Deploy the application

```bash
shiny deploy
```

Alternatively, follow the official Shiny for Python deployment guide:

https://shiny.posit.co/py/docs/deploy.html

---

## License

This project was developed for **educational purposes** as part of:

**STAT5243 – Applied Statistics**  
Columbia University
