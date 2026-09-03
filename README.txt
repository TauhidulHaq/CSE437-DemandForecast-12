# CSE437 Data Science Project: Bangladesh National Grid Demand Forecasting
**Group 12**

## The Problem
This project aims to forecast the daily peak electricity demand on the Bangladesh national grid (measured in Megawatts at the generation end). Additionally, it evaluates how specific supply-side constraints—such as gas shortages, coal limitations, and low hydro levels—drive power generation shortfalls and subsequent load shedding. Accurate day-ahead forecasting under constraint scenarios is critical for grid stability and resource allocation.

## The Dataset
* **Name:** Structured Dataset of Daily Electricity Demand, Generation, Load Shedding, and Supply Constraints in Bangladesh (2019–2024)
* **Source:** Mendeley Data (DOI: 10.17632/x7r7wdb39k.1), originally scraped from the Bangladesh Power Development Board (BPDB) daily generation archives.
* **Size:** 1,848 daily records and 41 features.
* **Version:** Version 1 (Raw extract). We explicitly utilize the raw dataset containing genuine missing values and structural zeros to demonstrate robust data auditing, chronological imputation, and preprocessing pipelines.

## Research Questions
1. How accurately can daily peak electricity demand be predicted using engineered calendar features (seasonality, day of the week, holidays) and historical lag variables (prior-day generation and demand)?
2. Which specific supply-side constraints (gas limitations vs. coal availability vs. Kaptai lake water levels) possess the strongest predictive power for severe load-shedding events?
3. Does electricity demand behavior differ significantly across regional divisions (e.g., Dhaka vs. Sylhet), and is a localized model necessary, or does a single national model generalize effectively to all nine divisions?

## How to Run the Project
This project is built to run entirely locally using relative paths. No Google Drive mounts or absolute local paths are required. 

**1. Clone the repository:**
```bash
git clone [https://github.com/](https://github.com/)<TauhidulHaq>/cse437-demand-forecast-12.git
cd cse437-demand-forecast-12


**2. Set up the virtual environment:**
```bash
python -m venv venv

On Windows: .\venv\Scripts\activate

On Mac/Linux: source venv/bin/activate

**3. Install dependencies:**
pip install -r requirements.txt

**4. Execute the Notebooks:**
Launch Jupyter and run the notebooks located in the notebooks/ directory sequentially from top to bottom on a fresh kernel:

01_data_audit_and_eda.ipynb: Explores the raw data and highlights missing value distributions.

02_preprocessing.ipynb: Handles chronological imputation and outlier removal, outputting to data/processed/.

03_feature_engineering.ipynb: Generates lag variables, calendar features, and regional PCA components.

04_modeling_and_tuning.ipynb: Trains and tunes the Ridge Regression and Random Forest models using rolling time-based cross-validation.

05_evaluation_and_error_analysis.ipynb: Evaluates test-set performance, generates feature importance metrics, and conducts final error analysis.

All generated charts will be saved automatically to the figures/ directory, and the trained models to the models/ directory.

