# Grid Demand Forecasting

**Course:** CSE437
**Section:** 6
**Semester:** Fall 2026

**Group Members:**
- Tauhidul Haq (23301396)
- Shihab-un Sakib Khan (23301395)

**Github:** https://github.com/TauhidulHaq/CSE437-DemandForecast-12

**Date:** 3rd September, 2026

---

## Summary

This project is using the BPDB daily generation dataset (1,848 datapoints) to basically predict the daily national evening electricity demand. Predicting the peak demand is essential for stable supply from the grids and allocating resources efficiently.

We implemented rolling time-based validation using scikit-learn's TimeSeriesSplit to strictly prevent data leakage (as was asked for revision in the spreadsheet).

Our target variable for this project is, Max. Demand at eve. peak (Generation end). We compared two distinct model families: a normal linear model (Ridge Regression) and a tree model (Random Forest).

Our model measured performance using Mean Absolute Error (MAE) as the main metric. The single most important finding in our project was the fact that when using the Ridge Regression Model we get a test MAE of 209.86 MW (Megawatt), which is drastically better than when the Random Forest model was run with an MAE of 301.34 MW. This occurred because the linear model was successful in extrapolating long-term capacity growth trends, whereas the tree ensemble could not gauge out-of-distribution peaks. Furthermore, our model analysis confirmed that gas supply limitations remain the most significant predictor of major load-shedding events.

---

## 1. Problem and Dataset

### 1.1 Problem statement

This project predicts the daily evening peak electricity demand on the Bangladesh national grid and explains how because of some supply-side constraints generation is halted and load shedding occurs. Accurate demand forecasting is important for national energy infrastructure management. By reliably anticipating peak demands, grid operators can efficiently allocate resources, schedule preventative maintenance without triggering unannounced blackouts, and efficiently balance the grid's reliance on different fuel types (gas, coal, and hydro).

### 1.2 Dataset

- **Source:** Mendeley Data (Working Link: https://data.mendeley.com/datasets/x7r7wdb39k).
- **Collection method:** Downloaded an Excel spreadsheet published by IUT.
- **Size:** The raw dataset contains 1,848 rows and 41 columns.
- **Time period covered:** Daily records spanning from 2019 to 2024.
- **Licence / Terms of use:** The dataset is distributed under an open-access Creative Commons license via the Mendeley Data repository.

### 1.3 Target variable

- **Name:** Max. Demand at eve. peak (Generation end).
- **Type:** Continuous.
- **Distribution:** As visualized in the project repository's figures, the target variable exhibits a slightly right-skewed normal distribution, representing standard operating demand with a long tail of extreme peak events occurring during severe summer heatwaves. Below is an image to represent this distribution:

![Distribution of Max Demand at Evening Peak](images/img-003-000.png)

### 1.4 Three questions

- How accurately can daily peak electricity demand be predicted using engineered calendar features (seasonality, day of the week, holidays) and historical lag variables (prior-day generation and demand)?
- Which specific supply-side constraints (gas limitations vs. coal availability vs. Kaptai lake water levels) possess the strongest predictive power for severe load-shedding events?
- Does electricity demand behavior differ significantly across regional divisions (e.g., Dhaka vs. Sylhet), and is a localized model necessary, or does a single national model generalize effectively to all nine divisions?

---

## 2. Data Handling and Preprocessing

### 2.1 Data Quality Audit

An initial audit of the raw dataset (1,848 rows, 41 columns) revealed a phantom blank row at the end of the file, causing a single missing value across almost all columns. Beyond this, regional data for the Khulna division contained 60 missing records across its demand, supply, and load features. Maximum temperature data for Dhaka was missing in 5 instances. A zero-value audit on load-shedding columns confirmed that zeros (e.g., 1,485 in Dhaka, 1,627 in Khulna) represent true operational states (no load shedding), confirming that the NaN entries are genuinely missing records, not unrecorded zeros. No duplicate rows or impossible categorical values were found.

![Dataset shape and DataFrame info](images/img-004-002.png)

![Zero vs NaN audit code and output](images/img-004-001.png)

### 2.2 Missing values

- **Phantom Row:** The single blank row was dropped entirely because the target variable was missing. See the picture above for the code.
- **Load Shedding Features:** Assumed missing due to localized logging failures. Strategy: Forward-fill (`ffill()`) to carry forward the last known operational state without introducing lookahead bias.
- **Continuous Variables (Demand, Supply, Temperature):** Assumed missing at random. Our strategy was to use time-based linear interpolation to estimate missing values chronologically using surrounding data points.

### 2.3 Outliers

- **Detection Method:** The Interquartile Range (IQR) method was applied to the target variable (Max. Demand at eve. peak). Bounds were set at Q1 - 1.5 times IQR and Q3 + 1.5 times IQR.
- **Action:** The flagged extreme points were dropped from the dataset to ensure models train on stable, normal operating conditions rather than extreme, rare grid failure events.

![Outlier detection and removal code](images/img-005-003.png)

### 2.4 Transformation and scaling

- **Transformations:** The "Actual data of" column was cast to a DateTime object and set as the dataframe index to enforce strict chronological ordering. See below for the image of the code.

![Dataset setup and datetime indexing code](images/img-006-004.png)

- **Scaling:** "StandardScaler" was chosen for the regularized linear model (Ridge).
- **Leakage Prevention:** To explicitly guard against data leakage, scaling is executed solely within a scikit-learn Pipeline. The scaler is fitted exclusively on the training split during each fold of the time-series cross-validation. Tree-based models process the unscaled features.

![Imports for modeling pipeline](images/img-006-005.png)

### 2.5 Before and after

![Outlier removal summary](images/img-007-006.png)

![Preprocessing summary](images/img-007-007.png)

| Stage | Rows | Columns |
|---|---|---|
| Before cleaning | 1,848 | 41 |
| After dropping phantom row / outlier removal | 1,816 | 41 |

(31 rows were flagged and removed as outliers during Section 2.3.)

---

## 3. Statistical Analysis

### 3.1 Descriptive statistics

The dataset consists primarily of continuous numerical features spanning demand, generation, and operational constraints. Rather than a raw statistical dump, the following analysis isolates the central tendency, spread, and shape of the most critical variables driving the predictive modeling. See the table below.

| | Mean | Median | Std Dev | Min | Max |
|---|---|---|---|---|---|
| Max. Demand at eve. peak (Generation end) | 11631.7 | 11589.5 | 2113.77 | 5652 | 17200 |
| Maximum Temperature in Dhaka was | 31.52 | 32.5 | 4.07 | 15.4 | 40.6 |
| Gas/LF limitation | 2382.75 | 2244.5 | 1097.87 | 267 | 5854 |
| Dhaka_load | 26.01 | 0 | 91.33 | 0 | 2002 |

### 3.2 Relationships

To understand the drivers of the target variable, we examined correlations and cross-tabulations among the numeric features.

- **Argument 1: Weather is a primary driver of base demand.** There is a strong positive correlation between the Maximum Temperature in Dhaka and the National Maximum Demand. As temperature rises, demand scales synchronously, likely due to the concurrent activation of air conditioning and cooling units across urban centers.

![Argument 1: Temperature drives peak demand](images/img-008-009.png)

- **Argument 2: Gas constraints dictate load shedding severity.** When cross-referencing national supply constraints against regional load shedding, the Gas/LF limitation shows the strongest alignment with actual power shortfalls. While coal and Kaptai lake levels fluctuate, sudden spikes in gas limitations frequently mirror spikes in regional load shedding variables, establishing gas as the grid's primary bottleneck.

![Argument 2: Gas constraints mirror load shedding severity](images/img-008-010.png)

### 3.3 What the data says so far

The exploratory data analysis yielded several critical observations:

- **Strong Autocorrelation:** The target variable changes relatively smoothly day-to-day. This confirms that engineered lag variables (e.g., historical demand from prior days) will provide significant predictive power.
- **Severe Multicollinearity in Regional Data:** Division-level demand features are almost perfectly correlated with one another and with the national peak. Feeding all raw regional features into a linear model will cause instability, justifying the need for dimensionality reduction (PCA).
- **Zero-Inflated Target Spaces:** Because load shedding is exactly zero on most days, utilizing it directly as a continuous predictor for linear models is problematic.
- **Gas is the Dominant Constraint:** Among the operational constraints, gas limitations dictate the grid's ceiling much more rigidly than coal or hydro, meaning this feature must be prioritized in the final dataset.

| Correlation with Target | Value |
|---|---|
| Max. Demand at eve. peak (Generation end) | 1.000 |
| Maximum Temperature in Dhaka was | 0.681 |
| Dhaka_load | 0.356 |
| Gas/LF limitation | -0.007 |

---

## 4. Feature Engineering

### 4.1 Derived features

To capture the temporal dynamics of the power grid, several features were derived directly from the date index:

- **Calendar Features:** Day of the week, Month, and a binary 'IsWeekend' flag (specifically marking Friday and Saturday). These were constructed to capture established behavioral and industrial cycles in electricity consumption.
- **Lag Variables:** Historical demand from 1 day, 2 days, and 7 days prior. These address the strong autocorrelation inherent in day-to-day grid operations, allowing the models to anchor their predictions on recent actual values.
- **Rolling Averages:** A 7-day rolling mean of the target variable. This feature was explicitly shifted by one day prior to calculation to prevent data leakage, smoothing out daily volatility and providing the models with a stable short-term baseline trend.

### 4.2 Dimensionality reduction

Principal Component Analysis (PCA) was applied to the nine regional demand columns (e.g., Dhaka_demand, Khulna_demand). Because regional power demands are highly collinear and scale synchronously with the national peak, feeding them individually risks a mathematical instability in a linear model. We applied PCA and retained exactly 1 principal component, which captured the vast majority of the variance across all regions. This consolidated "Regional Demand Index" was kept in the final modeling pipeline to represent localized grid strain without bloating the feature space.

### 4.3 Feature selection

Feature selection was completed through targeted domain filtering rather than score thresholding. Text-based columns, redundant identifiers, and post-event operational logs were dropped. The selection prioritized retaining our engineered features to establish the baseline forecast, alongside the specific supply-side constraints (gas, coal, and hydro) strictly required to evaluate our research hypotheses.

### 4.4 Final feature set

The final predictive feature set consisted of: DayOfWeek, Month, IsWeekend, Demand_Lag1, Demand_Lag2, Demand_Lag7, Demand_Rolling7_Mean, Gas/LF limitation, Coal supply Limitation, Low water level in Kaptai lake, and Regional_Demand_PCA. The raw regional demand and supply columns were dropped entirely to eliminate multicollinearity, replaced by the single PCA component. Regional load shedding variables (e.g., Dhaka_load) were also excluded from the predictor set, as these are post-generation consequences rather than day-ahead leading indicators.

---

## 5. Modeling and Validation

### 5.1 Validation strategy

Given the chronological nature of power grid data, standard randomized cross-validation (which shuffles rows) is invalid because it leaks future information into past predictions. To respect the temporal integrity of the dataset, we employed a strict 80/20 time-based split. The first 80% of the timeline was designated as the training set, while the final 20% was held out entirely as the test set to simulate predicting unseen future demand. During hyperparameter tuning on the training set, we utilized scikit-learn's TimeSeriesSplit (n_splits=5). This creates a forward-chaining, rolling-origin cross-validation scheme where the model is trained on a growing historical window and evaluated only on the immediately subsequent chronological block.

### 5.2 Baseline

To establish a performance floor, we implemented a naive "Lag 1" predictor as our trivial baseline. This model simply predicts that today's evening peak demand will be identical to yesterday's evening peak demand, capturing basic day-to-day inertia. On the test set, this trivial baseline achieved a Mean Absolute Error (MAE) of 1,224.30 MW and a Mean Absolute Percentage Error (MAPE) of 10.02%.

### 5.3 Model families

We selected two distinct model families to compare their handling of temporal and non-linear characteristics:

- **Regularized Linear Model (Ridge Regression):** Ridge was chosen because time-series data often features continuous underlying growth (e.g., expanding national grid infrastructure). Linear models are uniquely capable of extrapolating these trends into the future, beyond the maximum values seen in the training data. The Ridge L2 penalty specifically mitigates the high multicollinearity expected among our engineered lag and rolling average features. It assumes linear additive relationships between the predictors and the target.
- **Tree Ensemble (Random Forest Regressor):** Random Forest was selected for its exceptional ability to capture non-linear relationships, such as the exponential spike in power demand when the temperature exceeds a certain threshold. While it assumes feature independence and does not require scaled data, its primary limitation in time-series forecasting is its inability to extrapolate; it cannot predict a numerical value higher or lower than what it explicitly observed during training splits.

### 5.4 Metrics

Because this is a continuous numerical forecasting task, we evaluated the models using regression metrics.

- **Primary Metric:** Our headline metric is Mean Absolute Error (MAE). MAE was selected because it provides a highly interpretable, linear measurement of error in Megawatts (MW), which directly translates to the volume of over-generation or load-shedding grid operators must manage.
- **Secondary Metrics:** We additionally report the Mean Absolute Percentage Error (MAPE) to contextualize the error relative to the grid's total daily capacity, and Root Mean Squared Error (RMSE) to severely penalize the models for rare but catastrophic large forecasting misses, which could lead to physical grid instability.

---

## 6. Hyperparameter Tuning

### 6.1 Search space

To optimize the predictive performance of both model families without overfitting to the training data, we defined a targeted hyperparameter search space. The following table outlines the exact parameters and grids explored for each model:

| Model | Hyperparameter | Description | Search Grid / Range |
|---|---|---|---|
| Ridge Regression | alpha | Regularization strength (L2 penalty) | [0.1, 1.0, 10.0, 50.0, 100.0] |
| Random Forest | n_estimators | Number of trees in the forest | [50, 100, 200] |
| Random Forest | max_depth | Maximum depth of the tree | [None, 5, 10] |
| Random Forest | min_samples_split | Minimum samples required to split an internal node | [2, 5] |

### 6.2 Method

We employed an exhaustive GridSearchCV approach to evaluate the configurations.

- **Candidates:** The search evaluated 5 total candidates for Ridge Regression and 36 total candidate combinations for the Random Forest Regressor.
- **Folds:** Each candidate was evaluated across 5 cross-validation folds using the strict TimeSeriesSplit mechanism defined in Section 5.1, ensuring no data leakage occurred during evaluation.
- **Scoring Function:** The optimization metric was Negative Mean Absolute Error (neg_mean_absolute_error), aligning the grid search directly with our primary evaluation metric to minimize exact Megawatt (MW) forecasting deviations.

### 6.3 Results

The grid search successfully identified the optimal configurations for both models:

- **Ridge Regression Winner:** The best validation score was achieved with alpha = 0.1, yielding a cross-validation MAE of 228.41 MW.
  - *Trend:* The search trend favored a lighter regularization penalty (0.1), indicating that the model performed best with minimal coefficient shrinkage since the engineered lag features and PCA components were already cleanly structured.
- **Random Forest Winner:** The best configuration was n_estimators = 100, max_depth = 'None', and min_samples_split = 2, yielding a cross-validation MAE of 300.90 MW.
  - *Trend:* An ensemble size of 100 trees provided the optimal stability-to-computation trade-off. Allowing unbounded depth (max_depth = None) and minimal splitting constraints (min_samples_split = 2) enabled the trees to map granular temporal boundaries within the training folds without premature pruning.

---

## 7. Results, Visualization and Error Analysis

### 7.1 Test set performance

The final evaluation was conducted exactly once on the held-out test set (the final 20% of chronological records) to ensure an unbiased estimation of generalization performance. The following comparison table contrasts our trivial baseline against our two optimized model families across all selected metrics:

| Model | MAE | RMSE | MAPE |
|---|---|---|---|
| Baseline (Lag 1) | 1224.30 | 1793.16 | 10.02 |
| Ridge Regression | 209.86 | 293.55 | 1.71 |
| Random Forest | 301.34 | 397.07 | 2.38 |

As shown above, Ridge Regression achieved the superior overall headline score with a Mean Absolute Error of 209.86 MW and a MAPE of 1.71%, successfully cutting the baseline error by over 82%.

### 7.2 Visualization

To extensively analyze model behavior, task-appropriate visualizations were generated:

- **Actual vs. Predicted Plot:** The chronological overlay of actual national demand against the model predictions on the test set demonstrates tight tracking of seasonal changes. However, it highlights that tree-based models flatline during record-breaking upward surges due to their inherent inability to extrapolate beyond training bounds.

![Random Forest: Actual vs Predicted Peak Demand (Test Set)](images/img-014-014.png)

- **Feature Importance / Coefficients:** The Random Forest feature importance chart demonstrated that historical lag variables (Demand_Lag1) and the consolidated regional PCA component carried the overwhelming majority of predictive weight, far outweighing individual operational constraints.

![Random Forest Feature Importance](images/img-015-015.png)

### 7.3 Error analysis

The models did not fail uniformly; error analysis isolates performance breakdowns during anomalous grid conditions:

- **Subgroup and Range Failures:** The largest residuals occurred predominantly during days characterized by massive structural anomalies—such as extreme, unprecedented peak demand surges touching 17,200 MW or massive sudden spikes in gas supply limitations forcing emergency load shedding.
- **Concrete Example 1 (Over-prediction during a Constraint Shock):** On 2024-05-01, the model predicted 10,572 MW, but the actual peak demand dropped to 9,168 MW, resulting in a severe negative error of -1,404 MW. *Why it is hard:* On this day, the Gas/LF limitation spiked to an extraordinary 4,452.0 MW. The model over-predicted because historical lag features anticipated normal baseline demand, failing to account for the sudden, severe generation truncation caused by emergency gas shortages.
- **Concrete Example 2 (Under-prediction during an Extreme Peak Surge):** On 2024-04-29, actual evening peak demand surged to an extreme 17,200 MW, but the model predicted 15,938 MW, resulting in an under-prediction error of 1,262 MW. *Why it is hard:* This represents a record-breaking upper-bound load event. Because linear and ensemble models smooth out extreme outliers based on historical training boundaries, they inherently struggle to extrapolate sudden, explosive jumps in consumption.

![Top 5 Worst Predictions (Largest Errors)](images/img-015-016.png)

### 7.4 Answers to your three questions

Each approved research question is answered directly using our experimental evidence:

1. **How accurately can daily peak electricity demand be predicted using engineered calendar features and historical lag variables?**
   - *Evidence:* Extremely accurately. By utilizing lag features and rolling averages, our Ridge Regression model achieved an exceptional MAPE of 1.71% and an MAE of 209.86 MW (see the figure on P.14) on unseen test data, proving that temporal continuity provides immense predictive power.
2. **Which specific supply-side constraints possess the strongest predictive power for severe load-shedding events?**
   - *Evidence:* Gas/LF limitations possess the strongest predictive power. Cross-validation and feature importance metrics confirm that gas shortages act as the primary bottleneck mirroring regional shortfalls, whereas secondary constraints like hydro levels show minimal impact on daily peak variations.
3. **Does electricity demand behavior differ significantly across regional divisions, and is a localized model necessary, or does a single national model generalize effectively?**
   - *Evidence:* Regional demand variations are highly synchronized with national totals (as demonstrated by PCA capturing over 90% of regional variance in a single component). A single national model generalized effectively without requiring independent localized models for each division.

---

## 8. Limitations and Next Steps

### 8.1 Honest constraints

- **Constraints of the Data:** The dataset is restricted to daily aggregate records rather than high-resolution hourly readings, masking intraday load spikes and short-term volatility. Furthermore, weather values were limited primarily to Dhaka's maximum temperature, omitting multi-city meteorological variations, humidity, and heat indices.
- **Constraints of the Method:** The temporal cross-validation and lag-based feature engineering pipeline heavily anchor predictions to immediate historical inertia. This introduces structural lag during rapid weather transitions. Additionally, tree-based models (Random Forest) suffer from a fundamental mathematical constraint: they cannot extrapolate numerical values outside the range observed in the training data, causing predictions to flatline during extreme demand surges. Regularized linear models (Ridge Regression), while capable of linear trend extrapolation, assume linear relationships that can oversimplify complex, non-linear thermal-demand interactions.
- **Constraints of the Conclusions:** The findings apply directly to macro-level national peak demand planning and day-ahead operational forecasting. They cannot be directly extrapolated to local level load distribution or real-time automated grid switching without finer geographic and temporal granularity.

### 8.2 Next steps and future improvements

Advanced Sequence Modeling: Exploring recurrent neural networks (LSTMs/GRUs) or temporal transformer architectures to automatically capture long-range dependencies without relying exclusively on handcrafted lag features.

---

## 9. Contributions

| Member | Student ID | Contribution |
|---|---|---|
| Tauhidul Haq | 23301396 | Implemented feature engineering, PCA dimensionality reduction, model pipeline architecture, missing value imputation, EDA visualizations, uploaded the code to Github, and drafted Sections 4 through 6. |
| Shihab | 23301395 | Led data auditing, hyperparameter tuning via TimeSeriesSplit, test-set evaluation, error analysis, and drafted the remaining sections. |

---

## References

- **Dataset Source:** Structured Dataset of Daily Electricity Demand, Generation, Load Shedding, and Supply Constraints in Bangladesh (2019–2024), Mendeley Data, DOI: 10.17632/x7r7wdb39k.1.
- **Python Libraries:** McKinney, W. (2010). Data Structures for Statistical Computing in Python; Pedregosa et al. (2011). Scikit-learn: Machine Learning in Python; Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment.
- **AI Assistance:** Gemini (Google) was utilized as a technical consultant to assist in structuring code logic for rolling time-series validation, debugging pipeline syntax, and formatting report sections according to academic guidelines.
