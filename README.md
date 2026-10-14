# Customer Segmentation Using RFM Analysis and K-Means Clustering

## Project Overview

This project applies unsupervised machine learning to retail transaction data to identify customer segments based on purchasing behaviour. Using Recency, Frequency, and Monetary (RFM) analysis, the project groups customers with similar characteristics and translates the resulting patterns into actionable business insights.

The objective is to support customer retention, targeted marketing, and customer value analysis through data-driven segmentation.

## Problem Statement

Retail businesses serve customers with different purchasing patterns, spending levels, and levels of engagement. Treating every customer identically can limit the effectiveness of marketing and retention strategies.

This project uses K-Means clustering to discover customer groups without predefined labels, enabling the business to develop strategies tailored to each segment.

## Dataset

**Source:** [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

The analysis uses the first period of the Online Retail II dataset, covering December 2009 to December 2010.

The dataset contains transaction-level information, including:

- Invoice numbers
- Product codes and descriptions
- Quantities purchased
- Invoice dates
- Unit prices
- Customer IDs
- Countries

The original dataset contains 525,461 transaction records. Following duplicate removal and exclusion of records without customer IDs, 410,763 records remained for analysis.

The raw Excel dataset is not included in this repository. Download it from the source above and place `online_retail_II.xlsx` in the project directory before running the notebook.

## Technologies Used

- **Python** — data processing and model development
- **Pandas and NumPy** — data manipulation and numerical operations
- **Matplotlib and Seaborn** — exploratory data analysis
- **Scikit-learn** — preprocessing, K-Means clustering, and model evaluation
- **Joblib** — model persistence
- **Jupyter Notebook** — interactive analysis and experimentation

## Project Workflow

### 1. Data Inspection and Cleaning

Inspected the dataset structure, data types, missing values, duplicate records, and date range. Removed duplicate records and transactions without customer IDs while retaining recorded returns and cancellations in the cleaned transaction data.

### 2. Exploratory Data Analysis

Examined transaction quantities, prices, country-level activity, sales patterns, and customer spending to understand the dataset before feature engineering.

### 3. RFM Feature Engineering

Constructed three customer-level features:

| Feature | Definition |
|---|---|
| Recency | Number of days since the customer's most recent qualifying purchase |
| Frequency | Number of distinct invoices associated with qualifying purchases |
| Monetary | Net sales value after accounting for recorded negative sales |

For Recency and Frequency, cancellation invoices and transactions with non-positive quantities were excluded from qualifying purchases. Monetary values were calculated from the cleaned transaction records, including recorded negative sales associated with returns and cancellations.

Customers with non-positive net Monetary values were excluded from the clustering dataset because the selected transformation and business interpretation required positive net spending.

### 4. Preprocessing

The RFM features had different scales and substantially skewed distributions. A scikit-learn pipeline was used to apply:

1. **Yeo-Johnson Power Transformation** to reduce feature skewness.
2. **StandardScaler** to standardise the transformed features.

The fitted pipeline was retained so that new customer records can undergo the same transformations before prediction.

### 5. K-Means Clustering

K-Means was evaluated using different cluster counts. Inertia was examined to understand how within-cluster variation changed as the number of clusters increased. Silhouette scores were used to compare cluster separation.

The final model used **K=2**, selected because it achieved the highest silhouette score among the tested values and produced interpretable differences in customer purchasing behaviour.

### 6. Cluster Profiling

The resulting customer groups were profiled using customer counts and median RFM values. Median values were used to describe typical customers while reducing the influence of extreme spending and purchasing behaviour.

### 7. Business Interpretation

The identified segments were translated into potential retention and re-engagement strategies. These recommendations are hypotheses that should be evaluated using actual campaign performance.

### 8. Model Persistence and Prediction

The fitted preprocessing pipeline, trained K-Means model, and feature names were saved together in a Joblib-compatible artifact for reuse. A new customer's RFM values can be transformed and assigned to a cluster without retraining the model.

## Model Evaluation

Silhouette scores were compared for cluster counts from K=2 to K=6.

| Number of Clusters | Silhouette Score |
|---:|---:|
| 2 | 0.4224 |
| 3 | 0.3258 |
| 4 | 0.3356 |
| 5 | 0.3232 |
| 6 | 0.3237 |

K=2 achieved the highest silhouette score of **0.4224** among the tested configurations.

The silhouette score measures how well-separated and cohesive the clusters are. It does not, by itself, establish business value or prove that the segments will improve marketing outcomes.

## Customer Segmentation Results

The final clustering dataset contained **4,283 customers**.

| Metric | Cluster 0 | Cluster 1 |
|---|---:|---:|
| Customer Count | 2,036 | 2,247 |
| Median Recency | 22 days | 112 days |
| Median Frequency | 5 orders | 1 order |
| Median Monetary | £1,709.31 | £311.78 |

### Cluster 0: Active, Higher-Value Customers

Cluster 0 contains customers who purchased more recently, placed more orders, and had higher median net spending.

**Recommended strategy:**
- Encourage repeat purchases through loyalty initiatives.
- Provide relevant product recommendations.
- Test personalised offers designed to strengthen customer retention.
- Monitor repeat-purchase rates and customer spending over time.

### Cluster 1: Less Active, Lower-Value Customers

Cluster 1 contains customers whose last qualifying purchase was generally further in the past, with fewer orders and lower median net spending.

**Recommended strategy:**
- Test targeted re-engagement campaigns.
- Recommend products based on previous purchases.
- Evaluate incentives designed to encourage another purchase.
- Monitor reactivation rates and subsequent purchasing behaviour.

These segment names describe historical purchasing patterns rather than permanent customer characteristics. Cluster labels are arbitrary identifiers; the RFM profiles provide their business interpretation.

## Repository Structure

```text
Customer-Segmentation-ML/
├── customer_segmentation_ML.ipynb
├── customer_segmentation_model.pkl
└── .gitignore
```

- `customer_segmentation_ML.ipynb` — notebook containing data cleaning, exploratory analysis, RFM feature engineering, preprocessing, model selection, clustering, and business interpretation.
- `customer_segmentation_model.pkl` — saved preprocessing pipeline, trained K-Means model, and feature names.
- `.gitignore` — excludes the raw dataset, Jupyter checkpoints, Python cache files, and `.env` files.

## Installation and Usage

### Prerequisites

Install Python and the required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib openpyxl jupyter
```

### Run the Notebook

1. Clone the repository:

   ```bash
   git clone https://github.com/kenetworkltd-oss/Customer-Segmentation-ML.git
   ```

2. Enter the project directory:

   ```bash
   cd Customer-Segmentation-ML
   ```

3. Download the Online Retail II dataset from the UCI Machine Learning Repository and place `online_retail_II.xlsx` in the project directory.

4. Launch Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

5. Open `customer_segmentation_ML.ipynb` and run the notebook cells in order.

### Load the Saved Model

```python
import joblib

artifact = joblib.load("customer_segmentation_model.pkl")

preprocessor = artifact["preprocessor"]
model = artifact["model"]
features = artifact["features"]
```

To assign a cluster to a new customer, supply raw RFM values in the expected feature order and apply the existing fitted preprocessing pipeline before prediction:

```python
import pandas as pd

new_customer = pd.DataFrame({
    "Recency": [20],
    "Frequency": [6],
    "Monetary": [2000]
})

new_customer_scaled = preprocessor.transform(new_customer[features])
predicted_cluster = model.predict(new_customer_scaled)

print(predicted_cluster[0])
```

This example demonstrates how to use the saved model with a hypothetical customer; it is not a new observation from the original dataset.

## Limitations and Future Improvements

- The analysis reflects historical transactions from the selected dataset period and may not represent current purchasing behaviour.
- K-Means is sensitive to feature representation, scaling, outliers, and the selected number of clusters.
- Silhouette scores measure clustering structure rather than profitability or marketing effectiveness.
- Excluding customers with non-positive net Monetary values means the final segments do not represent every customer in the cleaned transaction dataset.
- The proposed marketing strategies require validation through campaign experiments and subsequent customer behaviour.

Potential improvements include comparing alternative clustering algorithms, evaluating segment stability, incorporating additional behavioural features, and deploying the model through an API or interactive application.

## Conclusion

This project demonstrates how RFM feature engineering and K-Means clustering can uncover distinct customer purchasing patterns in retail transaction data. The resulting two-segment model differentiates more active, higher-spending customers from less active, lower-spending customers and provides a structured basis for customer retention and re-engagement strategies.

The saved model and preprocessing pipeline support consistent assignment of new customer records to the learned segments, while further testing and deployment would be required for production use.
