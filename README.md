# Spark-ML

End-to-end PySpark project on the UCI Online Retail dataset: customer segmentation via RFM + clustering (BisectingKMeans, GMM) and high-spender classification with Random Forest. Advanced feature engineering, evaluation (F1 0.88), and business recommendations focused on retention and growth.

---

## What it does

- Builds RFM features (Recency, Frequency, Monetary) per customer
- Clusters customers with BisectingKMeans and GaussianMixture
- Predicts high-spenders with a Random Forest classifier

## Results

| Model | Metric | Score |
|---|---|---|
| BisectingKMeans (k=3) | Silhouette | 0.74 |
| Random Forest | Accuracy | 0.87 |
| Random Forest | F1 | 0.88 |

**Segments found:**

| Cluster | Label | Customers | Avg Recency | Avg Frequency | Avg Monetary |
|---|---|---|---|---|---|
| 1 | VIP | 25 | 5.5 days | 67.8 orders | £85,860 |
| 0 | Regulars | 3,200 | 40 days | 4.7 orders | £1,856 |
| 2 | Inactive | 1,109 | 244 days | 1.6 orders | £586 |

**Top predictive features:** Frequency (0.46) → UniqueProducts (0.23) → OrderFrequencyRate (0.18)

## Setup

```bash
pip install pyspark gdown
```

Download the dataset:

```python
FILE_ID = "1e3JrdZHo4fpmcnf4h8-m6XA4JKt-u2hb"
# !gdown --id $FILE_ID -O Online_Retail_CSV.csv
```

Run the notebook in Jupyter or Google Colab.

## Features engineered

```
Recency            — days since last purchase
Frequency          — distinct invoice count
AvgUnitPrice       — mean unit price
UniqueProducts     — distinct SKUs purchased
PurchaseSpan       — days between first and last order
OrderFrequencyRate — orders per active day
CountryIndex       — encoded country
```

`Monetary` is used only to derive the binary label (above/below median spend) — not as a model feature.

## Notes

- Decimal separator fixed from EU format (`,` → `.`) before casting
- Cancelled invoices (`InvoiceNo` starting with `C`) excluded
- Non-product stock codes (postage, fees) filtered via regex `^[0-9]{5}`
- StandardScaler applied before clustering (`withMean=False` to avoid NaN on sparse vectors)

## Limitations

- ~1 year of data from a single UK-based retailer
- No demographic or acquisition channel data
- 25 VIP customers heavily skew the Monetary distribution

## Dataset

[UCI Online Retail](https://archive.ics.uci.edu/ml/datasets/online+retail)


## Collaborators

- Jacques Allison
- Cedric Manelli
- Sufyan Nadat
