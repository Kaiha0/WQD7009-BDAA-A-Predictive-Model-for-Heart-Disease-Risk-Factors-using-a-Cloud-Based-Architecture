# Developing a Predictive Model for Heart Disease Risk Factors using a Cloud-Based Architecture

This project implements a comprehensive, cloud-native data lifecycle on **Google Cloud Platform (GCP)** to identify and predict heart disease risk factors through a multi-layer hybrid architecture.

## Table of Content
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Dataset Description](#dataset-description)
4. [Implementation Phases](#implementation-phases)
    * [Phase 1: Data Ingestion (Bronze Layer)](#phase-1-data-ingestion-bronze-layer)
    * [Phase 2: Data Processing (Silver Layer)](#phase-2-data-processing-silver-layer)
    * [Phase 3: Structured Storage & Analytics](#phase-3-structured-storage--analytics)
    * [Phase 4: Advanced Modeling (Gold Layer)](#phase-4-advanced-modeling-gold-layer)
    * [Phase 5: Data Visualization](#phase-5-data-visualization)
5. [Performance Evaluation](#performance-evaluation)
6. [Conclusion](#conclusion)
7. [Dataset Description & Access](#dataset-description-&-Access)
8. [Disclaimer](#disclaimer)

---

## Project Overview
The project addresses the need for scalable and efficient predictive systems in healthcare. By leveraging GCP, it provides a framework for managing large datasets and ensuring model interpretability to improve preventive healthcare strategies.

**Key Objectives:**
* Design a robust architecture for high-volume data processing.
* Implement ML algorithms to identify critical heart disease risk factors.
* Validate performance using clinically relevant metrics like ROC-AUC.

---

## System Architecture
The architecture follows a **Five-Layer Data Lifecycle** designed to transform raw input into actionable medical insights.


* **Bronze Layer:** Raw data ingestion.
* **Silver Layer:** Cleaned and structured data.
* **Gold Layer:** Enriched data with model predictions.

---

## Dataset Description
* **Source:** `heart_2022_no_nans.csv` from Kaggle.
* **Format:** Raw CSV (Initial) → Cleaned BigQuery Tables (Silver) → Enriched Prediction Tables (Gold).

---

## Implementation Phases

### Phase 1: Data Ingestion (Bronze Layer)
* **Tool:** **Google Cloud Storage (GCS)**.
* **Action:** Established the `cardiopredict-bronzedata-12345` bucket to store raw CSV data and PySpark scripts as read-only "Bronze" files.

### Phase 2: Data Processing (Silver Layer)
* **Tool:** **Dataproc with PySpark**.
* **Action:** Deployed a Spark cluster to resolve data type issues and convert categorical variables into numerical values.
* **Efficiency:** The preprocessing job completed in approximately **1.9 minutes**.

### Phase 3: Structured Storage & Analytics
* **Tool:** **BigQuery & BigQuery ML**.
* **Action:** Created the `heart_analytics_dataset` for structured storage.
* **Analytics:** Used SQL commands to train a Logistic Regression model for feature selection, retaining features with weights > 0.3.

### Phase 4: Advanced Modeling (Gold Layer)
* **Tool:** **Vertex AI Workbench**.
* **Technique:** Applied **SMOTE** (Synthetic Minority Over-sampling Technique) to balance target labels and trained a final Logistic Regression model.
* **Storage:** Prediction results were saved as the `heart_analytics_gold` table.

### Phase 5: Data Visualization
* **Tool:** **Power BI Desktop**.
* **Action:** Linked Power BI directly to BigQuery via a cloud connector to create interactive dashboards showing risk levels across demographics.

---

## Performance Evaluation
* **Processing Speed:** Dataproc tasks achieved maximum parallelism across 4 CPU cores.
* **Memory Management:** Disk usage remained at **0.0 B**, signifying that all operations were handled entirely in memory.
* **SQL Performance:** Most analytical queries were completed in approximately **2 seconds**.

---

## Conclusion
The practical implementation confirms that the hybrid cloud architecture successfully transforms raw healthcare data into high-quality insights in a reliable and scalable manner.

---

## Dataset Description & Access
Due to the file size (approx. 80MB), the raw dataset is not hosted directly in this repository. 

* **Dataset Source:** [Kaggle - Personal Key Indicators of Heart Disease](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease/data)
* **File Name:** `heart_2022_no_nans.csv`

### How to add the data to your GCP Environment:
1. Download the `heart_2022_no_nans.csv` from the link above.
2. Upload the file to your **Google Cloud Storage (GCS)** bucket named `cardiopredict-bronzedata-12345`.
3. Ensure the file path in `src/dataproc_heart_analysis.py` matches your GCS URI.

---

## Disclaimer
This project was developed for **educational purposes**. While commercial usage is welcomed, the author is **not liable** for any losses or GCP service charges incurred due to the use of this repository. Users are responsible for monitoring their own cloud billing.
