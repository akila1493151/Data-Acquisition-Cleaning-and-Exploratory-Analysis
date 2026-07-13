# Automated Exploratory Data Analysis (EDA) & Data Cleaning Pipeline

An end-to-end Python pipeline engineered to ingest raw, messy tabular datasets, perform rigorous structural audits, diagnose quality anomalies (nulls, duplicates, outliers), engineer targeted statistical imputations, and export a clean data asset ready for machine learning production.

---

## 📋 Table of Contents
1. [Pipeline Architecture](#-pipeline-architecture)
2. [Data Processing Workflow](#-data-processing-workflow)
3. [Key Statistical Insights](#-key-statistical-insights)
4. [Visualizations Generated](#-visualizations-generated)
5. [Setup & Installation](#-setup--installation)
6. [Usage & Outputs](#-usage--outputs)

---

## 🏗️ Pipeline Architecture

The pipeline processes data deterministically through 10 isolated tasks, transitioning the dataset from raw ingestion to serialized output:
