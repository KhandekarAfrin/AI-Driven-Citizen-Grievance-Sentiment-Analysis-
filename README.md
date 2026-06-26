# AI-Driven-Citizen-Grievance-Sentiment-Analysis-
AI-powered NLP system that classifies citizen grievances into government departments and scores sentiment/urgency to prioritize civic complaint resolution. Built with TF-IDF, scikit-learn, and FastAPI.

# 📌 Problem Statement
Citizen complaint systems are often disjointed and slow: complaints sit unsorted across departments, and officials must manually read through large volumes of feedback to figure out what's urgent. This project automates that triage using NLP — classifying each complaint by department and flagging its sentiment/urgency so high-priority issues surface first.

# ✨ Features
- Text preprocessing pipeline — cleans raw complaint text (lowercasing, URL/special character removal, lemmatization, stopword removal).
- Exploratory Data Analysis — word clouds and n-gram frequency distributions to surface the most common complaint themes.
- Department classification — predicts which civic department a complaint should be routed to (e.g. Sanitation, Water, Transport), using TF-IDF vectorization and supervised ML models.

# Project Structure
.
├── notebooks/
│   ├── week1_data_cleaning_eda.ipynb
│   └── week2_department_classification.ipynb
├── requirements.txt
├── .gitignore
└── README.md

# Week 1 — Data Cleaning & EDA
- Cleaned raw complaint text: lowercased, removed URLs/special characters, applied lemmatization and stopword removal.
- Visualized common complaint themes using word clouds and n-gram frequency distributions.

# Week 2 — Department Classification
- Vectorized cleaned text using TF-IDF.
- Trained and compared Logistic Regression and Random Forest classifiers to predict the target department.
- Identified and fixed a label leakage bug (the department name was leaking into the text features) — corrected by removing       leaking columns before vectorization.
- Used cross-validation and class balancing to ensure the model generalizes across departments, including minority classes.

# Planned Next Steps
- Week 3: Sentiment and urgency scoring (VADER-based pseudo-labeling + classifier).
- Week 4: Model evaluation, FastAPI deployment, and a Streamlit-based web interface for citizens and civic officials.
