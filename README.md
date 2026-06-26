# AI-Driven-Citizen-Grievance-Sentiment-Analysis-
AI-powered NLP system that classifies citizen grievances into government departments and scores sentiment/urgency to prioritize civic complaint resolution. Built with TF-IDF, scikit-learn, and FastAPI.

# 📌 Problem Statement
Citizen complaint systems are often disjointed and slow: complaints sit unsorted across departments, and officials must manually read through large volumes of feedback to figure out what's urgent. This project automates that triage using NLP — classifying each complaint by department and flagging its sentiment/urgency so high-priority issues surface first.

# ✨ Features
- Department classification — predicts which civic department (e.g. Sanitation, Water, Transport) a complaint should be routed to, using TF-IDF + supervised ML models.
- Sentiment & urgency scoring — classifies each complaint as Positive, Neutral, Negative, or Critical/Urgent, combining a lexicon-based sentiment score (VADER) with rule-based urgency keyword detection.

# Week 1 — Data Cleaning & EDA
- Cleaned raw complaint text: lowercased, removed URLs/special characters, applied lemmatization and stopword removal.
- Visualized common complaint themes using word clouds and n-gram frequency distributions.

# Week 2 — Department Classification
- Vectorized cleaned text using TF-IDF.
- Trained and compared Logistic Regression and Random Forest classifiers to predict the target department.
- Identified and fixed a label leakage bug (the department name was leaking into the text features) — corrected by removing       leaking columns before vectorization.
- Used cross-validation and class balancing to ensure the model generalizes across departments, including minority classes.
