# Book Recommendation System

## 1. Objective
The goal of this project is to recommend the next book a user is likely to read using historical reading interactions and chapter metadata.

Although the raw data is chapter-level, the main recommendation task is framed at the **book level** because:
- the assignment allows flexible scoping
- book recommendation is a meaningful business problem
- the interaction file does not include timestamps needed for stronger sequential modeling

A simple continue-reading / next-chapter output is also included for users who appear to be progressing through an existing book.

## 2. Data Used

### interactions.csv
This file captures reading behavior:
- user_id
- book_id
- chapter_id

It tells us which chapters and books a user has interacted with.

### chapters.csv
This file contains content metadata:
- chapter_id
- chapter_sequence_no
- book_id
- author_id
- published_date
- tags

This helps create book-level descriptive features.

## 3. Problem Framing
I framed the assignment as a **next-book recommendation problem** supported by a simple next-chapter continuation heuristic.

This framing is practical because:
- it uses the available data effectively
- it supports a proper machine learning setup
- it remains easy to explain and evaluate

## 4. Methodology

### 4.1 Candidate generation
To avoid scoring every book for every user, I first create candidate books using:
- popularity
- collaborative co-reading patterns
- content-based similarity

This makes the pipeline faster and more realistic.

### 4.2 Feature engineering
For each user-candidate book pair, I create features such as:
- popularity of the candidate book
- whether the candidate author matches authors in the user’s history
- overlap between candidate tags and user’s historical tags
- chapter count
- publication year distance
- collaborative signal from prior co-read books

### 4.3 ML model
I use a supervised learning model to rank candidate books:
- XGBoost if available
- otherwise HistGradientBoostingClassifier

The model predicts the probability that a user will interact with a candidate book.

### 4.4 Ranking
For each user, candidate books are scored and ranked. The top-ranked books are returned as recommendations.

## 5. Evaluation Strategy
I use an offline evaluation setup where one known user interaction is held out and the model must recover it from the candidate list.

Metrics:
- **Hit@5 / Hit@10**: whether the true book is present in the top recommendations
- **MRR@10**: rewards higher-ranked correct recommendations
- **NDCG@10**: measures ranking quality with more weight for top positions

These are standard ranking metrics for recommendation systems.

## 6. Why this is a strong solution
This solution is strong because it:
- uses a real ML model rather than only heuristics
- combines behavioral and content signals
- is scalable enough for Colab
- is explainable and interview-friendly
- follows a practical two-stage recommendation design

## 7. Limitations
- lack of timestamps limits true sequence modeling
- no explicit feedback strength is available
- cold-start users remain challenging
- chapter continuation is not modeled with a dedicated sequence model


## 8. Conclusion
This project demonstrates a practical recommendation system pipeline using:
- reading behavior data
- content metadata
- candidate generation
- supervised ML ranking
- ranking-based evaluation