# Book Recommendation System (ML-based)

## Overview
This project builds a recommendation system to predict the next book a user is likely to read based on historical reading interactions.

The solution is designed for Google Colab and uses:
- candidate generation from popularity, collaborative filtering, and content signals
- a supervised machine learning ranking model
- evaluation using ranking metrics

## Problem Statement
Given user reading history at chapter level, recommend what the user should read next.

Because the available interaction data does not include explicit timestamps, the main task is framed as **next-book recommendation**. In addition, the project provides a simple **continue-reading / next-chapter heuristic** for users who are already progressing through a book.

## Input Files
The project expects these two CSV files:

### 1. `interactions.csv`
Contains user interaction history:
- 'user_id'
- 'book_id'
- 'chapter_id'

Each row means a user interacted with a chapter belonging to a book.

### 2. `chapters.csv`
Contains chapter and content metadata:
- 'chapter_id'
- 'book_id'
- 'chapter_sequence_no'
- 'author_id'
- 'published_date'
- 'tags'

This file is used to derive book-level content features such as author, chapter count, publication year, and tags.

## Approach

### Step 1: Data preparation
- Load both CSV files
- Clean and standardize columns
- Aggregate chapter-level interactions into user-book interactions
- Derive book-level metadata from 'chapters.csv'

### Step 2: Candidate generation
Generate recommendation candidates using:
- Popularity-based candidates
- Collaborative filtering style co-read patterns
- Content similarity signals

### Step 3: Feature engineering
Create training features such as:
- book popularity
- author match with user history
- tag overlap with user history
- chapter count
- publication year gap
- collaborative similarity score

### Step 4: ML model
Train a supervised recommendation model:
- **Primary model:** XGBoost classifier if available
- **Fallback model:** HistGradientBoostingClassifier from scikit-learn

The model predicts the likelihood that a user will read a candidate book.

### Step 5: Ranking
For each user:
- score candidate books
- rank them by predicted score
- return top-K recommendations

### Step 6: Evaluation
Evaluate recommendations using:
- Hit@5
- Hit@10
- MRR@10
- NDCG@10

## Output Files
The script saves outputs under '/content/reco_outputs' in Google Colab.

Expected output files:
- 'evaluation_summary.csv' - overall model metrics
- 'evaluation_detailed.csv' - detailed user-level evaluation
- 'sample_recommendations.csv' - example recommendations for users
- 'continue_reading_examples.csv' - next-chapter suggestions for active books
- 'run_report.txt' - run summary and configuration details

The script also creates and downloads:
- 'reco_outputs.zip'

## How to Run in Google Colab
1. Open a new notebook in Google Colab.
2. Copy the code from 'google_colab_best_model_recommender_fast.py' into a code cell.
3. Run the cell.
4. Upload:
   - 'interactions.csv'
   - 'chapters.csv'
5. Wait for the model to train and outputs to be generated.
6. Download the generated zip file automatically.


## Limitations
- no explicit timestamps in interactions, so strict sequence modeling is limited
- evaluation is offline and based on held-out interactions
- recommendations are mainly book-level, with chapter continuation handled through a rule-based extension

## Future Improvements
- add time-aware sequence modeling if timestamps become available
- use LightGBM ranker or LambdaMART
- add user-level embeddings
- improve cold-start handling for new users and new books
- deploy as a two-stage retrieval and ranking system

## Author Note
This project was created as a data science assignment submission demonstrating recommendation-system design, ML modeling, feature engineering, and ranking evaluation.

##  Author

Raja Naik
