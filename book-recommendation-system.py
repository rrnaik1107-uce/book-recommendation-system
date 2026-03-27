# Python script: hybrid recommender + ML ranking model

import os
import re
import math
import time
import json
import random
import zipfile
import warnings
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
MAX_USERS_FOR_TRAIN_EVAL = 15000   # increase if you want more coverage, decrease if slower
MAX_CF_CANDS = 20
MAX_POP_CANDS = 20
MAX_NEGATIVES_PER_USER = 20
TOP_K_EVAL = 10
OUTPUT_DIR = '/content/reco_outputs1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

start_all = time.time()

print('=' * 90)
print('Upload interactions.csv and chapters.csv')
print('=' * 90)
from google.colab import files
_ = files.upload()

print('\nCurrent files in session:')
print(os.listdir())

if not os.path.exists('interactions.csv') or not os.path.exists('chapters.csv'):
    raise ValueError('Upload completed, but interactions.csv or chapters.csv is not present in current session.')
#  Load data
load_t0 = time.time()
interactions = pd.read_csv('interactions.csv', usecols=['user_id', 'chapter_id', 'book_id'])
chapters = pd.read_csv('chapters.csv', usecols=['chapter_id', 'chapter_sequence_no', 'book_id', 'author_id', 'published_date', 'tags'])
print(f'Loaded interactions: {interactions.shape}, chapters: {chapters.shape} in {time.time() - load_t0:.1f}s')

interactions = interactions.drop_duplicates().copy()
chapters = chapters.drop_duplicates().copy()


# 2 Clean metadata and build book-level table
prep_t0 = time.time()
chapters['author_id'] = chapters['author_id'].fillna('unknown').astype(str)
chapters['tags'] = chapters['tags'].fillna('').astype(str)
chapters['published_date'] = pd.to_datetime(chapters['published_date'], errors='coerce')
chapters['published_year'] = chapters['published_date'].dt.year
median_year = int(chapters['published_year'].dropna().median()) if chapters['published_year'].notna().any() else 0
chapters['published_year'] = chapters['published_year'].fillna(median_year).astype(int)
chapters['chapter_sequence_no'] = pd.to_numeric(chapters['chapter_sequence_no'], errors='coerce')

def normalize_tags(x: str):
    x = x.lower().replace('|', ' ').replace(',', ' ')
    x = re.sub(r'[^a-z0-9\s]+', ' ', x)
    toks = [t for t in x.split() if len(t) > 1]
    return toks[:20]

chapters['tag_tokens'] = chapters['tags'].apply(normalize_tags)

book_meta = chapters.groupby('book_id').agg(
    author_id=('author_id', lambda s: s.mode().iloc[0] if not s.mode().empty else str(s.iloc[0])),
    published_year=('published_year', 'median'),
    n_chapters=('chapter_id', 'nunique'),
    max_chapter_no=('chapter_sequence_no', 'max'),
).reset_index()
book_meta['published_year'] = book_meta['published_year'].fillna(median_year).astype(int)
book_meta['n_chapters'] = book_meta['n_chapters'].fillna(0).astype(int)
book_meta['max_chapter_no'] = book_meta['max_chapter_no'].fillna(0)

# top tags per book 
book_tag_counter = chapters.groupby('book_id')['tag_tokens'].sum().apply(Counter)
book_top_tags = {b: set([t for t, _ in cnt.most_common(12)]) for b, cnt in book_tag_counter.items()}

# 3 User-book table and popularity
user_book = interactions[['user_id', 'book_id']].drop_duplicates().copy()
user_book['interaction'] = 1

book_pop = user_book.groupby('book_id')['user_id'].nunique().rename('book_popularity').reset_index()
book_pop['book_popularity_log'] = np.log1p(book_pop['book_popularity'])
book_meta = book_meta.merge(book_pop, on='book_id', how='left')
book_meta['book_popularity'] = book_meta['book_popularity'].fillna(0)
book_meta['book_popularity_log'] = book_meta['book_popularity_log'].fillna(0)

user_counts = user_book.groupby('user_id')['book_id'].nunique().rename('num_books').reset_index()
eligible_users = user_counts[user_counts['num_books'] >= 2]['user_id']
user_book = user_book[user_book['user_id'].isin(eligible_users)].copy()

# sample users for faster runtime
all_eligible_users = user_book['user_id'].drop_duplicates().tolist()
if len(all_eligible_users) > MAX_USERS_FOR_TRAIN_EVAL:
    rng = np.random.default_rng(SEED)
    sampled_users = set(rng.choice(all_eligible_users, size=MAX_USERS_FOR_TRAIN_EVAL, replace=False).tolist())
    user_book = user_book[user_book['user_id'].isin(sampled_users)].copy()
else:
    sampled_users = set(all_eligible_users)

print(f'Users kept for training/eval: {user_book.user_id.nunique():,}')
print(f'Books in scope: {user_book.book_id.nunique():,}')
print(f'Prep completed in {time.time() - prep_t0:.1f}s')


# 4 Hold out one book per user
split_t0 = time.time()
heldout_rows = []
train_rows = []
for user, grp in user_book.groupby('user_id'):
    books = grp['book_id'].tolist()
    rnd = random.Random((hash(str(user)) + SEED) % (2**32 - 1))
    pos = rnd.choice(books)
    heldout_rows.append((user, pos))
    for b in books:
        if b != pos:
            train_rows.append((user, b))

train_user_book = pd.DataFrame(train_rows, columns=['user_id', 'book_id'])
heldout = pd.DataFrame(heldout_rows, columns=['user_id', 'target_book_id'])
train_counts = train_user_book.groupby('user_id')['book_id'].nunique().rename('n_train').reset_index()
heldout = heldout.merge(train_counts, on='user_id', how='left')
heldout = heldout[heldout['n_train'] >= 1].copy()
train_user_book = train_user_book[train_user_book['user_id'].isin(set(heldout['user_id']))].copy()
print(f'Holdout users: {heldout.user_id.nunique():,} in {time.time() - split_t0:.1f}s')

# 5 Collaborative filtering neighbors
cf_t0 = time.time()
all_books = sorted(book_meta['book_id'].unique())
train_users = sorted(train_user_book['user_id'].unique())
book_to_idx = {b: i for i, b in enumerate(all_books)}
idx_to_book = {i: b for b, i in book_to_idx.items()}
user_to_idx = {u: i for i, u in enumerate(train_users)}

rows = train_user_book['user_id'].map(user_to_idx).values
cols = train_user_book['book_id'].map(book_to_idx).values
vals = np.ones(len(train_user_book), dtype=np.float32)
user_item = sparse.csr_matrix((vals, (rows, cols)), shape=(len(train_users), len(all_books)))
item_user = user_item.T.tocsr()

n_neighbors = min(16, len(all_books))
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
knn.fit(item_user)
distances, indices = knn.kneighbors(item_user)

book_neighbors = {}
for item_idx in range(len(all_books)):
    neigh = []
    for d, j in zip(distances[item_idx], indices[item_idx]):
        if j == item_idx:
            continue
        sim = 1.0 - float(d)
        if sim > 0:
            neigh.append((idx_to_book[j], sim))
    book_neighbors[idx_to_book[item_idx]] = neigh
print(f'CF neighbors ready in {time.time() - cf_t0:.1f}s')

# 6 Dictionaries for fast features

book_meta_dict = book_meta.set_index('book_id').to_dict('index')
popular_books = book_pop.sort_values('book_popularity', ascending=False)['book_id'].tolist()
all_book_set = set(all_books)
user_train_books = train_user_book.groupby('user_id')['book_id'].apply(list).to_dict()

user_authors = {}
user_tags = {}
user_years = {}
for user, books in user_train_books.items():
    authors = set()
    tags = set()
    years = []
    for b in books:
        info = book_meta_dict.get(b, {})
        authors.add(str(info.get('author_id', 'unknown')))
        tags |= book_top_tags.get(b, set())
        years.append(int(info.get('published_year', median_year)))
    user_authors[user] = authors
    user_tags[user] = tags
    user_years[user] = years

# 7 Candidate generation

def cf_candidate_scores(user_books):
    scores = defaultdict(float)
    seen = set(user_books)
    for b in user_books:
        for nb, sim in book_neighbors.get(b, []):
            if nb not in seen:
                scores[nb] += sim
    return scores


def generate_candidates(user, positive_book=None):
    seen = set(user_train_books.get(user, []))
    scores = cf_candidate_scores(list(seen))
    cf_cands = [b for b, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:MAX_CF_CANDS]]

    cands = []
    cands.extend(cf_cands)
    for b in popular_books:
        if b not in seen:
            cands.append(b)
        if len(cands) >= (MAX_CF_CANDS + MAX_POP_CANDS):
            break

    if positive_book is not None and positive_book not in seen:
        cands.append(positive_book)

    out = []
    used = set()
    for b in cands:
        if b not in seen and b not in used:
            used.add(b)
            out.append(b)
    return out


def cf_feature_values(user_books, candidate_book):
    sims = []
    for b in user_books:
        for nb, sim in book_neighbors.get(b, []):
            if nb == candidate_book:
                sims.append(sim)
    if not sims:
        return 0.0, 0.0
    return float(max(sims)), float(sum(sims))


def build_feature_row(user, candidate_book):
    user_books = user_train_books.get(user, [])
    info = book_meta_dict.get(candidate_book, {})
    cand_author = str(info.get('author_id', 'unknown'))
    cand_year = int(info.get('published_year', median_year))
    cand_tags = book_top_tags.get(candidate_book, set())
    u_tags = user_tags.get(user, set())
    inter = len(cand_tags & u_tags)
    union = len(cand_tags | u_tags)
    tag_jacc = inter / union if union else 0.0
    cf_max, cf_sum = cf_feature_values(user_books, candidate_book)
    years = user_years.get(user, [median_year])
    min_year_gap = min(abs(cand_year - y) for y in years) if years else 0
    return {
        'user_id': user,
        'book_id': candidate_book,
        'user_num_books': len(user_books),
        'book_popularity': float(info.get('book_popularity', 0)),
        'book_popularity_log': float(info.get('book_popularity_log', 0)),
        'n_chapters': float(info.get('n_chapters', 0)),
        'published_year': float(cand_year),
        'author_match': 1.0 if cand_author in user_authors.get(user, set()) else 0.0,
        'tag_overlap_count': float(inter),
        'tag_jaccard': float(tag_jacc),
        'cf_sim_max': float(cf_max),
        'cf_sim_sum': float(cf_sum),
        'min_year_gap': float(min_year_gap),
    }


# 8 Build ML training data

train_t0 = time.time()
train_feature_rows = []
for i, row in enumerate(heldout.itertuples(index=False), start=1):
    user = row.user_id
    pos_book = row.target_book_id
    cands = generate_candidates(user, positive_book=pos_book)
    negs = [b for b in cands if b != pos_book][:MAX_NEGATIVES_PER_USER]
    selected = [pos_book] + negs
    labels = [1] + [0] * len(negs)
    for b, label in zip(selected, labels):
        feat = build_feature_row(user, b)
        feat['label'] = label
        train_feature_rows.append(feat)
    if i % 5000 == 0:
        print(f'Built features for {i:,} users')

train_df = pd.DataFrame(train_feature_rows)
feature_cols = [
    'user_num_books', 'book_popularity', 'book_popularity_log', 'n_chapters',
    'published_year', 'author_match', 'tag_overlap_count', 'tag_jaccard',
    'cf_sim_max', 'cf_sim_sum', 'min_year_gap'
]

print(f'Training rows: {len(train_df):,}, positive rate: {train_df.label.mean():.4f}')
print(f'Feature building done in {time.time() - train_t0:.1f}s')


# 9 Train ML model

model_t0 = time.time()
X = train_df[feature_cols].fillna(0)
y = train_df['label'].astype(int)
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

model_name = None
try:
    from xgboost import XGBClassifier
    model = XGBClassifier(
        n_estimators=120,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=SEED,
        n_jobs=2,
        reg_lambda=1.0,
        tree_method='hist'
    )
    model.fit(X_train, y_train)
    model_name = 'XGBoostClassifier'
except Exception as e:
    print('XGBoost unavailable, using HistGradientBoostingClassifier')
    print('Reason:', str(e))
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_depth=5,
        max_iter=120,
        random_state=SEED
    )
    model.fit(X_train, y_train)
    model_name = 'HistGradientBoostingClassifier'

if hasattr(model, 'predict_proba'):
    valid_scores = model.predict_proba(X_valid)[:, 1]
else:
    valid_scores = model.decision_function(X_valid)
auc = roc_auc_score(y_valid, valid_scores)
print(f'Model trained: {model_name}, valid ROC-AUC: {auc:.4f} in {time.time() - model_t0:.1f}s')

# 10 Recommend and evaluate


def get_scores(df_feats):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(df_feats[feature_cols].fillna(0))[:, 1]
    return model.decision_function(df_feats[feature_cols].fillna(0))


def recommend_for_user(user, top_k=TOP_K_EVAL):
    cands = generate_candidates(user, positive_book=None)
    rows = [build_feature_row(user, b) for b in cands]
    if not rows:
        return []
    cand_df = pd.DataFrame(rows)
    cand_df['score'] = get_scores(cand_df)
    cand_df = cand_df.sort_values(['score', 'book_popularity_log'], ascending=[False, False])
    return cand_df['book_id'].head(top_k).tolist()


def hit_at_k(actual, preds, k):
    return 1.0 if actual in preds[:k] else 0.0


def reciprocal_rank(actual, preds, k):
    for rank, item in enumerate(preds[:k], start=1):
        if item == actual:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(actual, preds, k):
    for rank, item in enumerate(preds[:k], start=1):
        if item == actual:
            return 1.0 / math.log2(rank + 1)
    return 0.0


eval_t0 = time.time()
eval_rows = []
sample_reco_rows = []
for i, row in enumerate(heldout.itertuples(index=False), start=1):
    user = row.user_id
    actual = row.target_book_id
    preds = recommend_for_user(user, top_k=TOP_K_EVAL)
    eval_rows.append({
        'user_id': user,
        'actual_book_id': actual,
        'predicted_top10': '|'.join(map(str, preds)),
        'hit_at_5': hit_at_k(actual, preds, 5),
        'hit_at_10': hit_at_k(actual, preds, 10),
        'mrr_at_10': reciprocal_rank(actual, preds, 10),
        'ndcg_at_10': ndcg_at_k(actual, preds, 10),
    })
    if i <= 200:
        for rank, rec in enumerate(preds, start=1):
            sample_reco_rows.append({
                'user_id': user,
                'rank': rank,
                'recommended_book_id': rec,
                'actual_book_id': actual,
            })
    if i % 5000 == 0:
        print(f'Evaluated {i:,} users')


eval_df = pd.DataFrame(eval_rows)
summary_df = pd.DataFrame([{
    'model_name': model_name,
    'users_evaluated': int(eval_df['user_id'].nunique()),
    'hit_at_5': eval_df['hit_at_5'].mean(),
    'hit_at_10': eval_df['hit_at_10'].mean(),
    'mrr_at_10': eval_df['mrr_at_10'].mean(),
    'ndcg_at_10': eval_df['ndcg_at_10'].mean(),
    'validation_roc_auc': auc,
    'total_runtime_seconds': round(time.time() - start_all, 2),
}])
print('Evaluation done in %.1fs' % (time.time() - eval_t0))
print('\nEvaluation summary:')
print(summary_df.round(4).to_string(index=False))


# 11 Continue-reading examples

cr_t0 = time.time()
merged = interactions.merge(chapters[['chapter_id', 'book_id', 'chapter_sequence_no']], on=['chapter_id', 'book_id'], how='left')
last_chapter = merged.groupby(['user_id', 'book_id'])['chapter_sequence_no'].max().reset_index()
continue_reading_examples = last_chapter.merge(book_meta[['book_id', 'max_chapter_no']], on='book_id', how='left')
continue_reading_examples['recommended_next_chapter_no'] = continue_reading_examples['chapter_sequence_no'] + 1
continue_reading_examples = continue_reading_examples[
    continue_reading_examples['recommended_next_chapter_no'] <= continue_reading_examples['max_chapter_no']
].head(500)
print(f'Continue-reading output ready in {time.time() - cr_t0:.1f}s')


# 12 Save outputs and download

summary_path = os.path.join(OUTPUT_DIR, 'evaluation_summary.csv')
detail_path = os.path.join(OUTPUT_DIR, 'evaluation_detailed.csv')
sample_path = os.path.join(OUTPUT_DIR, 'sample_recommendations.csv')
continue_path = os.path.join(OUTPUT_DIR, 'continue_reading_examples.csv')
report_path = os.path.join(OUTPUT_DIR, 'run_report.txt')

summary_df.to_csv(summary_path, index=False)
eval_df.to_csv(detail_path, index=False)
pd.DataFrame(sample_reco_rows).to_csv(sample_path, index=False)
continue_reading_examples.to_csv(continue_path, index=False)

with open(report_path, 'w') as f:
    f.write('Fast hybrid recommender with ML ranking model\n')
    f.write(f'Model used: {model_name}\n')
    f.write(f'Interactions shape: {interactions.shape}\n')
    f.write(f'Chapters shape: {chapters.shape}\n')
    f.write(f'Users evaluated: {heldout.user_id.nunique()}\n')
    f.write(f'Total runtime seconds: {time.time() - start_all:.2f}\n')
    f.write('\nEvaluation summary:\n')
    f.write(summary_df.round(6).to_string(index=False))
    f.write('\n\nFeature columns:\n')
    for c in feature_cols:
        f.write(f'- {c}\n')
    f.write('\n\nSpeed controls used:\n')
    f.write(json.dumps({
        'MAX_USERS_FOR_TRAIN_EVAL': MAX_USERS_FOR_TRAIN_EVAL,
        'MAX_CF_CANDS': MAX_CF_CANDS,
        'MAX_POP_CANDS': MAX_POP_CANDS,
        'MAX_NEGATIVES_PER_USER': MAX_NEGATIVES_PER_USER,
        'TOP_K_EVAL': TOP_K_EVAL,
    }, indent=2))

zip_path = '/content/reco_outputs1.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    for fn in os.listdir(OUTPUT_DIR):
        zf.write(os.path.join(OUTPUT_DIR, fn), arcname=fn)

print('\nOutputs saved in:', OUTPUT_DIR)
print('Files:', sorted(os.listdir(OUTPUT_DIR)))
print('Total runtime: %.1fs' % (time.time() - start_all))
files.download(zip_path)
