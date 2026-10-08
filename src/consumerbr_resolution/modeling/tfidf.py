import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from consumerbr_resolution.config import (
    TFIDF_LOWERCASE, TFIDF_MAX_DF, TFIDF_MAX_FEATURES, TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE, TFIDF_STRIP_ACCENTS, TFIDF_SUBLINEAR_TF,
)


def create_tfidf_vectorizer():
    return TfidfVectorizer(
        ngram_range=TFIDF_NGRAM_RANGE, min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF, max_features=TFIDF_MAX_FEATURES,
        sublinear_tf=TFIDF_SUBLINEAR_TF, strip_accents=TFIDF_STRIP_ACCENTS,
        lowercase=TFIDF_LOWERCASE, dtype=np.float32,
    )


def load_split(connection, source, membership, split):
    if split not in ("train", "validation", "test"):
        raise ValueError("Only included temporal partitions can be loaded.")
    return connection.execute(
        "SELECT f.record_id, f.complaint_id, f.company, f.opening_date, "
        "f.target_resolved, f.complaint_text FROM read_parquet(?) f "
        "JOIN read_parquet(?) m USING (record_id) "
        "WHERE m.split = ? ORDER BY f.record_id",
        [str(source), str(membership), split],
    ).df()
