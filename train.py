"""
Full end-to-end pipeline:
  1. Load ALL training data (2.2M S1, full S2+S3)
  2. Build hybrid blocking index (numeric token + Latin token + TF-IDF)
  3. Mine positive + negative pairs
  4. Train LightGBM classifier with proper class balancing
  5. Validate on held-out 10% split & tune threshold
  6. Save model to output/model.pkl

Key improvements over 0.65 baseline:
  - Full dataset (not 30k sample)
  - Cross-script blocking via numeric address tokens
  - 37 features including cross-script signals
  - LightGBM (faster, better)
  - Proper 10% val split for threshold tuning
"""
import collections
import gc
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("code/business_entity_resolution"))

from src.blocking import HybridBlockingIndex
from src.features import compute_pair_features
from src.metric import compute_macro_f05
from src.preprocess import clean_address, clean_name

# ─── Paths ────────────────────────────────────────────────────────────────────
TRAIN_S1 = "student_resource/dataset/train/train_source1.tsv"
TRAIN_S2 = "student_resource/dataset/train/train_source2.tsv"
TRAIN_S3 = "student_resource/dataset/train/train_source3.tsv"
TRAIN_GT = "student_resource/dataset/train/train_ground_truth.tsv"
OUTPUT_DIR = "output"

CHUNK_SIZE = 500_000
VAL_FRACTION = 0.10

# Blocking params
MAX_TOKEN_BUCKET = 300
TFIDF_TOP_K = 15
TFIDF_MIN_SCORE = 0.15


# ─── Loaders ──────────────────────────────────────────────────────────────────

def load_df(path: str, nrows=None) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows)
    df["business_name"] = df["business_name"].fillna("")
    df["business_address"] = df["business_address"].fillna("")
    df["country"] = df["country"].fillna("Unknown")
    df["clean_name"] = df["business_name"].apply(clean_name)
    df["clean_addr"] = df["business_address"].apply(clean_address)
    return df


def load_df_chunked(path: str) -> pd.DataFrame:
    parts = []
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
        chunk["business_name"] = chunk["business_name"].fillna("")
        chunk["business_address"] = chunk["business_address"].fillna("")
        chunk["country"] = chunk["country"].fillna("Unknown")
        chunk["clean_name"] = chunk["business_name"].apply(clean_name)
        chunk["clean_addr"] = chunk["business_address"].apply(clean_address)
        parts.append(chunk)
    return pd.concat(parts, ignore_index=True)


def load_gt(path: str) -> dict:
    gt = {}
    for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
        for _, row in chunk.iterrows():
            s1_id = row["source1_entity_id"]
            raw = str(row.get("matched_entity_ids", ""))
            gt[s1_id] = (
                set(x.strip() for x in raw.split(",") if x.strip())
                if raw and raw != "nan"
                else set()
            )
    return gt


def build_lookup(df: pd.DataFrame) -> dict:
    """Returns {entity_id: (clean_name, clean_addr, country, raw_name, raw_addr)}"""
    return {
        row["entity_id"]: (
            row["clean_name"],
            row["clean_addr"],
            row["country"],
            row["business_name"],
            row["business_address"],
        )
        for _, row in df.iterrows()
    }


# ─── Pair mining ──────────────────────────────────────────────────────────────

def mine_pairs(s1_df, s23_lookup, gt_dict, blocker, n_neg_per_pos=5, batch_size=5000):
    """
    IMPORTANT: pairs are mined ONLY from what the blocker actually retrieves.

    The previous version did:
        all_cands = set(cands) | (true_m & set(s23_lookup.keys()))
    which force-injected every ground-truth match into the candidate set,
    even ones the blocker never would have found. That trains the classifier
    on positives with weak/absent blocking signal (features it'll never see
    a matching pattern for at inference, since predict.py has no such
    injection), which loosens the decision boundary and hurts precision —
    exactly what F_0.5 punishes hardest. Any true match the blocker misses
    is a blocking-recall problem, not something to paper over here; it
    should show up in the Step 5 recall measurement instead.
    """
    X_rows, y_labels = [], []
    n = len(s1_df)

    for start in range(0, n, batch_size):
        batch = s1_df.iloc[start : start + batch_size]
        eids = batch["entity_id"].tolist()
        rnames = batch["business_name"].tolist()
        raddrs = batch["business_address"].tolist()
        cnames = batch["clean_name"].tolist()
        caddrs = batch["clean_addr"].tolist()
        ctries = batch["country"].tolist()

        cand_lists = blocker.get_candidates_batch(rnames, raddrs, cnames, caddrs, ctries)

        for s1_id, s1_rn, s1_ra, s1_cn, s1_ca, s1_ct, cands in zip(
            eids, rnames, raddrs, cnames, caddrs, ctries, cand_lists
        ):
            true_m = gt_dict.get(s1_id, set())
            cands = [c for c in cands if c in s23_lookup]  # no GT injection

            positives = [c for c in cands if c in true_m]
            negatives = [c for c in cands if c not in true_m]

            max_neg = max(n_neg_per_pos * max(len(positives), 1), 3)
            if len(negatives) > max_neg:
                rng = np.random.default_rng(abs(hash(s1_id)) % (2**31))
                negatives = list(rng.choice(negatives, size=max_neg, replace=False))

            for cand_id in positives + negatives:
                cn, ca, ct, rn, ra = s23_lookup[cand_id]
                feats = compute_pair_features(
                    s1_cn, s1_ca, s1_ct,
                    cn, ca, ct, cand_id,
                    s1_rn, s1_ra, rn, ra,
                )
                X_rows.append(list(feats.values()))
                y_labels.append(1 if cand_id in true_m else 0)

    return np.array(X_rows, dtype=np.float32), np.array(y_labels, dtype=np.int32)


# ─── Classifier ───────────────────────────────────────────────────────────────

def train_lgbm(X, y, scale_pos_weight=1.0):
    """
    scale_pos_weight defaults to 1.0 — mine_pairs() already balances the
    pos:neg ratio via negative downsampling (n_neg_per_pos). Computing
    scale_pos_weight = neg/pos from THAT already-downsampled ratio and
    passing it to LightGBM double-corrects: it pushes the model to favor
    positives on top of a ratio you already fixed, inflating recall at
    precision's expense — the wrong direction for an F_0.5 target that
    weights precision 2x. If you want more positive-leaning behavior,
    tune n_neg_per_pos in mine_pairs (try 3 vs 8 vs 15) and compare val
    F_0.5, rather than reweighting here on top of it.
    """
    import lightgbm as lgb
    pos = int(y.sum())
    neg = len(y) - pos
    print(f"  LightGBM: {len(X):,} pairs | {pos} pos | {neg} neg (ratio {neg/max(pos,1):.1f}:1) | scale_pos_weight={scale_pos_weight}")
    clf = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.04,
        num_leaves=63,
        max_depth=8,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(X, y)
    return clf


def optimize_threshold(clf, val_pairs, gt_dict, val_s1_ids):
    if not val_pairs:
        return 0.5
    X_val = np.array([p[0] for p in val_pairs], dtype=np.float32)
    meta_val = [p[1] for p in val_pairs]
    probs = clf.predict_proba(X_val)[:, 1]

    by_s1 = collections.defaultdict(list)
    for (s1_id, cid), p in zip(meta_val, probs):
        by_s1[s1_id].append((cid, p))

    gt_sub = {s1: gt_dict.get(s1, set()) for s1 in val_s1_ids}
    best_score, best_thresh = -1.0, 0.5

    for thresh in np.arange(0.30, 0.95, 0.025):
        preds = {
            s1: {cid for cid, p in by_s1.get(s1, []) if p >= thresh}
            for s1 in val_s1_ids
        }
        score = compute_macro_f05(gt_sub, preds)
        if score > best_score:
            best_score = score
            best_thresh = thresh

    print(f"  Best threshold: {best_thresh:.3f} → val F_0.5 = {best_score:.4f}")
    return best_thresh


# ─── Main ─────────────────────────────────────────────────────────────────────

def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    print("=== Step 1: Load S1 ===")
    s1_df = load_df(TRAIN_S1)
    print(f"  S1: {len(s1_df):,}")

    np.random.seed(42)
    val_mask = np.random.rand(len(s1_df)) < VAL_FRACTION
    val_s1_ids = list(s1_df["entity_id"][val_mask])
    train_s1_ids = set(s1_df["entity_id"][~val_mask])
    print(f"  Train: {len(train_s1_ids):,} | Val: {len(val_s1_ids):,}")

    print("\n=== Step 2: Load GT ===")
    gt_dict = load_gt(TRAIN_GT)
    print(f"  GT: {len(gt_dict):,}")

    print("\n=== Step 3: Load S2 + S3 ===")
    s2_df = load_df_chunked(TRAIN_S2)
    print(f"  S2: {len(s2_df):,}")
    s3_df = load_df_chunked(TRAIN_S3)
    print(f"  S3: {len(s3_df):,}")

    s23_df = pd.concat([s2_df, s3_df], ignore_index=True).drop_duplicates("entity_id")
    print(f"  S2+S3: {len(s23_df):,}")
    s23_lookup = build_lookup(s23_df)

    print("\n=== Step 4: Build blocking index ===")
    blocker = HybridBlockingIndex(
        max_token_bucket=MAX_TOKEN_BUCKET,
        tfidf_top_k=TFIDF_TOP_K,
        tfidf_min_score=TFIDF_MIN_SCORE,
    )
    blocker.add_records(
        s23_df["entity_id"].tolist(),
        s23_df["business_name"].tolist(),
        s23_df["business_address"].tolist(),
        s23_df["clean_name"].tolist(),
        s23_df["clean_addr"].tolist(),
        s23_df["country"].tolist(),
    )
    blocker.prune_token_index()
    print("  Building TF-IDF...")
    blocker.build_tfidf()
    print("  Index ready.")

    del s2_df, s3_df, s23_df
    gc.collect()

    print("\n=== Step 5: Measure blocking recall on val sample ===")
    val_sample = s1_df[s1_df["entity_id"].isin(set(val_s1_ids[:2000]))].reset_index(drop=True)
    rn, rd = 0, 0
    for _, row in val_sample.iterrows():
        true_m = gt_dict.get(row["entity_id"], set())
        if not true_m:
            continue
        cands = set(blocker.get_candidates(
            row["business_name"], row["business_address"],
            row["clean_name"], row["clean_addr"], row["country"]
        ))
        rn += len(true_m & cands)
        rd += len(true_m)
    print(f"  Blocking recall: {rn/rd:.4f} ({rn}/{rd})")

    print("\n=== Step 6: Mine train pairs ===")
    train_df = s1_df[s1_df["entity_id"].isin(train_s1_ids)].reset_index(drop=True)
    X_train, y_train = mine_pairs(train_df, s23_lookup, gt_dict, blocker)
    print(f"  Pairs: {len(X_train):,} | Pos: {int(y_train.sum())} ({y_train.mean():.2%})")

    print("\n=== Step 7: Mine val pairs ===")
    # Same fix as mine_pairs: no GT injection. The threshold you tune here
    # must be tuned against exactly the candidate sets predict.py will see
    # at inference — injecting GT matches that the blocker couldn't find
    # made validation artificially easy and would have picked a threshold
    # miscalibrated for real (blocker-limited) test-time candidates.
    val_df = s1_df[s1_df["entity_id"].isin(set(val_s1_ids))].reset_index(drop=True)
    val_pairs = []
    VAL_BATCH = 5000
    for start in range(0, len(val_df), VAL_BATCH):
        batch = val_df.iloc[start : start + VAL_BATCH]
        eids = batch["entity_id"].tolist()
        rnames = batch["business_name"].tolist()
        raddrs = batch["business_address"].tolist()
        cnames = batch["clean_name"].tolist()
        caddrs = batch["clean_addr"].tolist()
        ctries = batch["country"].tolist()

        cand_lists = blocker.get_candidates_batch(rnames, raddrs, cnames, caddrs, ctries)

        for s1_id, s1_rn, s1_ra, s1_cn, s1_ca, s1_ct, cands in zip(
            eids, rnames, raddrs, cnames, caddrs, ctries, cand_lists
        ):
            for cid in cands:
                if cid not in s23_lookup:
                    continue
                cn, ca, ct, rn_c, ra_c = s23_lookup[cid]
                feats = compute_pair_features(
                    s1_cn, s1_ca, s1_ct,
                    cn, ca, ct, cid,
                    s1_rn, s1_ra, rn_c, ra_c,
                )
                val_pairs.append((list(feats.values()), (s1_id, cid)))
    print(f"  Val pairs: {len(val_pairs):,}")

    print("\n=== Step 8: Train LightGBM ===")
    clf = train_lgbm(X_train, y_train)

    print("\n=== Step 9: Optimize threshold ===")
    best_thresh = optimize_threshold(clf, val_pairs, gt_dict, val_s1_ids)

    print("\n=== Step 10: Save ===")
    model_path = os.path.join(OUTPUT_DIR, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump({"clf": clf, "threshold": best_thresh}, f)
    print(f"  Saved to {model_path}")
    print(f"  Total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    run()