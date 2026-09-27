"""
Test inference: loads trained model and runs prediction on the full test set.
Outputs: output/matching_results.tsv and output/candidate_pairs.tsv
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
from src.preprocess import clean_address, clean_name

TEST_S1 = "student_resource/dataset/test/test_source1.tsv"
TEST_S2 = "student_resource/dataset/test/test_source2.tsv"
TEST_S3 = "student_resource/dataset/test/test_source3.tsv"
OUTPUT_DIR = "output"
MODEL_PATH = "output/model.pkl"

CHUNK_SIZE = 500_000
MAX_TOKEN_BUCKET = 300
TFIDF_TOP_K = 15
TFIDF_MIN_SCORE = 0.15


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["business_name"] = chunk["business_name"].fillna("")
    chunk["business_address"] = chunk["business_address"].fillna("")
    chunk["country"] = chunk["country"].fillna("Unknown")
    chunk["clean_name"] = chunk["business_name"].apply(clean_name)
    chunk["clean_addr"] = chunk["business_address"].apply(clean_address)
    return chunk


def run_inference():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    t0 = time.time()

    # ── Load model ─────────────────────────────────────────────────────────────
    print("=== Loading model ===")
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    clf = saved["clf"]
    threshold = saved["threshold"]
    print(f"  Threshold: {threshold:.3f}")

    # ── Build test index ───────────────────────────────────────────────────────
    print("\n=== Building test S2+S3 index ===")
    blocker = HybridBlockingIndex(
        max_token_bucket=MAX_TOKEN_BUCKET,
        tfidf_top_k=TFIDF_TOP_K,
        tfidf_min_score=TFIDF_MIN_SCORE,
    )
    s23_lookup = {}  # {entity_id: (clean_name, clean_addr, country, raw_name, raw_addr)}

    for path in [TEST_S2, TEST_S3]:
        print(f"  Indexing {path}...")
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=CHUNK_SIZE):
            chunk = clean_chunk(chunk)
            eids = chunk["entity_id"].tolist()
            rnames = chunk["business_name"].tolist()
            raddrs = chunk["business_address"].tolist()
            cnames = chunk["clean_name"].tolist()
            caddrs = chunk["clean_addr"].tolist()
            ctries = chunk["country"].tolist()

            blocker.add_records(eids, rnames, raddrs, cnames, caddrs, ctries)
            for eid, cn, ca, ct, rn, ra in zip(eids, cnames, caddrs, ctries, rnames, raddrs):
                s23_lookup[eid] = (cn, ca, ct, rn, ra)

    blocker.prune_token_index()
    print("  Building TF-IDF...")
    blocker.build_tfidf()
    print(f"  Done. Lookup size: {len(s23_lookup):,}")

    # ── Stream test S1 ─────────────────────────────────────────────────────────
    print("\n=== Inference on test S1 ===")
    cand_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
    match_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")

    total = 0
    with open(cand_path, "w", encoding="utf-8") as fc, \
         open(match_path, "w", encoding="utf-8") as fm:

        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        fm.write("source1_entity_id\tmatched_entity_ids\n")

        # Smaller chunks than before (100k -> 50k): the batched TF-IDF query
        # builds a (country_subset x chunk_size) sparse matrix per country
        # group, so a huge chunk size trades one kind of slowdown for a
        # memory spike. 50k keeps that bounded while still getting the
        # batching win over the old one-row-at-a-time calls.
        for chunk in pd.read_csv(TEST_S1, sep="\t", dtype=str, chunksize=50_000):
            chunk = clean_chunk(chunk)
            eids = chunk["entity_id"].tolist()
            rnames = chunk["business_name"].tolist()
            raddrs = chunk["business_address"].tolist()
            cnames = chunk["clean_name"].tolist()
            caddrs = chunk["clean_addr"].tolist()
            ctries = chunk["country"].tolist()

            cand_lists = blocker.get_candidates_batch(rnames, raddrs, cnames, caddrs, ctries)

            cand_buf, match_buf = [], []
            for s1_id, s1_rn, s1_ra, s1_cn, s1_ca, s1_ct, cands in zip(
                eids, rnames, raddrs, cnames, caddrs, ctries, cand_lists
            ):
                valid = [c for c in cands if c in s23_lookup]
                cand_buf.append(f"{s1_id}\t{','.join(valid)}")

                if not valid:
                    match_buf.append(f"{s1_id}\t")
                    continue

                feat_matrix = []
                for c in valid:
                    cn, ca, ct, rn, ra = s23_lookup[c]
                    feats = compute_pair_features(
                        s1_cn, s1_ca, s1_ct,
                        cn, ca, ct, c,
                        s1_rn, s1_ra, rn, ra,
                    )
                    feat_matrix.append(list(feats.values()))

                probs = clf.predict_proba(np.array(feat_matrix, dtype=np.float32))[:, 1]
                matches = [c for c, p in zip(valid, probs) if p >= threshold]
                match_buf.append(f"{s1_id}\t{','.join(matches)}")

            fc.write("\n".join(cand_buf) + "\n")
            fm.write("\n".join(match_buf) + "\n")
            total += len(chunk)
            print(f"  {total:,} S1 records processed...")

    print(f"\n=== Done! {time.time()-t0:.1f}s ===")
    print(f"  {match_path}")
    print(f"  {cand_path}")


if __name__ == "__main__":
    run_inference()