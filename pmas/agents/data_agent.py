# pmas/agents/data_agent.py
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import io
import requests

class DataAgent:
    """
    Robust loader for Pima Indians Diabetes.
    Tries multiple known mirrors. Handles files with or without headers.
    If a downloaded CSV does not resemble the Pima dataset (9 canonical columns),
    it will skip that URL and try the next one.
    Produces train/test splits and a binary sensitive attribute 'age_group'.
    """
    CANDIDATE_URLS = [
        # common mirrors (try these in order)
        "https://raw.githubusercontent.com/selva86/datasets/master/PimaIndiansDiabetes.csv",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
        # fallback raw UCI (may return headerless)
        "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    ]
    LOCAL_PATH = Path("data/pima_diabetes.csv")
    AGE_CUTOFF = 30

    # canonical column names for Pima dataset
    CANONICAL_COLS = ["pregnant","glucose","pressure","triceps","insulin","mass","pedigree","age","y"]

    def __init__(self, test_size=0.30, seed=42, force_redownload=False):
        self.test_size = test_size
        self.seed = seed
        self.force_redownload = force_redownload

    def _read_csv_from_bytes(self, content: bytes):
        """
        Try to read CSV content with and without header.
        Returns a DataFrame or raises.
        """
        # try read with pandas (let pandas infer header)
        try:
            df = pd.read_csv(io.BytesIO(content))
            return df
        except Exception:
            # try to read as headerless (comma separated)
            df = pd.read_csv(io.BytesIO(content), header=None)
            return df

    def _download_if_missing(self):
        # if a good local file exists and no force_redownload, return it
        if self.LOCAL_PATH.exists() and not self.force_redownload:
            try:
                df = pd.read_csv(self.LOCAL_PATH)
                # quick sanity: must contain canonical cols or 9 columns
                cols = [c.lower() for c in df.columns]
                if set(self.CANONICAL_COLS).issubset(set(cols)) or len(cols) == 9:
                    return df
                # otherwise treat as bad and redownload
            except Exception:
                try:
                    self.LOCAL_PATH.unlink()
                except Exception:
                    pass

        last_exc = None
        for url in self.CANDIDATE_URLS:
            try:
                print(f"Attempting to download Pima CSV from: {url}")
                r = requests.get(url, timeout=20)
                r.raise_for_status()
                content = r.content
                df = self._read_csv_from_bytes(content)
                # normalize column names to lowercase strings for checking
                cols = [str(c).strip().lower() for c in df.columns]
                # Accept if it already contains canonical column names
                if set(self.CANONICAL_COLS).issubset(set(cols)):
                    # good
                    df.columns = cols
                    os.makedirs(self.LOCAL_PATH.parent, exist_ok=True)
                    df.to_csv(self.LOCAL_PATH, index=False)
                    return df
                # Accept if it has exactly 9 columns (likely headerless pima .data)
                if len(cols) == 9:
                    # assign canonical names
                    print("Downloaded CSV has 9 columns; assigning canonical Pima column names.")
                    df.columns = list(self.CANONICAL_COLS)
                    os.makedirs(self.LOCAL_PATH.parent, exist_ok=True)
                    df.to_csv(self.LOCAL_PATH, index=False)
                    return df
                # If columns look entirely different (e.g., v1..v34 or class label for different dataset),
                # skip this URL and try the next one.
                print(f"Downloaded CSV from {url} did not match expected Pima shape. Columns: {cols[:8]}... (len={len(cols)}) — skipping.")
                continue
            except Exception as e:
                last_exc = e
                print(f"Download/read failed for {url}: {e}")
                continue
        raise RuntimeError("Failed to download a suitable Pima dataset from known mirrors.") from last_exc

    def _normalize_and_prepare(self, df: pd.DataFrame):
        df = df.copy()
        # Lowercase columns
        df.columns = [str(c).strip().lower() for c in df.columns]

        # handle common name variants
        rename_map = {}
        if "diabetes" in df.columns:
            rename_map["diabetes"] = "y"
        if "outcome" in df.columns:
            rename_map["outcome"] = "y"
        if "bmi" in df.columns and "mass" not in df.columns:
            rename_map["bmi"] = "mass"
        if "skin" in df.columns and "triceps" not in df.columns:
            rename_map["skin"] = "triceps"
        if rename_map:
            df = df.rename(columns=rename_map)

        # Final check
        if not set(self.CANONICAL_COLS).issubset(set(df.columns)):
            raise RuntimeError(f"Pima dataset missing expected columns after normalization. Found: {list(df.columns)}")

        # keep only canonical columns and convert types
        df = df[self.CANONICAL_COLS].copy()
        # ensure numeric types
        for c in self.CANONICAL_COLS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().reset_index(drop=True)

        # create age_group sensitive attribute
        df["age_group"] = (df["age"] >= self.AGE_CUTOFF).astype(int)
        df["y"] = df["y"].astype(int)
        return df

    def perform(self, action, params, state=None):
        if action == "load":
            df = self._download_if_missing()
            df = self._normalize_and_prepare(df)
            train, test = train_test_split(df, test_size=self.test_size, random_state=self.seed, stratify=df["y"])
            return {"train_df": train.reset_index(drop=True), "test_df": test.reset_index(drop=True)}
        raise NotImplementedError(action)
