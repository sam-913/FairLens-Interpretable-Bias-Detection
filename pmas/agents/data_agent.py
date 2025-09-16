# pmas/agents/data_agent.py
"""
DataAgent: loads datasets used by the P-MAS demo.
Supports:
 - Pima Indians Diabetes (small / fast)
 - Adult Income (larger; will try multiple mirrors and save a local cache)
This version is robust for offline/Streamlit Cloud deployment.
"""

from pathlib import Path
import os
import pandas as pd
import logging
from typing import Optional

LOG = logging.getLogger(__name__)

class DataAgent:
    # Pima canonical column names (used by other agents)
    PIMA_COLS = ["pregnant","glucose","pressure","triceps","insulin","mass","pedigree","age","class"]

    # Adult dataset mirrors to try (order matters)
    ADULT_URLS = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://raw.githubusercontent.com/selva86/datasets/master/Adult.csv",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult.csv",
    ]
    ADULT_COLS = [
        "age","workclass","fnlwgt","education","education_num","marital_status",
        "occupation","relationship","race","sex","capital_gain","capital_loss",
        "hours_per_week","native_country","y"
    ]

    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def perform(self, action: str, params: dict = None, state: dict = None):
        """
        action: 'load' or 'prepare' (the orchestrator uses 'load' semantics)
        params: {"dataset": "Pima" or "Adult", "use_cache_only": bool}
        """
        params = params or {}
        dataset = params.get("dataset", params.get("dataset_choice", "Pima"))
        use_cache_only = params.get("use_cache_only", False)

        if action == "load":
            if str(dataset).lower().startswith("pima"):
                return self._load_pima(use_cache_only=use_cache_only)
            else:
                return self._load_adult(use_cache_only=use_cache_only)

        # default behaviour for other orchestrator calls: return nothing
        return {}

    # ---------------------------
    # Pima loader
    # ---------------------------
    def _load_pima(self, use_cache_only: bool = False) -> pd.DataFrame:
        """
        Download or load cached Pima Indians Diabetes dataset and normalize columns.
        Accepts either a cached csv at data/pima_diabetes.csv or tries known mirrors.
        """
        cached = self.data_dir / "pima_diabetes.csv"
        # if cached exists, use it
        if cached.exists():
            df = pd.read_csv(cached)
            df = self._normalize_pima(df)
            return df

        if use_cache_only:
            raise RuntimeError("Pima dataset not found in data/pima_diabetes.csv (use_cache_only=True).")

        # try known mirror (jbrownlee)
        mirrors = [
            "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
            "https://raw.githubusercontent.com/selva86/datasets/master/PimaIndiansDiabetes.csv"
        ]
        last_exc = None
        for url in mirrors:
            try:
                df = pd.read_csv(url, header=None)
                # if it's the jbrownlee raw data (no header, 9 cols) assign canonical names
                if df.shape[1] == 9:
                    df.columns = self.PIMA_COLS
                else:
                    # if selva86 variant has named columns, try that
                    try:
                        df = pd.read_csv(url)
                    except Exception:
                        pass
                df = self._normalize_pima(df)
                # save cache
                try:
                    df.to_csv(cached, index=False)
                except Exception:
                    pass
                return df
            except Exception as e:
                last_exc = e
                continue

        raise RuntimeError(f"Pima dataset could not be downloaded: {last_exc}")

    def _normalize_pima(self, df: pd.DataFrame) -> pd.DataFrame:
        # ensure canonical columns exist
        cols = list(df.columns)
        if set(self.PIMA_COLS).issubset(set(cols)):
            # ok
            df = df[self.PIMA_COLS].copy()
        else:
            # try to handle variations (e.g., extra columns from some mirrors)
            # If the last column is the class label and there are 9 cols, map by position
            if df.shape[1] >= 9:
                df = df.iloc[:, :9]
                df.columns = self.PIMA_COLS
            else:
                raise RuntimeError(f"Pima dataset missing expected columns. Found: {list(df.columns)}")
        # drop rows with NaNs if any, and coerce numeric columns
        for c in ["pregnant","glucose","pressure","triceps","insulin","mass","pedigree","age"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().reset_index(drop=True)
        # ensure class is integer 0/1 (some sources use '1'/'0' or strings)
        df["class"] = df["class"].astype(int)
        return df

    # ---------------------------
    # Adult loader (robust)
    # ---------------------------
    def _load_adult(self, use_cache_only: bool = False) -> pd.DataFrame:
        """
        Robust Adult loader:
         - prefer existing local cache `data/adult.csv`
         - otherwise try a list of mirrors (ADULT_URLS)
         - if all fail and use_cache_only=True, raise informative error
        """
        cache_path = self.data_dir / "adult.csv"
        # 1) cache-first
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path, header=None)
                # if matches expected shape, assign canonical names
                if df.shape[1] == len(self.ADULT_COLS):
                    df.columns = self.ADULT_COLS
                else:
                    # attempt reading with header (selva86 variant)
                    try:
                        df = pd.read_csv(cache_path)
                    except Exception:
                        pass
                df = self._sanitize_adult(df)
                return df
            except Exception as e:
                LOG.warning("Failed to parse cached adult.csv: %s", e)
                # fallthrough to try mirrors or raise if cache_only

        if use_cache_only:
            raise RuntimeError(
                "Adult dataset not found in repo. Please add data/adult.csv for offline runs."
            )

        # 2) try mirrors
        last_exc = None
        for url in self.ADULT_URLS:
            try:
                # read raw first
                df_try = pd.read_csv(url, header=None)
                # if correct width, set column names
                if df_try.shape[1] == len(self.ADULT_COLS):
                    df_try.columns = self.ADULT_COLS
                else:
                    # try reading with header fallback
                    try:
                        df_try = pd.read_csv(url)
                    except Exception:
                        # if still odd, continue to next mirror
                        pass
                # sanitize
                df_try = self._sanitize_adult(df_try)
                # save a cache copy (best-effort)
                try:
                    os.makedirs(self.data_dir, exist_ok=True)
                    df_try.to_csv(cache_path, index=False)
                except Exception:
                    pass
                return df_try
            except Exception as e:
                last_exc = e
                continue

        # 3) all mirrors failed
        raise RuntimeError(
            "Failed to download the Adult Income dataset from known mirrors. "
            "Please manually download and save it to data/adult.csv. "
            f"Last exception: {last_exc}"
        )

    def _sanitize_adult(self, df: pd.DataFrame) -> pd.DataFrame:
        # If the dataset has an extra unnamed column or trailing comma, drop empty columns
        # Trim whitespace from string columns and normalize the label column to binary 0/1
        # Attempt to ensure we have the expected columns; if not, try best-effort mapping.
        if df.shape[1] == len(self.ADULT_COLS):
            df.columns = self.ADULT_COLS
        else:
            # If dataset has header with different names, try to align 'y' or 'income' column
            cols = list(df.columns)
            # common label names:
            label_candidates = [c for c in cols if str(c).lower() in ("income","y","class","target")]
            if label_candidates:
                label_col = label_candidates[0]
                df = df.rename(columns={label_col: "y"})
            # if numeric columns exist, we keep as-is; else try to coerce
        # strip whitespace for object types
        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype(str).str.strip()
            # also convert '?' to NaN
            df[c] = df[c].replace("?", pd.NA)

        # normalize label column to 0/1 if present
        if "y" in df.columns:
            df["y"] = df["y"].map(lambda v: 1 if str(v).strip().endswith(">50K") or str(v).strip().endswith(">50K.") or str(v).strip().endswith("1") else 0)
        else:
            # fallback: if last column looks like the label
            df.columns = list(df.columns[:-1]) + ["y"]
            df["y"] = df["y"].map(lambda v: 1 if str(v).strip().endswith(">50K") or str(v).strip().endswith(">50K.") else 0)

        df = df.dropna().reset_index(drop=True)
        return df
