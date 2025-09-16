# pmas/agents/data_agent.py
"""
DataAgent for FairLens demo
Supports:
  - Pima Indians Diabetes
  - Adult Income dataset
Works offline (via cached CSV in data/) or online (via mirrors).
"""

from pathlib import Path
import pandas as pd
import logging

LOG = logging.getLogger(__name__)


class DataAgent:
    # -----------------
    # Known sources
    # -----------------
    PIMA_URLS = [
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "https://raw.githubusercontent.com/selva86/datasets/master/PimaIndiansDiabetes.csv",
    ]
    PIMA_COLS = [
        "pregnant", "glucose", "pressure", "triceps", "insulin",
        "mass", "pedigree", "age", "y"
    ]

    ADULT_URLS = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://raw.githubusercontent.com/selva86/datasets/master/Adult.csv",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/adult.csv",
    ]
    ADULT_COLS = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race", "sex",
        "capital_gain", "capital_loss", "hours_per_week", "native_country", "y"
    ]

    def __init__(self):
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    # -----------------
    # Main entry
    # -----------------
    def perform(self, action: str, params: dict = None, state: dict = None):
        params = params or {}
        dataset = params.get("dataset", "Pima")
        use_cache_only = params.get("use_cache_only", False)

        if action == "load":
            if str(dataset).lower().startswith("pima"):
                return self._load_pima(use_cache_only)
            else:
                return self._load_adult(use_cache_only)

        return {}

    # -----------------
    # Pima loader
    # -----------------
    def _load_pima(self, use_cache_only=False) -> pd.DataFrame:
        cache_path = self.data_dir / "pima_diabetes.csv"

        if cache_path.exists():
            df = pd.read_csv(cache_path)
            return self._normalize_pima(df)

        if use_cache_only:
            raise RuntimeError("Pima dataset not found. Please add data/pima_diabetes.csv.")

        last_exc = None
        for url in self.PIMA_URLS:
            try:
                df = pd.read_csv(url, header=None)
                if df.shape[1] == 9:
                    df.columns = self.PIMA_COLS
                df = self._normalize_pima(df)
                df.to_csv(cache_path, index=False)
                return df
            except Exception as e:
                last_exc = e

        raise RuntimeError(f"Failed to download Pima dataset. Last error: {last_exc}")

    def _normalize_pima(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.shape[1] >= 9:
            df = df.iloc[:, :9]
            df.columns = self.PIMA_COLS
        for c in df.columns[:-1]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["y"] = df["y"].astype(int)
        df = df.dropna().reset_index(drop=True)
        return df

    # -----------------
    # Adult loader
    # -----------------
    def _load_adult(self, use_cache_only=False) -> pd.DataFrame:
        cache_path = self.data_dir / "adult.csv"

        if cache_path.exists():
            return self._parse_adult(cache_path)

        if use_cache_only:
            raise RuntimeError("Adult dataset not found. Please add data/adult.csv.")

        last_exc = None
        for url in self.ADULT_URLS:
            try:
                df = pd.read_csv(url, header=None)
                return self._normalize_adult(df, cache_path)
            except Exception as e:
                last_exc = e

        raise RuntimeError(f"Failed to download Adult dataset. Last error: {last_exc}")

    def _parse_adult(self, path: Path) -> pd.DataFrame:
        try:
            df = pd.read_csv(path, header=None)
            return self._normalize_adult(df, None)
        except Exception:
            df = pd.read_csv(path)  # maybe has headers
            return self._normalize_adult(df, None)

    def _normalize_adult(self, df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
        if df.shape[1] >= 15:
            df = df.iloc[:, :15]
            df.columns = self.ADULT_COLS

        for c in df.select_dtypes(include="object").columns:
            df[c] = df[c].astype(str).str.strip()
            df[c] = df[c].replace("?", pd.NA)

        if "y" not in df.columns:
            df.columns = list(df.columns[:-1]) + ["y"]

        df["y"] = df["y"].map(
            lambda v: 1 if str(v).strip().startswith(">50K") else 0
        )

        df = df.dropna().reset_index(drop=True)

        if cache_path:
            df.to_csv(cache_path, index=False)

        return df
