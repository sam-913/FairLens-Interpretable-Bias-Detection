# pmas/agents/data_agent.py
import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


class DataAgent:
    """
    DataAgent for loading and preparing datasets.
    Supports:
      - Pima Diabetes dataset
      - Adult Income dataset
    """

    LOCAL_PATH = Path("data")
    AGE_CUTOFF = 30

    # Canonical Pima column names
    PIMA_CANONICAL = [
        "pregnant", "glucose", "pressure", "triceps", "insulin",
        "mass", "pedigree", "age", "y"
    ]
    PIMA_URLS = [
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "https://raw.githubusercontent.com/selva86/datasets/master/PimaIndiansDiabetes.csv",
        "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv",
    ]

    # Adult dataset
    ADULT_URLS = [
        "https://raw.githubusercontent.com/selva86/datasets/master/Adult.csv",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
    ]
    ADULT_LOCAL = LOCAL_PATH / "adult.csv"

    def __init__(self, test_size=0.30, seed=42):
        self.test_size = test_size
        self.seed = seed
        os.makedirs(self.LOCAL_PATH, exist_ok=True)

    # -------------------------
    # helper: download CSV
    # -------------------------
    def _download_csv(self, url, timeout=12):
        import requests, io
        r = requests.get(url, timeout=timeout)
        r.raise_for_status()
        content = r.content
        try:
            return pd.read_csv(io.BytesIO(content))
        except Exception:
            return pd.read_csv(io.BytesIO(content), header=None)

    # -------------------------
    # PIMA LOADER
    # -------------------------
    def _load_pima(self):
        for url in self.PIMA_URLS:
            try:
                df = self._download_csv(url)
                # normalize headers
                df.columns = [str(c).strip().lower() for c in df.columns]

                # Case 1: proper headers
                if set(self.PIMA_CANONICAL).issubset(set(df.columns)):
                    df = df[self.PIMA_CANONICAL]
                    df = df.dropna().reset_index(drop=True)
                    df["y"] = df["y"].astype(int)
                    return df

                # Case 2: headerless numeric data
                if df.shape[1] == 9:
                    df.columns = self.PIMA_CANONICAL
                    df = df.dropna().reset_index(drop=True)
                    df["y"] = df["y"].astype(int)
                    return df
            except Exception:
                continue
        raise RuntimeError("Failed to download a suitable Pima dataset from known mirrors.")

    # -------------------------
    # ADULT LOADER
    # -------------------------
    # pmas/agents/data_agent.py
# Replace the adult-loading logic with this robust loader

import os
import pandas as pd
import urllib.error
from pathlib import Path

class DataAgent:
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

    def _load_adult(self, use_cache_only: bool = False):
        """
        Robust adult loader:
         - prefer existing local cache `data/adult.csv`
         - otherwise try a list of mirrors (ADULT_URLS)
         - if all fail and use_cache_only=True, raise informative error
        """
        cache_path = Path("data") / "adult.csv"
        # 1) cache-first
        if cache_path.exists():
            df = pd.read_csv(cache_path, header=None, names=self.ADULT_COLS)
            # sanitize trailing spaces in category text
            for c in df.select_dtypes(include="object").columns:
                df[c] = df[c].str.strip()
            return df

        if use_cache_only:
            raise RuntimeError(
                "Adult dataset not found in repo. Please add data/adult.csv for offline runs."
            )

        # 2) try mirrors
        last_exc = None
        for url in self.ADULT_URLS:
            try:
                df = pd.read_csv(url, header=None)
                # attempt to normalize columns if URL has header or different shape
                if df.shape[1] == len(self.ADULT_COLS):
                    df.columns = self.ADULT_COLS
                else:
                    # try reading with no header and assigning canonical names
                    df.columns = list(range(df.shape[1]))
                    # if it's in selva86 format (has header), attempt to match common names
                    try:
                        df = pd.read_csv(url)
                        df = df.rename(columns=lambda s: s.strip())
                    except Exception:
                        pass
                # sanitize text fields
                for c in df.select_dtypes(include="object").columns:
                    df[c] = df[c].str.strip()
                # save a local cache copy for future runs
                try:
                    os.makedirs("data", exist_ok=True)
                    df.to_csv(cache_path, index=False)
                except Exception:
                    pass
                return df
            except Exception as e:
                last_exc = e
                # continue trying next mirror

        # 3) all mirrors failed
        raise RuntimeError(
            "Failed to download the Adult Income dataset from known mirrors. "
            "Please manually download and save it to data/adult.csv. "
            f"Last exception: {last_exc}"
        )


    # -------------------------
    # Adult postprocess
    # -------------------------
    def _postprocess_adult(self, df):
        # normalize label
        if "income" not in df.columns and "class" in df.columns:
            df = df.rename(columns={"class": "income"})
        if "income" in df.columns and "y" not in df.columns:
            df["income"] = df["income"].astype(str)
            df["y"] = df["income"].apply(
                lambda x: 1 if (">" in x and "50" in x) or (">50" in x) else 0
            )

        # create sex_bin
        if "sex" in df.columns and "sex_bin" not in df.columns:
            df["sex_bin"] = df["sex"].apply(
                lambda s: 1 if str(s).strip().lower().startswith("m") else 0
            )
        return df

    # -------------------------
    # perform() entrypoint
    # -------------------------
    def perform(self, action, params, state=None):
        if action != "load":
            raise NotImplementedError(action)

        dataset = params.get("dataset", "Pima Diabetes")
        use_cache_only = params.get("use_cache_only", False)

        if dataset == "Pima Diabetes":
            df = self._load_pima()
            df["age_group"] = (df["age"] >= self.AGE_CUTOFF).astype(int)
            FEATURES = [
                "pregnant","glucose","pressure","triceps","insulin",
                "mass","pedigree","age"
            ]
            sensitive_name = "age_group"

        elif dataset == "Adult Income":
            df = self._load_adult(use_cache_only=use_cache_only)
            # choose candidate features
            candidate_features = [
                c for c in ["age","education_num","capital_gain","capital_loss","hours_per_week"]
                if c in df.columns
            ]
            if not candidate_features:
                candidate_features = [c for c in df.select_dtypes(include=["number"]).columns
                                      if c not in ("sex_bin","y")]
            FEATURES = candidate_features[:5]
            sensitive_name = "sex_bin"
            if "y" not in df.columns:
                raise RuntimeError("Adult loader failed to produce label 'y'")

        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        # split
        try:
            train_df, test_df = train_test_split(
                df, test_size=self.test_size, random_state=self.seed, stratify=df["y"]
            )
        except Exception:
            train_df, test_df = train_test_split(
                df, test_size=self.test_size, random_state=self.seed
            )

        return {
            "train_df": train_df.reset_index(drop=True),
            "test_df": test_df.reset_index(drop=True),
            "features": FEATURES,
            "sensitive_name": sensitive_name,
            "dataset": dataset,
        }
