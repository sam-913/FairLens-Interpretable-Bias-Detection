# pmas/agents/data_agent.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class DataAgent:
    """
    Loads datasets (Pima Diabetes or Adult Income), preprocesses them into train/test,
    ensures numeric input (via get_dummies + imputation), and sets sensitive features.
    """

    PIMA_URLS = [
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv",
        "https://raw.githubusercontent.com/selva86/datasets/master/PimaIndiansDiabetes.csv",
    ]
    ADULT_URLS = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        "https://raw.githubusercontent.com/selva86/datasets/master/Adult.csv",
    ]

    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    # ----------------------------
    # Dataset loaders
    # ----------------------------
    def _load_pima(self, use_cache_only=False):
        cache_path = os.path.join(self.data_dir, "pima_diabetes.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path)
        elif not use_cache_only:
            for url in self.PIMA_URLS:
                try:
                    df = pd.read_csv(url, header=None)
                    if df.shape[1] == 9:
                        df.columns = [
                            "preg", "glucose", "bp", "skin", "insulin",
                            "bmi", "pedigree", "age", "y"
                        ]
                        df.to_csv(cache_path, index=False)
                        break
                except Exception:
                    continue
            else:
                raise RuntimeError("Failed to download Pima dataset.")
        else:
            raise RuntimeError("Pima dataset not available in cache.")
        return df

    def _load_adult(self, use_cache_only=False):
        cache_path = os.path.join(self.data_dir, "adult.csv")
        if os.path.exists(cache_path):
            df = pd.read_csv(cache_path, header=None)
        elif not use_cache_only:
            for url in self.ADULT_URLS:
                try:
                    df = pd.read_csv(url, header=None)
                    df.to_csv(cache_path, index=False)
                    break
                except Exception:
                    continue
            else:
                raise RuntimeError("Failed to download Adult dataset.")
        else:
            raise RuntimeError("Adult dataset not available in cache.")

        # Assign column names
        df.columns = [
            "age", "workclass", "fnlwgt", "education", "education_num",
            "marital_status", "occupation", "relationship", "race", "sex",
            "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
        ]

        # Strip whitespace
        df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Binary target
        df["y"] = (df["income"] == ">50K").astype(int)
        df = df.drop(columns=["income"])
        return df

    # ----------------------------
    # Preprocessing
    # ----------------------------
    def _preprocess(self, df, dataset):
        if dataset == "Pima Diabetes":
            df["age_group"] = (df["age"] >= 30).astype(int)
            features = [c for c in df.columns if c != "y"]
            sensitive_name = "age_group"

        elif dataset == "Adult Income":
            df["sex_bin"] = (df["sex"] == "Male").astype(int)
            sensitive_name = "sex_bin"

            # One-hot encode categoricals
            df = pd.get_dummies(df, columns=[
                "workclass", "education", "marital_status",
                "occupation", "relationship", "race", "sex", "native_country"
            ], drop_first=True)

            # define features after encoding
            features = [c for c in df.columns if c != "y"]

        else:
            raise RuntimeError(f"Unknown dataset {dataset}")

        # Impute only numeric columns
        imputer = SimpleImputer(strategy="mean")
        df[features] = imputer.fit_transform(df[features])

        return df, features, sensitive_name

    # ----------------------------
    # Perform interface
    # ----------------------------
    def perform(self, action, params, state=None):
        if action != "load":
            raise NotImplementedError(action)
        dataset = params.get("dataset", "Pima Diabetes")
        use_cache_only = params.get("use_cache_only", False)

        if dataset == "Pima Diabetes":
            df = self._load_pima(use_cache_only)
        elif dataset == "Adult Income":
            df = self._load_adult(use_cache_only)
        else:
            raise RuntimeError(f"Unknown dataset {dataset}")

        df, features, sensitive_name = self._preprocess(df, dataset)

        # Train/test split
        train_df, test_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df["y"])

        return {
            "train_df": train_df,
            "test_df": test_df,
            "features": features,
            "sensitive_name": sensitive_name,
        }
