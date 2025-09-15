# check_env.py
import sys
import importlib

print("Python:", sys.version)
packages = ["numpy","pandas","sklearn","streamlit","fairlearn","shap","matplotlib"]
for pkg in packages:
    try:
        m = importlib.import_module(pkg)
        print(f"{pkg}: OK ({getattr(m, '__version__', 'unknown')})")
    except Exception as e:
        print(f"{pkg}: MISSING or error -> {e}")
