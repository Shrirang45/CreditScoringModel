# ============================================================
# Final Model Configuration
# ============================================================

MODEL_THRESHOLD = 0.25

MODEL_NAME = "CatBoost"

FEATURE_COUNT = 46

# Prototype business assumption:
# False Negative is considered 5x more costly
# than False Positive.
FN_COST = 5
FP_COST = 1