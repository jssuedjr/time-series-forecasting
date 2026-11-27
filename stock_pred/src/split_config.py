"""Frozen TSLA historical experiment; keep these dates identical across models."""
import pandas as pd

TRAIN_END_DATE = pd.Timestamp("2024-11-07")
VALIDATION_END_DATE = pd.Timestamp("2025-11-10")
TEST_END_DATE = pd.Timestamp("2026-11-19")
HORIZON = 1
WINDOW_SIZE = 252
INTERVAL_LEVEL = 0.95
