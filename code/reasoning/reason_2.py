# ============================================================
# DETECTOR INFERENCE
# ============================================================

df_a = (
    pd.DataFrame([feat_a])
    .reindex(columns=detector_A_features)
)

df_b = (
    pd.DataFrame([feat_b])
    .reindex(columns=detector_B_features)
)


if df_a.isnull().any().any():
    missing = df_a.columns[
        df_a.isnull().iloc[0]
    ].tolist()

    raise RuntimeError(
        f"Detector A feature reconstruction missing "
        f"features: {missing}"
    )


if df_b.isnull().any().any():
    missing = df_b.columns[
        df_b.isnull().iloc[0]
    ].tolist()

    raise RuntimeError(
        f"Detector B feature reconstruction missing "
        f"features: {missing}"
    )


# IMPORTANT:
# The original detector was fitted on NumPy arrays.
# Pass NumPy arrays here too, preserving the exact saved
# training-column order.

prob_a = detector_A.predict_proba(
    df_a.to_numpy(dtype=np.float32)
)[0, 1]

prob_b = detector_B.predict_proba(
    df_b.to_numpy(dtype=np.float32)
)[0, 1]


pred_a = int(
    prob_a >= thresh_A
)

pred_b = int(
    prob_b >= thresh_B
)
