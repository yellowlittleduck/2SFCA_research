import pandas as pd
import numpy as np
import glob, os

# ============================
# Configuration
# ============================
INPUT_FOLDER = r"E:\research4\studyarea_data\EA_2SFCA"
ACCESS_FIELD = "accessibility_score"
T0 = 15

TIME_POINTS = [f"{h:02d}0000" for h in range(6, 19)]
TIME_LABELS = [f"{h:02d}:00" for h in range(6, 19)]
N_TIMES = len(TIME_LABELS)

MIN_POSITIVE = 6  # 至少有一半的时段是正值，才纳入CV分析

# ============================
# Read data
# ============================
all_data = {}
all_grids = set()

for t, label in zip(TIME_POINTS, TIME_LABELS):
    pattern = os.path.join(INPUT_FOLDER, f"*f{t}_f{T0}.csv")
    files = glob.glob(pattern)
    assert len(files) >= 1, f"Missing: {t}"
    df = pd.read_csv(files[0])
    all_data[label] = df.set_index("pop_id")[ACCESS_FIELD]
    all_grids.update(df["pop_id"])

all_grids = sorted(all_grids)

# Align to common grid set
aligned = pd.DataFrame(index=all_grids, dtype=float)
for label in TIME_LABELS:
    aligned[label] = all_data[label].reindex(all_grids).fillna(0)

print(f"Total grids in union: {len(all_grids)}")
print(f"Time points: {len(TIME_LABELS)} hourly from 06:00 to 18:00")

# ============================
# Calculate CV per grid
# ============================
results = []

for g in all_grids:
    vals = aligned.loc[g].values  # 13 values
    pos_mask = vals > 0
    n_pos = pos_mask.sum()
    pos_vals = vals[pos_mask]

    if n_pos >= MIN_POSITIVE:
        mu = pos_vals.mean()
        sigma = pos_vals.std(ddof=1)  # sample std
        cv = sigma / mu if mu > 0 else np.nan
    else:
        mu = np.nan
        sigma = np.nan
        cv = np.nan

    results.append({
        "pop_id": g,
        "n_positive": n_pos,
        "mean_pos": mu,
        "std_pos": sigma,
        "cv": cv,
        "mean_all": vals.mean(),
        "min": vals.min(),
        "max": vals.max(),
    })

df_cv = pd.DataFrame(results)
df_cv_valid = df_cv.dropna(subset=["cv"])

# ============================
# Output
# ============================
print(f"\nGrids with >= {MIN_POSITIVE} positive time points: {len(df_cv_valid)} / {len(df_cv)}")

print("\n=== CV distribution ===")
print(df_cv_valid["cv"].describe())

print(f"\n=== Number of grids by CV quintile ===")
df_cv_valid["cv_quintile"] = pd.qcut(df_cv_valid["cv"], 5, labels=False)
q_counts = df_cv_valid["cv_quintile"].value_counts().sort_index()
for q in range(5):
    bounds = df_cv_valid.groupby("cv_quintile")["cv"].agg(["min", "max"])
    print(f"  Q{q+1}: {q_counts[q]} grids  "
          f"CV range: {bounds.loc[q, 'min']:.4f} ~ {bounds.loc[q, 'max']:.4f}")

# Save
out_path = os.path.join(INPUT_FOLDER, "grid_cv_15min.csv")
df_cv_valid.to_csv(out_path, index=False, encoding="utf-8-sig")
print(f"\nSaved: {out_path}")

print("\n=== Top 20 most variable grids (highest CV) ===")
print(df_cv_valid.sort_values("cv", ascending=False).head(20).to_string(index=False))

print("\n=== Bottom 20 most stable grids (lowest CV) ===")
print(df_cv_valid.sort_values("cv", ascending=True).head(20).to_string(index=False))
