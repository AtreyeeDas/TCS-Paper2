import os
import shutil
import pandas as pd
from tqdm import tqdm

# ==============================================================================
# CONFIGURATION & PATHS
# ==============================================================================
SLURP_SRC_DIR = "./slurp_general_1500"
PROCESSED_CSV_PATH = "./processed_nlu_data/slurp_processed.csv"

# Destination directories
SELECTED_AUDIO_DIR = "./slurp_general_filtered"
DROPPED_AUDIO_DIR = "./slurp_general_dropped"

MAX_AUDIO_PER_TRANSCRIPT = 2

os.makedirs(SELECTED_AUDIO_DIR, exist_ok=True)
os.makedirs(DROPPED_AUDIO_DIR, exist_ok=True)

# ==============================================================================
# 1. LOAD AND VALIDATE PROCESSED CSV
# ==============================================================================
if not os.path.exists(PROCESSED_CSV_PATH):
    # Check fallback if running directly inside processed_nlu_data
    if os.path.exists("slurp_processed.csv"):
        PROCESSED_CSV_PATH = "slurp_processed.csv"
    else:
        raise FileNotFoundError(f"Could not find {PROCESSED_CSV_PATH}")

print(f"[+] Loading: {PROCESSED_CSV_PATH}")
df = pd.read_csv(PROCESSED_CSV_PATH)
initial_count = len(df)
print(f"[+] Total initial SLURP records: {initial_count}")

# Normalize transcript for robust grouping (strip spaces, lowercase)
df["norm_transcript"] = df["transcript"].fillna("MASK").astype(str).str.strip().str.lower()

# ==============================================================================
# 2. SELECT AT MOST 2 AUDIOS PER UNIQUE TRANSCRIPT
# ==============================================================================
# Group by normalized transcript and keep up to MAX_AUDIO_PER_TRANSCRIPT
selected_indices = []
dropped_indices = []

for _, group in df.groupby("norm_transcript", sort=False):
    if len(group) <= MAX_AUDIO_PER_TRANSCRIPT:
        selected_indices.extend(group.index.tolist())
    else:
        selected_indices.extend(group.index[:MAX_AUDIO_PER_TRANSCRIPT].tolist())
        dropped_indices.extend(group.index[MAX_AUDIO_PER_TRANSCRIPT:].tolist())

df_selected = df.loc[selected_indices].copy()
df_dropped = df.loc[dropped_indices].copy()

print(f"[+] Unique transcripts found: {df['norm_transcript'].nunique()}")
print(f"[+] Selected records to keep : {len(df_selected)}")
print(f"[+] Dropped duplicate records: {len(df_dropped)}")

# ==============================================================================
# 3. COPY AUDIO FILES TO CORRESPONDING FOLDERS
# ==============================================================================
def resolve_source_audio_path(audio_path_col_val):
    """Finds the actual source path of the .wav file."""
    # Check direct relative path
    if os.path.exists(audio_path_col_val):
        return audio_path_col_val
    # Check inside SLURP_SRC_DIR
    filename = os.path.basename(audio_path_col_val)
    candidate = os.path.join(SLURP_SRC_DIR, filename)
    if os.path.exists(candidate):
        return candidate
    return None

print("\n[+] Copying selected audio files...")
for _, row in tqdm(df_selected.iterrows(), total=len(df_selected), desc="Copying Selected"):
    src_file = resolve_source_audio_path(str(row["audio_path"]))
    if src_file and os.path.exists(src_file):
        dest_file = os.path.join(SELECTED_AUDIO_DIR, os.path.basename(src_file))
        shutil.copy2(src_file, dest_file)

print("\n[+] Separating dropped audio files (for review/deletion)...")
for _, row in tqdm(df_dropped.iterrows(), total=len(df_dropped), desc="Archiving Dropped"):
    src_file = resolve_source_audio_path(str(row["audio_path"]))
    if src_file and os.path.exists(src_file):
        dest_file = os.path.join(DROPPED_AUDIO_DIR, os.path.basename(src_file))
        shutil.copy2(src_file, dest_file)

# ==============================================================================
# 4. UPDATE CSV AND UPDATE PATHS
# ==============================================================================
# Remove helper normalization column
df_selected = df_selected.drop(columns=["norm_transcript"])

# Update audio_path to reflect the filtered folder location
df_selected["audio_path"] = df_selected["audio_path"].apply(
    lambda p: f"slurp_general_filtered/{os.path.basename(str(p))}"
)

# Backup original CSV
backup_path = PROCESSED_CSV_PATH.replace(".csv", "_original_backup.csv")
shutil.copy2(PROCESSED_CSV_PATH, backup_path)
print(f"\n[+] Created backup of original CSV at: {backup_path}")

# Overwrite updated slurp_processed.csv
df_selected.to_csv(PROCESSED_CSV_PATH, index=False)
print(f"[+] Overwrote {PROCESSED_CSV_PATH} with {len(df_selected)} deduplicated records.")

# Also save a dedicated filtered CSV inside the new directory
df_selected.to_csv(os.path.join(SELECTED_AUDIO_DIR, "slurp_filtered_annotations.csv"), index=False)
print(f"[+] Done! Processed audio files saved in: {SELECTED_AUDIO_DIR}")
