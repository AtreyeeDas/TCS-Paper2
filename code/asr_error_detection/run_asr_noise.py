import os
import json
import re
import shutil
import numpy as np
import pandas as pd
import torch
import whisper
import jiwer
import soundfile as sf
from tqdm import tqdm

# =========================================================
# CONFIGURATION
# =========================================================

WHISPER_MODEL_PATH = "/home/spark2/Models/base.en.pt"

DATASET_CSV = (
    "/home/spark2/users/intern/Atreyee-Das/"
    "NLU_Robust_Experiment/dataset/"
    "nlu_robust_6000_scenario_paraphrase.csv"
)

TARGET_CSV = (
    "/home/spark2/users/intern/Atreyee-Das/"
    "NLU_Robust_Experiment/dataset/"
    "domain_term_targets_6000.csv"
)

AUDIO_DIR = (
    "/home/spark2/users/intern/Atreyee-Das/"
    "NLU_Robust_Experiment/audio"
)

ASR_DIR = (
    "/home/spark2/users/intern/Atreyee-Das/"
    "NLU_Robust_Experiment/asr_domain_corruption"
)

CORRUPTED_AUDIO_DIR = os.path.join(
    ASR_DIR, "corrupted_audio"
)

CLEAN_JSON_DIR = os.path.join(
    ASR_DIR, "clean_json"
)

CORRUPTED_JSON_DIR = os.path.join(
    ASR_DIR, "corrupted_json"
)

OUTPUT_CSV = os.path.join(
    ASR_DIR,
    "whisper_domain_error_6000.csv"
)

PROGRESS_FILE = os.path.join(
    ASR_DIR,
    "progress.json"
)

# Try progressively stronger LOCAL corruption.
LOCAL_SNRS_DB = [6.0, 3.0, 0.0]

# Expand target interval slightly on both sides.
# This avoids unrealistically sharp noise boundaries.
PAD_MS = 40

# Deterministic Whisper decoding.
WHISPER_TEMPERATURE = 0.0

# DO NOT set best_of=0.
# None/omitted is correct for greedy decoding.
WHISPER_BEST_OF = None
WHISPER_BEAM_SIZE = None


# =========================================================
# HELPERS
# =========================================================

def normalize_word(word):
    """
    Normalize a Whisper word for matching.
    """
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(word).lower()
    )


def normalize_phrase(text):
    return re.sub(
        r"[^a-z0-9 ]",
        "",
        str(text).lower()
    ).strip()


def find_target_span(result, target_term):
    """
    Find the target term in Whisper word timestamps.

    Handles both:
        single word
        multi-word phrase
    """

    target_words = [
        normalize_word(x)
        for x in str(target_term).split()
        if normalize_word(x)
    ]

    if not target_words:
        return None

    whisper_words = []

    for segment in result.get("segments", []):
        for w in segment.get("words", []):
            word = normalize_word(w.get("word", ""))

            if word:
                whisper_words.append({
                    "word": word,
                    "start": float(w["start"]),
                    "end": float(w["end"])
                })

    if not whisper_words:
        return None

    # -----------------------------------------------------
    # Exact contiguous word match
    # -----------------------------------------------------

    for i in range(
        len(whisper_words) - len(target_words) + 1
    ):
        candidate = [
            whisper_words[i + j]["word"]
            for j in range(len(target_words))
        ]

        if candidate == target_words:

            start = whisper_words[i]["start"]
            end = whisper_words[
                i + len(target_words) - 1
            ]["end"]

            return start, end

    # -----------------------------------------------------
    # Single-word fuzzy fallback
    # -----------------------------------------------------

    if len(target_words) == 1:

        target = target_words[0]

        best = None
        best_score = 0.0

        from difflib import SequenceMatcher

        for w in whisper_words:

            score = SequenceMatcher(
                None,
                target,
                w["word"]
            ).ratio()

            if score > best_score:
                best_score = score
                best = w

        # Require reasonable lexical similarity.
        if best is not None and best_score >= 0.60:
            return best["start"], best["end"]

    return None


def add_local_noise(
    audio,
    sr,
    start_sec,
    end_sec,
    snr_db
):
    """
    Add white Gaussian noise ONLY around the target term.
    """

    audio = audio.astype(np.float32).copy()

    start = max(
        0,
        int((start_sec * sr) - (PAD_MS / 1000 * sr))
    )

    end = min(
        len(audio),
        int((end_sec * sr) + (PAD_MS / 1000 * sr))
    )

    if end <= start:
        return audio

    target = audio[start:end]

    signal_rms = np.sqrt(
        np.mean(target ** 2) + 1e-12
    )

    noise_rms = (
        signal_rms /
        (10 ** (snr_db / 20))
    )

    noise = np.random.normal(
        0,
        noise_rms,
        size=len(target)
    ).astype(np.float32)

    # -----------------------------------------------------
    # Smooth fade-in / fade-out
    # -----------------------------------------------------

    fade_samples = min(
        int(0.04 * sr),
        len(target) // 4
    )

    if fade_samples > 1:

        fade_in = np.linspace(
            0.0,
            1.0,
            fade_samples
        )

        fade_out = np.linspace(
            1.0,
            0.0,
            fade_samples
        )

        noise[:fade_samples] *= fade_in
        noise[-fade_samples:] *= fade_out

    corrupted_target = target + noise

    # Prevent clipping.
    corrupted_target = np.clip(
        corrupted_target,
        -1.0,
        1.0
    )

    audio[start:end] = corrupted_target

    return audio


def target_changed(
    target_term,
    clean_transcript,
    corrupted_transcript
):
    """
    Conservative test for whether the target term disappeared
    or changed in the corrupted transcription.

    We compare normalized target presence rather than simply
    checking whole-sentence WER.
    """

    target = normalize_phrase(target_term)

    clean_text = normalize_phrase(
        clean_transcript
    )

    corrupt_text = normalize_phrase(
        corrupted_transcript
    )

    # If target wasn't actually present in clean decoding,
    # this is not a valid target-error experiment.
    if target not in clean_text:
        return False

    # Target no longer present -> definite target corruption.
    if target not in corrupt_text:
        return True

    return False


# =========================================================
# MAIN
# =========================================================

def main():

    os.makedirs(
        ASR_DIR,
        exist_ok=True
    )

    os.makedirs(
        CORRUPTED_AUDIO_DIR,
        exist_ok=True
    )

    os.makedirs(
        CLEAN_JSON_DIR,
        exist_ok=True
    )

    os.makedirs(
        CORRUPTED_JSON_DIR,
        exist_ok=True
    )

    print("Loading dataset...")

    df = pd.read_csv(DATASET_CSV)

    target_df = pd.read_csv(TARGET_CSV)

    print(
        f"Dataset rows: {len(df)}"
    )

    print(
        f"Target rows: {len(target_df)}"
    )

    # -----------------------------------------------------
    # Merge target terms
    # -----------------------------------------------------

    required_target_cols = [
        "sample_id"
    ]

    for c in required_target_cols:
        if c not in target_df.columns:
            raise ValueError(
                f"Target CSV missing column: {c}"
            )

    # Automatically find likely target-term column.
    possible_term_cols = [
        "target_term",
        "domain_term",
        "target",
        "term"
    ]

    target_col = None

    for c in possible_term_cols:
        if c in target_df.columns:
            target_col = c
            break

    if target_col is None:
        raise ValueError(
            "Could not find target-term column. "
            "Expected one of: "
            + str(possible_term_cols)
        )

    print(
        f"Using target-term column: {target_col}"
    )

    target_df = target_df[
        ["sample_id", target_col]
    ].rename(
        columns={
            target_col: "target_term"
        }
    )

    df["sample_id"] = (
        df["sample_id"].astype(str)
    )

    target_df["sample_id"] = (
        target_df["sample_id"].astype(str)
    )

    df = df.merge(
        target_df,
        on="sample_id",
        how="left"
    )

    # -----------------------------------------------------
    # Resume
    # -----------------------------------------------------

    completed = set()

    if os.path.exists(PROGRESS_FILE):

        with open(
            PROGRESS_FILE,
            "r"
        ) as f:

            completed = set(
                json.load(f)
            )

    results = []

    if os.path.exists(OUTPUT_CSV):

        results = pd.read_csv(
            OUTPUT_CSV
        ).to_dict(
            "records"
        )

    print(
        f"Already completed: "
        f"{len(completed)}"
    )

    # -----------------------------------------------------
    # Load Whisper
    # -----------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Loading Whisper on {device}..."
    )

    model = whisper.load_model(
        WHISPER_MODEL_PATH,
        device=device
    )

    # -----------------------------------------------------
    # Process
    # -----------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Domain-target ASR corruption"
    ):

        sample_id = str(
            row["sample_id"]
        )

        if sample_id in completed:
            continue

        target_term = str(
            row["target_term"]
        ).strip()

        audio_path = os.path.join(
            AUDIO_DIR,
            f"{sample_id}.wav"
        )

        if (
            not os.path.exists(audio_path)
            or not target_term
            or target_term.lower() == "nan"
        ):
            continue

        # -------------------------------------------------
        # Load audio
        # -------------------------------------------------

        audio, sr = sf.read(
            audio_path
        )

        if audio.ndim > 1:
            audio = np.mean(
                audio,
                axis=1
            )

        audio = audio.astype(
            np.float32
        )

        # -------------------------------------------------
        # CLEAN WHISPER TRANSCRIPTION
        # -------------------------------------------------

        clean_result = model.transcribe(
            audio_path,
            temperature=0.0,
            beam_size=None,
            best_of=None,
            word_timestamps=True,
            language="en"
        )

        clean_transcript = (
            clean_result["text"]
            .strip()
        )

        clean_json_path = os.path.join(
            CLEAN_JSON_DIR,
            f"{sample_id}.json"
        )

        with open(
            clean_json_path,
            "w"
        ) as f:

            json.dump(
                clean_result,
                f
            )

        # -------------------------------------------------
        # LOCATE DOMAIN TERM
        # -------------------------------------------------

        span = find_target_span(
            clean_result,
            target_term
        )

        if span is None:

            record = {
                "sample_id": sample_id,
                "scenario_id": row["scenario_id"],
                "split": row["split"],
                "ground_truth": row["transcript"],
                "target_term": target_term,
                "clean_whisper_transcript":
                    clean_transcript,
                "corrupted_whisper_transcript":
                    "",
                "target_found": 0,
                "target_corrupted": 0,
                "selected_snr_db": np.nan,
                "wer_clean":
                    jiwer.wer(
                        str(row["transcript"]),
                        clean_transcript
                    ),
                "wer_corrupted": np.nan,
                "temperature": 0.0,
                "beam_size": None,
                "best_of": None
            }

            results.append(record)
            completed.add(sample_id)
            continue

        start_sec, end_sec = span

        selected = None

        # -------------------------------------------------
        # TRY LOCAL SNR LEVELS
        # -------------------------------------------------

        for snr_db in LOCAL_SNRS_DB:

            corrupted_audio = add_local_noise(
                audio,
                sr,
                start_sec,
                end_sec,
                snr_db
            )

            temp_path = os.path.join(
                CORRUPTED_AUDIO_DIR,
                f"{sample_id}_temp.wav"
            )

            sf.write(
                temp_path,
                corrupted_audio,
                sr
            )

            corrupted_result = model.transcribe(
                temp_path,
                temperature=0.0,
                beam_size=None,
                best_of=None,
                word_timestamps=True,
                language="en"
            )

            corrupted_transcript = (
                corrupted_result["text"]
                .strip()
            )

            changed = target_changed(
                target_term,
                clean_transcript,
                corrupted_transcript
            )

            if changed:

                final_audio_path = os.path.join(
                    CORRUPTED_AUDIO_DIR,
                    f"{sample_id}.wav"
                )

                shutil.copy2(
                    temp_path,
                    final_audio_path
                )

                selected = (
                    snr_db,
                    corrupted_transcript,
                    corrupted_result,
                    final_audio_path
                )

                break

        # -------------------------------------------------
        # RECORD RESULT
        # -------------------------------------------------

        if selected is None:

            selected_snr = np.nan
            corrupted_transcript = ""
            corrupted_result = None

            target_corrupted = 0

        else:

            (
                selected_snr,
                corrupted_transcript,
                corrupted_result,
                final_audio_path
            ) = selected

            target_corrupted = 1

            corrupted_json_path = os.path.join(
                CORRUPTED_JSON_DIR,
                f"{sample_id}.json"
            )

            with open(
                corrupted_json_path,
                "w"
            ) as f:

                json.dump(
                    corrupted_result,
                    f
                )

        # -------------------------------------------------
        # METRICS
        # -------------------------------------------------

        ground_truth = str(
            row["transcript"]
        )

        wer_clean = jiwer.wer(
            ground_truth,
            clean_transcript
        )

        wer_corrupted = (
            jiwer.wer(
                ground_truth,
                corrupted_transcript
            )
            if corrupted_transcript
            else np.nan
        )

        record = {

            "sample_id":
                sample_id,

            "scenario_id":
                row["scenario_id"],

            "split":
                row["split"],

            "domain_label":
                row.get(
                    "domain_label",
                    ""
                ),

            "ground_truth":
                ground_truth,

            "target_term":
                target_term,

            "target_start_sec":
                start_sec,

            "target_end_sec":
                end_sec,

            "clean_whisper_transcript":
                clean_transcript,

            "corrupted_whisper_transcript":
                corrupted_transcript,

            "target_found":
                1,

            "target_corrupted":
                target_corrupted,

            "selected_snr_db":
                selected_snr,

            "wer_clean":
                wer_clean,

            "wer_corrupted":
                wer_corrupted,

            "temperature":
                0.0,

            "beam_size":
                None,

            "best_of":
                None,

            "language":
                clean_result.get(
                    "language",
                    "en"
                )
        }

        results.append(record)

        completed.add(
            sample_id
        )

        # -------------------------------------------------
        # CHECKPOINT
        # -------------------------------------------------

        if (
            len(completed) % 25 == 0
        ):

            pd.DataFrame(
                results
            ).to_csv(
                OUTPUT_CSV,
                index=False
            )

            with open(
                PROGRESS_FILE,
                "w"
            ) as f:

                json.dump(
                    list(completed),
                    f
                )

    # -----------------------------------------------------
    # FINAL SAVE
    # -----------------------------------------------------

    pd.DataFrame(
        results
    ).to_csv(
        OUTPUT_CSV,
        index=False
    )

    with open(
        PROGRESS_FILE,
        "w"
    ) as f:

        json.dump(
            list(completed),
            f
        )

    result_df = pd.DataFrame(
        results
    )

    print("\n====================================")
    print("DOMAIN-TERM ASR STRESS TEST COMPLETE")
    print("====================================")

    print(
        "Samples processed:",
        len(result_df)
    )

    print(
        "Target terms found:",
        result_df["target_found"].sum()
    )

    print(
        "Target terms actually corrupted:",
        result_df["target_corrupted"].sum()
    )

    print(
        "Target corruption rate:",
        result_df["target_corrupted"].mean()
    )

    print(
        "Mean clean WER:",
        result_df["wer_clean"].mean()
    )

    print(
        "Mean corrupted WER:",
        result_df["wer_corrupted"].mean()
    )

    print(
        "\nResults:",
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()
