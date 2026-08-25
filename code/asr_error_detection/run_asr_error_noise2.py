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
from difflib import SequenceMatcher


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
    "domain_term_targets_6000_multitarget.csv"
)

AUDIO_DIR = (
    "/home/spark2/users/intern/Atreyee-Das/"
    "NLU_Robust_Experiment/audio"
)

ASR_DIR = (
    "/home/spark2/users/intern/Atreyee-Das/"
    "NLU_Robust_Experiment/asr_domain_multitarget"
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
    "whisper_domain_multitarget_6000.csv"
)

PROGRESS_FILE = os.path.join(
    ASR_DIR,
    "progress.json"
)

# Try weakest → strongest local corruption.
LOCAL_SNRS_DB = [6.0, 3.0, 0.0]

# Expand target interval slightly.
PAD_MS = 40

# Deterministic Whisper decoding.
TEMPERATURE = 0.0
BEAM_SIZE = None
BEST_OF = None


# =========================================================
# TEXT NORMALIZATION
# =========================================================

def normalize_word(x):
    return re.sub(
        r"[^a-z0-9]",
        "",
        str(x).lower()
    )


def normalize_text(x):
    return " ".join(
        normalize_word(w)
        for w in str(x).split()
        if normalize_word(w)
    )


def phrase_words(x):
    return [
        normalize_word(w)
        for w in str(x).split()
        if normalize_word(w)
    ]


# =========================================================
# TARGET LIST
# =========================================================

def get_target_terms(row):

    raw = str(row["target_terms"])

    if raw.lower() in ("nan", "", "none"):
        return []

    return [
        x.strip()
        for x in raw.split(";")
        if x.strip()
    ]


# =========================================================
# FIND TARGET TIMESTAMP
# =========================================================

def find_target_span(
    whisper_result,
    target_term
):

    target = phrase_words(target_term)

    if not target:
        return None

    words = []

    for seg in whisper_result.get(
        "segments", []
    ):

        for w in seg.get(
            "words", []
        ):

            word = normalize_word(
                w.get("word", "")
            )

            if word:

                words.append({
                    "word": word,
                    "start": float(w["start"]),
                    "end": float(w["end"])
                })

    # Exact phrase match.
    for i in range(
        len(words) - len(target) + 1
    ):

        candidate = [
            words[i + j]["word"]
            for j in range(len(target))
        ]

        if candidate == target:

            return (
                words[i]["start"],
                words[
                    i + len(target) - 1
                ]["end"]
            )

    # Fuzzy fallback for single terms.
    if len(target) == 1:

        best = None
        best_score = 0

        for w in words:

            score = SequenceMatcher(
                None,
                target[0],
                w["word"]
            ).ratio()

            if score > best_score:
                best_score = score
                best = w

        if (
            best is not None
            and best_score >= 0.60
        ):

            return (
                best["start"],
                best["end"]
            )

    return None


# =========================================================
# ADD LOCAL NOISE TO MULTIPLE TARGET SPANS
# =========================================================

def corrupt_target_spans(
    audio,
    sr,
    spans,
    snr_db
):

    audio = audio.astype(
        np.float32
    ).copy()

    # Create one mask covering ALL target regions.
    mask = np.zeros(
        len(audio),
        dtype=np.float32
    )

    for start_sec, end_sec in spans:

        start = max(
            0,
            int(
                start_sec * sr
                - PAD_MS / 1000 * sr
            )
        )

        end = min(
            len(audio),
            int(
                end_sec * sr
                + PAD_MS / 1000 * sr
            )
        )

        if end <= start:
            continue

        mask[start:end] = 1.0

    # Nothing to corrupt.
    if mask.sum() == 0:
        return audio

    affected = audio[mask > 0]

    signal_rms = np.sqrt(
        np.mean(
            affected ** 2
        ) + 1e-12
    )

    noise_rms = (
        signal_rms /
        (10 ** (snr_db / 20))
    )

    noise = np.random.normal(
        0,
        noise_rms,
        size=len(audio)
    ).astype(np.float32)

    # Smooth transitions at every target region.
    for start_sec, end_sec in spans:

        start = max(
            0,
            int(
                start_sec * sr
                - PAD_MS / 1000 * sr
            )
        )

        end = min(
            len(audio),
            int(
                end_sec * sr
                + PAD_MS / 1000 * sr
            )
        )

        fade = min(
            int(0.04 * sr),
            (end - start) // 4
        )

        if fade > 1:

            noise[
                start:start + fade
            ] *= np.linspace(
                0, 1, fade
            )

            noise[
                end - fade:end
            ] *= np.linspace(
                1, 0, fade
            )

    corrupted = audio + noise * mask

    return np.clip(
        corrupted,
        -1.0,
        1.0
    )


# =========================================================
# CHECK WHICH TARGET TERMS SURVIVED
# =========================================================

def target_presence(
    targets,
    transcript
):

    text = normalize_text(
        transcript
    )

    results = {}

    for target in targets:

        target_norm = normalize_text(
            target
        )

        results[target] = (
            target_norm in text
        )

    return results


# =========================================================
# TARGET-LEVEL METRICS
# =========================================================

def target_metrics(
    targets,
    reference_text,
    hypothesis_text
):

    ref_presence = target_presence(
        targets,
        reference_text
    )

    hyp_presence = target_presence(
        targets,
        hypothesis_text
    )

    tp = 0
    fn = 0
    fp = 0

    for target in targets:

        ref = ref_presence[target]
        hyp = hyp_presence[target]

        if ref and hyp:
            tp += 1

        elif ref and not hyp:
            fn += 1

        elif not ref and hyp:
            fp += 1

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    return (
        tp,
        fp,
        fn,
        precision,
        recall
    )


# =========================================================
# DOMAIN-TERM WER
# =========================================================

def domain_term_wer(
    targets,
    reference_text,
    hypothesis_text
):

    ref_tokens = []

    for target in targets:

        ref_tokens.extend(
            phrase_words(target)
        )

    ref_tokens = list(
        dict.fromkeys(ref_tokens)
    )

    if not ref_tokens:
        return np.nan

    hyp_tokens = phrase_words(
        hypothesis_text
    )

    # Count reference domain tokens
    # that are absent from the hypothesis.
    ref_counts = {}

    for token in ref_tokens:
        ref_counts[token] = (
            ref_counts.get(token, 0) + 1
        )

    hyp_counts = {}

    for token in hyp_tokens:
        hyp_counts[token] = (
            hyp_counts.get(token, 0) + 1
        )

    substitutions_or_deletions = 0

    for token, count in ref_counts.items():

        matched = min(
            count,
            hyp_counts.get(token, 0)
        )

        substitutions_or_deletions += (
            count - matched
        )

    n = sum(
        ref_counts.values()
    )

    return (
        substitutions_or_deletions / n
        if n > 0
        else np.nan
    )


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

    print(
        "Loading datasets..."
    )

    df = pd.read_csv(
        DATASET_CSV
    )

    target_df = pd.read_csv(
        TARGET_CSV
    )

    df["sample_id"] = (
        df["sample_id"].astype(str)
    )

    target_df["sample_id"] = (
        target_df["sample_id"].astype(str)
    )

    df = df.merge(
        target_df[
            [
                "sample_id",
                "target_terms",
                "target_term_count",
                "target_sources"
            ]
        ],
        on="sample_id",
        how="left"
    )

    print(
        "Total samples:",
        len(df)
    )

    # -----------------------------------------------------
    # RESUME
    # -----------------------------------------------------

    completed = set()

    if os.path.exists(
        PROGRESS_FILE
    ):

        with open(
            PROGRESS_FILE
        ) as f:

            completed = set(
                json.load(f)
            )

    results = []

    if os.path.exists(
        OUTPUT_CSV
    ):

        results = pd.read_csv(
            OUTPUT_CSV
        ).to_dict(
            "records"
        )

    # -----------------------------------------------------
    # LOAD WHISPER
    # -----------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Loading Whisper on",
        device
    )

    model = whisper.load_model(
        WHISPER_MODEL_PATH,
        device=device
    )

    # -----------------------------------------------------
    # PROCESS
    # -----------------------------------------------------

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Multi-target ASR corruption"
    ):

        sample_id = str(
            row["sample_id"]
        )

        if sample_id in completed:
            continue

        targets = get_target_terms(
            row
        )

        audio_path = os.path.join(
            AUDIO_DIR,
            f"{sample_id}.wav"
        )

        if (
            not os.path.exists(audio_path)
            or len(targets) == 0
        ):

            continue

        # -------------------------------------------------
        # LOAD AUDIO
        # -------------------------------------------------

        audio, sr = sf.read(
            audio_path
        )

        if audio.ndim > 1:
            audio = audio.mean(
                axis=1
            )

        audio = audio.astype(
            np.float32
        )

        # -------------------------------------------------
        # CLEAN TRANSCRIPTION
        # -------------------------------------------------

        clean_result = model.transcribe(
            audio_path,
            temperature=TEMPERATURE,
            beam_size=BEAM_SIZE,
            best_of=BEST_OF,
            word_timestamps=True,
            language="en"
        )

        clean_transcript = (
            clean_result["text"]
            .strip()
        )

        with open(
            os.path.join(
                CLEAN_JSON_DIR,
                f"{sample_id}.json"
            ),
            "w"
        ) as f:

            json.dump(
                clean_result,
                f
            )

        # -------------------------------------------------
        # FIND ALL TARGET SPANS
        # -------------------------------------------------

        spans = []
        found_targets = []

        for target in targets:

            span = find_target_span(
                clean_result,
                target
            )

            if span is not None:

                spans.append(span)
                found_targets.append(
                    target
                )

        # -------------------------------------------------
        # TRY LOCAL CORRUPTION
        # -------------------------------------------------

        selected_snr = np.nan
        corrupted_transcript = ""
        corrupted_result = None
        target_corrupted = []

        if spans:

            for snr_db in LOCAL_SNRS_DB:

                corrupted_audio = (
                    corrupt_target_spans(
                        audio,
                        sr,
                        spans,
                        snr_db
                    )
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

                result = model.transcribe(
                    temp_path,
                    temperature=TEMPERATURE,
                    beam_size=BEAM_SIZE,
                    best_of=BEST_OF,
                    word_timestamps=True,
                    language="en"
                )

                transcript = (
                    result["text"]
                    .strip()
                )

                presence = target_presence(
                    found_targets,
                    transcript
                )

                changed = [
                    t for t in found_targets
                    if not presence[t]
                ]

                if changed:

                    selected_snr = snr_db
                    corrupted_transcript = transcript
                    corrupted_result = result
                    target_corrupted = changed

                    final_path = os.path.join(
                        CORRUPTED_AUDIO_DIR,
                        f"{sample_id}.wav"
                    )

                    shutil.copy2(
                        temp_path,
                        final_path
                    )

                    break

        # -------------------------------------------------
        # TARGET METRICS
        # -------------------------------------------------

        tp, fp, fn, precision, recall = (
            target_metrics(
                found_targets,
                clean_transcript,
                corrupted_transcript
            )
        )

        ds_wer = domain_term_wer(
            found_targets,
            clean_transcript,
            corrupted_transcript
        )

        # -------------------------------------------------
        # STANDARD WER
        # -------------------------------------------------

        ground_truth = str(
            row["transcript"]
        )

        clean_wer = jiwer.wer(
            ground_truth,
            clean_transcript
        )

        corrupted_wer = (
            jiwer.wer(
                ground_truth,
                corrupted_transcript
            )
            if corrupted_transcript
            else np.nan
        )

        # -------------------------------------------------
        # SAVE CORRUPTED JSON
        # -------------------------------------------------

        if corrupted_result is not None:

            with open(
                os.path.join(
                    CORRUPTED_JSON_DIR,
                    f"{sample_id}.json"
                ),
                "w"
            ) as f:

                json.dump(
                    corrupted_result,
                    f
                )

        # -------------------------------------------------
        # RECORD
        # -------------------------------------------------

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

            "target_terms":
                "; ".join(targets),

            "target_term_count":
                len(targets),

            "targets_found":
                len(found_targets),

            "targets_corrupted":
                len(target_corrupted),

            "corrupted_target_terms":
                "; ".join(target_corrupted),

            "target_start_end_count":
                len(spans),

            "selected_snr_db":
                selected_snr,

            "clean_whisper_transcript":
                clean_transcript,

            "corrupted_whisper_transcript":
                corrupted_transcript,

            "clean_WER":
                clean_wer,

            "corrupted_WER":
                corrupted_wer,

            "domain_term_WER":
                ds_wer,

            "domain_term_TP":
                tp,

            "domain_term_FP":
                fp,

            "domain_term_FN":
                fn,

            "domain_term_precision":
                precision,

            "domain_term_recall":
                recall,

            "temperature":
                TEMPERATURE,

            "beam_size":
                BEAM_SIZE,

            "best_of":
                BEST_OF,

            "language":
                clean_result.get(
                    "language",
                    "en"
                )
        }

        results.append(record)
        completed.add(sample_id)

        # -------------------------------------------------
        # CHECKPOINT
        # -------------------------------------------------

        if len(completed) % 25 == 0:

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

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
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

    print("\n====================================")
    print("MULTI-TARGET ASR STRESS TEST")
    print("====================================")

    print(
        "Samples processed:",
        len(result_df)
    )

    print(
        "Total target terms:",
        result_df["target_term_count"].sum()
    )

    print(
        "Total targets found:",
        result_df["targets_found"].sum()
    )

    print(
        "Total targets corrupted:",
        result_df["targets_corrupted"].sum()
    )

    print(
        "Target corruption rate:",
        result_df["targets_corrupted"].sum()
        /
        max(
            result_df["targets_found"].sum(),
            1
        )
    )

    print(
        "Mean clean WER:",
        result_df["clean_WER"].mean()
    )

    print(
        "Mean corrupted WER:",
        result_df["corrupted_WER"].mean()
    )

    print(
        "Mean domain-term WER:",
        result_df["domain_term_WER"].mean()
    )

    print(
        "Domain-term precision:",
        result_df["domain_term_precision"].mean()
    )

    print(
        "Domain-term recall:",
        result_df["domain_term_recall"].mean()
    )

    print(
        "\nSaved:",
        OUTPUT_CSV
    )


if __name__ == "__main__":
    main()
