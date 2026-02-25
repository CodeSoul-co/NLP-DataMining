#!/bin/bash
# =============================================================================
# THETA Data Cleaning Script
# =============================================================================
# Clean raw text data for topic modeling.
#
# Two modes:
#   1. CSV mode  (--input is a .csv file)
#      - User specifies which column(s) to clean and which to keep
#      - Row-by-row cleaning preserves original CSV structure
#      - Use --preview to inspect columns before committing
#
#   2. Directory mode  (--input is a directory of docx/txt files)
#      - Converts all files into a single CSV with cleaned text
#      - Uses the dataclean module (legacy behavior)
#
# Cleaning Steps (applied to text column only):
#   1. Remove URLs, emails, HTML tags
#   2. Collapse redaction markers (XXXX -> [REDACTED])
#   3. Remove special characters (keep basic punctuation)
#   4. Normalize whitespace
#   5. Lowercase
#   6. Remove short documents (< --min_words words)
#
# Usage:
#   # Preview CSV columns first:
#   ./02_clean_data.sh --input data.csv --preview
#
#   # Clean with explicit column selection:
#   ./02_clean_data.sh --input data.csv --language english \
#       --text_column "Consumer complaint narrative"
#
#   # Keep label columns alongside text:
#   ./02_clean_data.sh --input data.csv --language english \
#       --text_column "clean_text" --label_columns "label,category"
#
#   # Keep all original columns:
#   ./02_clean_data.sh --input data.csv --language english \
#       --text_column "text" --keep_all
#
#   # Directory mode (docx/txt files):
#   ./02_clean_data.sh --input /path/to/docs/ --language chinese
# =============================================================================

set -e

# Default values
LANGUAGE=""
INPUT_FILE=""
OUTPUT_FILE=""
TEXT_COLUMN=""
LABEL_COLUMNS=""
KEEP_ALL=false
PREVIEW=false
MIN_WORDS=3

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)          INPUT_FILE="$2"; shift 2 ;;
        --output)         OUTPUT_FILE="$2"; shift 2 ;;
        --language)       LANGUAGE="$2"; shift 2 ;;
        --text_column)    TEXT_COLUMN="$2"; shift 2 ;;
        --label_columns)  LABEL_COLUMNS="$2"; shift 2 ;;
        --keep_all)       KEEP_ALL=true; shift ;;
        --preview)        PREVIEW=true; shift ;;
        --min_words)      MIN_WORDS="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 --input <file_or_dir> --language <lang> [options]"
            echo ""
            echo "Required:"
            echo "  --input          Input CSV file or directory (docx/txt files)"
            echo "  --language       Language: english, chinese, german, spanish"
            echo ""
            echo "CSV Column Selection (required for CSV input):"
            echo "  --text_column    Name of the text column to clean (REQUIRED for CSV)"
            echo "  --label_columns  Comma-separated label/metadata columns to keep as-is"
            echo "  --keep_all       Keep ALL original columns (only text column is cleaned)"
            echo "  --preview        Show CSV columns and sample rows, then exit"
            echo ""
            echo "Optional:"
            echo "  --output         Output CSV path (default: {input}_cleaned.csv)"
            echo "  --min_words      Min words per document after cleaning (default: 3)"
            echo ""
            echo "Examples:"
            echo "  # Preview columns:"
            echo "  $0 --input data/FCPB/complaints_text_only.csv --preview"
            echo ""
            echo "  # Clean text column only:"
            echo "  $0 --input data/FCPB/complaints_text_only.csv --language english \\"
            echo "      --text_column 'Consumer complaint narrative'"
            echo ""
            echo "  # Keep label column:"
            echo "  $0 --input data/hatespeech/hatespeech_text_only.csv --language english \\"
            echo "      --text_column cleaned_content --label_columns Label"
            echo ""
            echo "  # Keep all columns, only clean the text column:"
            echo "  $0 --input raw.csv --language english --text_column text --keep_all"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate required parameters
if [ -z "$INPUT_FILE" ]; then
    echo "Error: --input is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

if [ ! -e "$INPUT_FILE" ]; then
    echo "Error: Input not found: $INPUT_FILE"
    exit 1
fi

# ============================================================
# Preview mode: show columns and exit
# ============================================================
if [ "$PREVIEW" = true ]; then
    python3 << PREVIEW_PY
import pandas as pd, sys

input_path = "$INPUT_FILE"

if input_path.endswith('.csv'):
    df = pd.read_csv(input_path, nrows=5)
    print("=" * 60)
    print(f"CSV Preview: {input_path}")
    print("=" * 60)
    print(f"Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        dtype = df[col].dtype
        sample = str(df[col].iloc[0])[:80] if len(df) > 0 else "N/A"
        print(f"  [{i}] {col:30s}  ({dtype})  e.g. {sample}")
    print()
    total = len(pd.read_csv(input_path, usecols=[0]))
    print(f"Total rows: {total}")
    print()
    print("Suggested usage:")
    # Heuristic: text column = longest average string length among object columns
    obj_cols = [c for c in df.columns if df[c].dtype == 'object']
    if obj_cols:
        avg_lens = {c: df[c].astype(str).str.len().mean() for c in obj_cols}
        text_col = max(avg_lens, key=avg_lens.get)
    else:
        text_col = df.columns[0]
    # Label columns = non-text columns with few unique values or numeric type
    label_cols = [c for c in df.columns if c != text_col and (df[c].nunique() < 100 or df[c].dtype != 'object')]
    print(f"  --text_column '{text_col}'", end="")
    if label_cols:
        print(f" --label_columns '{','.join(label_cols)}'", end="")
    print()
else:
    import os
    files = [f for f in os.listdir(input_path) if os.path.isfile(os.path.join(input_path, f))]
    print(f"Directory: {input_path}")
    print(f"Files: {len(files)}")
    for f in files[:10]:
        print(f"  {f}")
    if len(files) > 10:
        print(f"  ... and {len(files)-10} more")
PREVIEW_PY
    exit 0
fi

# ============================================================
# Determine mode: CSV vs Directory
# ============================================================
IS_CSV=false
if [ -f "$INPUT_FILE" ] && [[ "$INPUT_FILE" == *.csv ]]; then
    IS_CSV=true
fi

# For CSV mode, text_column is required
if [ "$IS_CSV" = true ] && [ -z "$TEXT_COLUMN" ]; then
    echo "Error: --text_column is required for CSV input"
    echo ""
    echo "Use --preview to see available columns:"
    echo "  $0 --input $INPUT_FILE --preview"
    exit 1
fi

# Language is required for cleaning
if [ -z "$LANGUAGE" ]; then
    echo "Error: --language is required"
    echo "Run '$0 --help' for usage"
    exit 1
fi

# Validate language
case $LANGUAGE in
    english|chinese|german|spanish) ;;
    *) echo "Error: Unknown language '$LANGUAGE'. Must be: english, chinese, german, spanish"; exit 1 ;;
esac

# Auto-generate output path
if [ -z "$OUTPUT_FILE" ]; then
    if [ -d "$INPUT_FILE" ]; then
        DATASET_NAME=$(basename "$INPUT_FILE")
        OUTPUT_FILE="/root/autodl-tmp/data/${DATASET_NAME}/${DATASET_NAME}_cleaned.csv"
    else
        DIR=$(dirname "$INPUT_FILE")
        BASENAME=$(basename "$INPUT_FILE" | sed 's/\.[^.]*$//')
        OUTPUT_FILE="${DIR}/${BASENAME}_cleaned.csv"
    fi
fi

echo "=========================================="
echo "THETA Data Cleaning"
echo "=========================================="
echo "Input:    $INPUT_FILE"
echo "Output:   $OUTPUT_FILE"
echo "Language: $LANGUAGE"

if [ "$IS_CSV" = true ]; then
    echo "Mode:     CSV (row-by-row)"
    echo "Text col: $TEXT_COLUMN"
    [ -n "$LABEL_COLUMNS" ] && echo "Labels:   $LABEL_COLUMNS"
    [ "$KEEP_ALL" = true ]   && echo "Keep:     ALL columns"
    echo "Min words: $MIN_WORDS"
    echo ""

    # ============================================================
    # CSV mode: row-by-row cleaning with column preservation
    # ============================================================
    export CLEAN_INPUT="$INPUT_FILE"
    export CLEAN_OUTPUT="$OUTPUT_FILE"
    export CLEAN_TEXT_COL="$TEXT_COLUMN"
    export CLEAN_LABEL_COLS="$LABEL_COLUMNS"
    export CLEAN_KEEP_ALL="$KEEP_ALL"
    export CLEAN_LANGUAGE="$LANGUAGE"
    export CLEAN_MIN_WORDS="$MIN_WORDS"

    python3 << 'CSV_CLEAN_PY'
import pandas as pd
import re
import sys
import os

# Read parameters from environment variables
INPUT_FILE  = os.environ["CLEAN_INPUT"]
OUTPUT_FILE = os.environ["CLEAN_OUTPUT"]
TEXT_COLUMN = os.environ["CLEAN_TEXT_COL"]
LABEL_COLS  = os.environ.get("CLEAN_LABEL_COLS", "")
KEEP_ALL    = os.environ.get("CLEAN_KEEP_ALL", "false") == "true"
LANGUAGE    = os.environ.get("CLEAN_LANGUAGE", "english")
MIN_WORDS   = int(os.environ.get("CLEAN_MIN_WORDS", "3"))

# ---- Load ----
print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"  Shape: {df.shape}, Columns: {list(df.columns)}")

if TEXT_COLUMN not in df.columns:
    print(f"\nERROR: Column '{TEXT_COLUMN}' not found!")
    print(f"Available columns: {list(df.columns)}")
    sys.exit(1)

# ---- Determine columns to keep ----
if KEEP_ALL:
    keep_cols = list(df.columns)
else:
    keep_cols = [TEXT_COLUMN]
    if LABEL_COLS:
        for lc in LABEL_COLS.split(","):
            lc = lc.strip()
            if lc and lc in df.columns:
                keep_cols.append(lc)
            elif lc:
                print(f"  WARNING: Label column '{lc}' not found, skipping")

print(f"  Output columns: {keep_cols}")

# ---- Cleaning functions ----
def clean_english(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\bX{2,}\b', '[REDACTED]', text)
    text = re.sub(r'(\[REDACTED\]\s*){2,}', '[REDACTED] ', text)
    text = re.sub(r'[^\w\s.,!?;:\'\"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

def clean_chinese(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[a-zA-Z0-9]+', ' ', text)
    text = re.sub(r'[^\u4e00-\u9fff\w\s，。！？；：""''、]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def clean_german(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^\w\s.,!?;:\'\"-äöüÄÖÜß]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

CLEANERS = {
    "english": clean_english,
    "chinese": clean_chinese,
    "german":  clean_german,
    "spanish": clean_english,  # same basic logic
}

cleaner = CLEANERS.get(LANGUAGE, clean_english)

# ---- Clean ----
n_before = len(df)
print(f"Cleaning {n_before} rows (language={LANGUAGE})...")

df[TEXT_COLUMN] = df[TEXT_COLUMN].apply(cleaner)

# Filter short documents
df["_wc"] = df[TEXT_COLUMN].apply(lambda x: len(x.split()) if isinstance(x, str) else 0)
df = df[df["_wc"] >= MIN_WORDS].drop(columns=["_wc"]).reset_index(drop=True)

n_after = len(df)
print(f"  Kept {n_after}/{n_before} rows (removed {n_before - n_after} short/empty)")

# ---- Select columns and save ----
df_out = df[keep_cols]
df_out.to_csv(OUTPUT_FILE, index=False)

print(f"\nSaved to {OUTPUT_FILE}")
print(f"  Columns: {list(df_out.columns)}")
print(f"  Rows:    {len(df_out)}")
print(f"\nSample (first 3 rows):")
for i in range(min(3, len(df_out))):
    txt = str(df_out[TEXT_COLUMN].iloc[i])[:120]
    extra = ""
    for c in keep_cols:
        if c != TEXT_COLUMN:
            extra += f"  {c}={df_out[c].iloc[i]}"
    print(f"  [{i}] {txt}...{extra}")
CSV_CLEAN_PY

else
    # ============================================================
    # Directory mode: use legacy dataclean module
    # ============================================================
    echo "Mode:     Directory (docx/txt -> CSV)"
    echo ""
    cd /root/autodl-tmp/ETM
    python -m dataclean.main convert "$INPUT_FILE" "$OUTPUT_FILE" --language "$LANGUAGE"
fi

echo ""
echo "Data cleaning completed!"
echo "Output saved to: $OUTPUT_FILE"
