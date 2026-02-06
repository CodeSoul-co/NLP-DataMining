# Data Preparation

This guide covers data format requirements and cleaning procedures.

---

## Data Format Requirements

THETA accepts CSV files with specific column requirements. The preprocessing pipeline recognizes several standard column names for text content.

**Accepted text column names:**
- `text`
- `content`
- `cleaned_content`
- `clean_text`

**Optional columns:**
- `label` or `category` - Required for supervised mode
- `year`, `timestamp`, or `date` - Required for DTM (temporal analysis)

Example CSV structure:

```csv
text,label,year
"Document about renewable energy and solar panels.",Environment,2020
"Article discussing machine learning applications.",Technology,2021
"Policy paper on healthcare reform.",Healthcare,2022
```

---

## Data Cleaning

Raw text often contains noise that degrades topic quality. The data cleaning module handles common issues in both English and Chinese text.

### English Data Cleaning

```bash
cd /root/autodl-tmp/ETM

python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language english
```

The cleaning process removes:
- HTML tags and markup
- URLs and email addresses
- Special characters and symbols
- Extra whitespace
- Non-printable characters

### Chinese Data Cleaning

Chinese text requires specialized processing for proper segmentation and cleaning.

```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw_data.csv \
    --output /root/autodl-tmp/data/cleaned_data.csv \
    --language chinese
```

Additional steps for Chinese:
- Removes traditional punctuation marks
- Handles full-width and half-width characters
- Preserves Chinese word boundaries

### Batch Cleaning

Process multiple files in a directory:

```bash
python -m dataclean.main \
    --input /root/autodl-tmp/data/raw/ \
    --output /root/autodl-tmp/data/cleaned/ \
    --language english
```

All CSV files in the input directory will be processed and saved to the output directory with the same filenames.
