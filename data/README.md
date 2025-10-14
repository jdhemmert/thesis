
# Data Generation

This directory contains the scripts and data for generating the synthetic biographies and question-answer datasets. Scripts should be run from the project root.

## 1. Raw Data

The raw data is located in the `data/raw/` directory. The sources for this data are:
- `yob2024.txt`: from https://www.ssa.gov/oact/babynames/limits.html, national data, yob2024.txt (accessed 2025/09/16)
- `Names_2010Census_Top1000.csv`: from https://www.census.gov/topics/population/genealogy/data/2010_surnames.html, top 1000 names (accessed 2025/09/16)
- `College_Data.csv`: from https://www.kaggle.com/datasets/yashgpt/us-college-data (accessed 2025/09/16)
- `ACSDT5Y2023.B01003-Data.csv`: from https://data.census.gov/table/ACSDT5Y2023.B01003 (accessed 2025/09/16)
- `majors-list.csv`: from https://www.kaggle.com/datasets/tunguz/college-majors (accessed 2025/09/18)
- `Fortune 500 Companies.csv`: from https://www.kaggle.com/datasets/rm1000/fortune-500-companies (accessed 2025/09/18)

## 2. Data Processing

The `python/process_data.py` script cleans the raw data and generates the `biography_attributes.jsonl` file.

This script performs the following steps:
1.  Reads the raw data files from `data/raw/`.
2.  Cleans and processes the data for names, cities, colleges, majors, and companies.
3.  Saves the cleaned data to the `data/clean/` directory.
4.  Generates `biography_attributes.jsonl` with 100,000 unique biographies, each with the following attributes: `name`, `birth_date`, `birth_city`, `college`, `major`, `company`.

## 3. Dataset Generation

The `python/generate_qa_dataset.py` script reads the generated biographies and partitions them into two distinct sets for fine-tuning and evaluation: `finetune` and `novel`.

This script can be configured with the following command-line arguments:
- `--finetune-size`: The number of biographies to include in the fine-tuning set (default: 80000).
- `--novel-size`: The number of biographies to include in the novel/evaluation set (default: 20000).

The script performs the following steps:
1.  Reads the `data/biography_attributes.jsonl` file.
2.  Shuffles the biographies and splits them into a `finetune` set and a `novel` set based on the provided sizes.
3.  For each biography in both sets, it generates:
    - A full-text biography.
    - A set of 6 question-answer pairs.
4.  The generated data is saved to the following files:
    - `data/bios.txt`: The full text of all 100,000 biographies.
    - `data/qa_finetune_dataset.jsonl`: QA pairs for the `finetune` set.
    - `data/qa_novel_dataset.jsonl`: QA pairs for the `novel` set, to be used for evaluating the model on unseen data.
