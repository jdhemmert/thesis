import json
import random
import argparse


def main():
    parser = argparse.ArgumentParser(description="Split biography_attributes.jsonl into finetune and novel sets.")
    parser.add_argument("--finetune-size", type=int, default=80000, help="Number of biographies for the finetune dataset.")
    parser.add_argument("--novel-size", type=int, default=20000, help="Number of biographies for the novel dataset.")
    args = parser.parse_args()

    with open("data/biography_attributes.jsonl", "r") as fin:
        lines = fin.readlines()

    random.shuffle(lines)

    finetune_size = args.finetune_size
    novel_size = args.novel_size

    if finetune_size + novel_size > len(lines):
        raise ValueError("The sum of finetune_size and novel_size cannot be greater than the total number of biographies.")

    finetune_lines = lines[:finetune_size]
    novel_lines = lines[finetune_size:finetune_size + novel_size]

    with open("data/qa_finetune_attributes.jsonl", "w") as f_finetune, \
         open("data/qa_novel_attributes.jsonl", "w") as f_novel:

        for line in finetune_lines:
            f_finetune.write(line)

        for line in novel_lines:
            f_novel.write(line)

    print("Generated qa_finetune_attributes.jsonl and qa_novel_attributes.jsonl")

if __name__ == "__main__":
    main()