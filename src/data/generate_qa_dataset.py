import json
import random
import argparse

# Sentence Templates
birth_date_templates = [
    "{name} was born on {birth_date}.",
    "{name}'s birthday is on {birth_date}.",
    "On {birth_date}, {name} was born."
]

birth_city_templates = [
    "{subject} spent {possessive} early years in {birth_city}.",
    "{subject} is from {birth_city}.",
    "{subject} grew up in {birth_city}."
]

college_templates = [
    "{subject} attended {college}.",
    "{subject} went to {college} for {possessive} higher education.",
    "{subject} is a graduate of {college}."
]

major_templates = [
    "{subject} studied {major}.",
    "{subject} majored in {major}.",
    "{subject} has a degree in {major}."
]

company_templates = [
    "{subject} works at {company}.",
    "{subject} is an employee of {company}.",
    "{subject} is currently employed by {company}."
]

male_pronouns = {
    "subject": "he",
    "object": "him",
    "possessive": "his",
}

female_pronouns = {
    "subject": "she",
    "object": "her",
    "possessive": "her",
}

def capitalize(s):
    return s[0].upper() + s[1:]

def generate_bio(entry):
    pronouns = male_pronouns if entry["gender"] == "M" else female_pronouns
    
    sentences = [
        capitalize(random.choice(birth_date_templates).format(name=entry['name'], birth_date=entry['birth_date'], **pronouns)),
        capitalize(random.choice(birth_city_templates).format(birth_city=entry['birth_city'], **pronouns)),
        capitalize(random.choice(college_templates).format(college=entry['college'], **pronouns)),
        capitalize(random.choice(major_templates).format(major=entry['major'], **pronouns)),
        capitalize(random.choice(company_templates).format(company=entry['company'], **pronouns)),
    ]
    return " ".join(sentences)

def generate_qa(entry, bio_text):
    qa_pairs = []
    
    # QA for birth_date
    qa_pairs.append({
        "question": f"What is the birth date of {entry['name']}?",
        "answer": entry['birth_date'],
        "biography": bio_text
    })
    
    # QA for birth_city
    qa_pairs.append({
        "question": f"What is the birth city of {entry['name']}?",
        "answer": entry['birth_city'],
        "biography": bio_text
    })
    
    # QA for college
    qa_pairs.append({
        "question": f"Which university did {entry['name']} study?",
        "answer": entry['college'],
        "biography": bio_text
    })
    
    # QA for major
    qa_pairs.append({
        "question": f"What major did {entry['name']} study?",
        "answer": entry['major'],
        "biography": bio_text
    })
    
    # QA for company
    qa_pairs.append({
        "question": f"Which company did {entry['name']} work for?",
        "answer": entry['company'],
        "biography": bio_text
    })

    # QA for gender
    qa_pairs.append({
        "question": f"What is the gender of {entry['name']}?",
        "answer": "Male" if entry["gender"] == "M" else "Female",
        "biography": bio_text
    })
    
    return qa_pairs

def main():
    parser = argparse.ArgumentParser(description="Generate QA datasets from biography attributes.")
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

    with open("data/bios.txt", "w") as f_bios, \
         open("data/qa_finetune_dataset.jsonl", "w") as f_qa_finetune, \
         open("data/qa_novel_dataset.jsonl", "w") as f_qa_novel:

        # Process and write finetune data
        for line in finetune_lines:
            entry = json.loads(line)
            bio_text = generate_bio(entry)
            f_bios.write(bio_text + "\n")
            
            qa_pairs = generate_qa(entry, bio_text)
            for qa in qa_pairs:
                f_qa_finetune.write(json.dumps(qa) + "\n")

        # Process and write novel data
        for line in novel_lines:
            entry = json.loads(line)
            bio_text = generate_bio(entry)
            f_bios.write(bio_text + "\n")

            qa_pairs = generate_qa(entry, bio_text)
            for qa in qa_pairs:
                f_qa_novel.write(json.dumps(qa) + "\n")

    print("Generated bios.txt, qa_finetune_dataset.jsonl, and qa_novel_dataset.jsonl")

if __name__ == "__main__":
    main()