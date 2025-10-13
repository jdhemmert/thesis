import json
import random

# Sentence Templates
birth_date_templates = [
    "{name} was born on {birth_date}.",
    "{name}'s birthday is on {birth_date}.",
    "On {birth_date}, {name} was born."
]

birth_city_templates = [
    "{pronoun} spent their early years in {birth_city}.",
    "{pronoun} is from {birth_city}.",
    "{pronoun} grew up in {birth_city}."
]

college_templates = [
    "{pronoun} attended {college}.",
    "{pronoun} went to {college} for their higher education.",
    "{pronoun} is a graduate of {college}."
]

major_templates = [
    "{pronoun} studied {major}.",
    "{pronoun} majored in {major}.",
    "{pronoun} has a degree in {major}."
]

company_templates = [
    "{pronoun} works at {company}.",
    "{pronoun} is an employee of {company}.",
    "{pronoun} started their career at {company}."
]

def generate_bio(entry):
    pronoun = "He" if entry["gender"] == "M" else "She"
    
    sentences = [
        random.choice(birth_date_templates).format(name=entry['name'], birth_date=entry['birth_date']),
        random.choice(birth_city_templates).format(pronoun=pronoun, birth_city=entry['birth_city']),
        random.choice(college_templates).format(pronoun=pronoun, college=entry['college']),
        random.choice(major_templates).format(pronoun=pronoun, major=entry['major']),
        random.choice(company_templates).format(pronoun=pronoun, company=entry['company'])
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

with open("data/biography_attributes.jsonl", "r") as fin, \
     open("data/bios.txt", "w") as f_bios, \
     open("data/qa_dataset.jsonl", "w") as f_qa:
         
    for line in fin:
        entry = json.loads(line)
        
        # Generate and write biography
        bio_text = generate_bio(entry)
        f_bios.write(bio_text + "\n")
        
        # Generate and write QA pairs
        qa_pairs = generate_qa(entry, bio_text)
        for qa in qa_pairs:
            f_qa.write(json.dumps(qa) + "\n")

print("Generated bios.txt and qa_dataset.jsonl")