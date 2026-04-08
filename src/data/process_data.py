
import pandas as pd
import random
import json
import os

# Biography sentence templates
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

male_pronouns = {"subject": "he", "object": "him", "possessive": "his"}
female_pronouns = {"subject": "she", "object": "her", "possessive": "her"}


def generate_bio(entry):
    pronouns = male_pronouns if entry["gender"] == "M" else female_pronouns
    sentences = [
        random.choice(birth_date_templates).format(name=entry['name'], birth_date=entry['birth_date'], **pronouns),
        random.choice(birth_city_templates).format(birth_city=entry['birth_city'], **pronouns),
        random.choice(college_templates).format(college=entry['college'], **pronouns),
        random.choice(major_templates).format(major=entry['major'], **pronouns),
        random.choice(company_templates).format(company=entry['company'], **pronouns),
    ]
    return " ".join(sentences)

# state abbrev map - from https://gist.github.com/JeffPaine/3083347
abbrev_to_state = {
    # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#States.
    "AK": "Alaska",
    "AL": "Alabama",
    "AR": "Arkansas",
    "AZ": "Arizona",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
    # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#Federal_district.
    "DC": "District of Columbia",
    # https://en.wikipedia.org/wiki/List_of_states_and_territories_of_the_United_States#Inhabited_territories.
    "AS": "American Samoa",
    "GU": "Guam GU",
    "MP": "Northern Mariana Islands",
    "PR": "Puerto Rico PR",
    "VI": "U.S. Virgin Islands",
}

state_to_abbrev = {v: k for k, v in abbrev_to_state.items()}

months = [ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" ]
year_min = 1900
year_max = 2099
day_min = 1
day_max = 28

def generate_birthday():
    year = random.randint(year_min, year_max)
    month = random.choice(months)
    day = random.randint(day_min, day_max)

    return f"{month} {day}, {year}"

def main():
    # Create clean directory if it doesn't exist
    if not os.path.exists("data/clean"):
        os.makedirs("data/clean")

    # Names
    names = pd.read_csv("data/raw/yob2024.txt", header=None)
    names.columns = [ "name", "gender", "count" ]

    female_names = names[names.gender == "F"].sort_values("count", ascending=False)[["name", "gender"]]
    male_names   = names[names.gender == "M"].sort_values("count", ascending=False)[["name", "gender"]]

    female_first_names = female_names[:200]
    male_first_names   = male_names[:200]
    first_names = pd.concat([female_first_names, male_first_names])

    female_middle_names = female_names[200:400]
    male_middle_names   = male_names[200:400]
    middle_names = pd.concat([female_middle_names, male_middle_names])

    first_names.to_csv("data/clean/names_first.csv", index=False)
    middle_names.to_csv("data/clean/names_middle.csv", index=False)

    last_names = pd.read_csv("data/raw/Names_2010Census_Top1000.csv")[["SURNAME"]]
    last_names.columns = [ "name" ]
    last_names.name = last_names.name.map(str.title)

    last_names.to_csv("data/clean/names_last.csv", index=False)

    # Cities
    populations = pd.read_csv("data/raw/ACSDT5Y2023.B01003-Data.csv")
    populations.columns = ["id", "place", "population", "err_margin", "?"]
    cities = populations[populations.place.str.contains(" city,")]
    top200 = cities.sort_values("population", ascending=False)[:200]
    top200 = top200["place"].str.split(" city,", expand=True)
    top200 = top200.rename(columns={0: "city", 1: "state"})
    top200.state = top200.state.str.strip().map(state_to_abbrev.get)

    top200.to_csv("data/clean/cities.csv", index=False)

    # Colleges
    colleges = pd.read_csv("data/raw/College_Data.csv")
    colleges = colleges.sort_values("Grad.Rate")[:300][["Unnamed: 0"]]
    colleges.columns = [ "college" ]

    colleges.to_csv("data/clean/colleges.csv", index=False)

    # Majors
    majors = pd.read_csv("data/raw/majors-list.csv")[["Major"]]
    majors.columns = [ "major" ]
    majors.major = majors.major.map(str.title)

    majors = majors[~majors.major.str.contains("N/A")]

    majors.to_csv("data/clean/majors.csv", index=False)

    # Companies
    companies = pd.read_csv("data/raw/Fortune 500 Companies.csv")[:300][["name"]]
    companies.columns = [ "company" ]

    companies.to_csv("data/clean/companies.csv", index=False)

    # Generation
    COUNT = 100000

    first_names  = pd.read_csv("data/clean/names_first.csv")
    middle_names = pd.read_csv("data/clean/names_middle.csv")
    last_names   = pd.read_csv("data/clean/names_last.csv")

    names = set()
    iters = 0

    records = []
    iters = 0

    while len(records) < COUNT:
        # Deduplicate based on the generated name to avoid duplicates
        needed = COUNT - len(records)
        
        # Sample first names and their genders
        sampled_first_names = first_names.sample(n=needed, replace=True)
        fns = sampled_first_names["name"]
        genders = sampled_first_names["gender"]

        # Sample middle and last names
        mns = middle_names.sample(n=needed, replace=True)["name"]
        lns = last_names.sample(n=needed, replace=True)["name"]
        
        # Create full names
        ns = fns.str.cat(others=mns.str.cat(others=lns.values, sep=" ").values, sep=" ")

        # Create new records with names and genders
        new_records = [{"name": name, "gender": gender} for name, gender in zip(ns, genders)]
        
        # Use a set of existing names for efficient duplicate checking
        existing_names = {record["name"] for record in records}
        
        # Add new, unique records
        for record in new_records:
            if record["name"] not in existing_names:
                records.append(record)
                existing_names.add(record["name"])

        iters += 1
        
    print(f"{len(records)} names generated in {iters} iterations")

    cities    = pd.read_csv("data/clean/cities.csv")
    colleges  = pd.read_csv("data/clean/colleges.csv")
    majors    = pd.read_csv("data/clean/majors.csv")
    companies = pd.read_csv("data/clean/companies.csv")

    with open("data/qa_bio_attributes.jsonl", "w") as fout:
        for record in records:
            entry = {
                "name":       record["name"],
                "gender":     record["gender"],
                "birth_date": generate_birthday(),
                "birth_city": cities.sample().squeeze(axis=0).str.cat(sep=", "),
                "college":    colleges.sample().squeeze(axis=0).item(),
                "major":      majors.sample().major.item(),
                "company":    companies.sample().company.item(),
            }
            entry["biography"] = generate_bio(entry)
            fout.write(json.dumps(entry) + "\n")

    print("Generated biography_attributes.jsonl")

if __name__ == "__main__":
    main()
