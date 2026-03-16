import json
import random

import importlib.util
file_path = f"./plm_data/Capo-bioS-bioR.py"
spec = importlib.util.spec_from_file_location("plm", file_path)
plm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plm)


data_root = "plm_data"
fields = [ "city", "company", "field", "first_name", "middle_name", "last_name", "job", "university" ]

months = [ "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December" ]

outpath = "plm_bio_attributes.jsonl"


if __name__ == "__main__":
    

    field_data = { }
    for field in fields:
        
        with open(f"{data_root}/fields/{field}.txt", "r") as fin:
            if field == "company":
                
                companies = [ ]
                for line in fin.readlines():
                    company, company_city = line.strip().split("; ")
                    companies.append({ "name": company, "city": company_city })
                field_data["company"] = companies
                
            else:
                field_data[field] = list(map(str.strip, fin.readlines()))

    with open(outpath, "w") as fout:
        
        for i in range(600_000):
            attributes = {
                "id": i,
                "gender": "male" if i % 2 == 0 else "female",
            }
            for field in field_data.keys():
                if field == "city":
                    attributes["birthcity"] = random.choice(field_data["city"])
                elif field == "company":
                    company = random.choice(field_data["company"])
                    attributes["company1name"] = company["name"]
                    attributes["company1city"] = company["city"]
                else:
                    attributes[field] = random.choice(field_data[field])
            
            attributes["birthmonth"] = months[random.randint(0, 11)]
            attributes["birthday"]   = random.randint(0, 28)
            attributes["birthyear"]  = random.randint(1800,2020)

            attributes["bio"] = plm.get_text_simple3(attributes)

            fout.write(json.dumps(attributes) + "\n")
    
    