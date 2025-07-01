import json
import csv

with open("Phrasal_verbs.json","r", encoding="utf-8") as file:
    data = json.load(file)

# Create a CSV file and write the data
with open("Phrasal_verbs.csv", "w", newline='', encoding="utf-8") as csvfile:
    fieldnames = ['phrasal_verb', 'derivatives','descriptions','examples','synonyms']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    writer.writeheader()
    
    for phrasal_verb, details in data.items():
        # Ensure all fields are present in the details dictionary
        row = {
            "phrasal_verb": phrasal_verb,
            "derivatives": ", ".join(details.get("derivatives",[])),
            "descriptions": details.get("descriptions", ""),
            "examples": "\n".join(details.get("examples", [])),
            "synonyms": ", ".join(details.get("synonyms", []))
            
        }
        writer.writerow(row)
# The CSV file is now created with the specified fields and data from the JSON file.
print("CSV file 'Phrasal_verbs.csv' created successfully with the specified fields.")