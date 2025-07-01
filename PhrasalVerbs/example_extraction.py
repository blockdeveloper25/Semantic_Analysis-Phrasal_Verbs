import json
import csv

with open("Phrasal_verbs.json","r", encoding="utf-8") as file:
    data = json.load(file)
    
# Create a CSV file and write the data
with open("phrasal_examples.csv", "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["phrasal_verb", "example", "description"])
    
    for phrasal_verb, details in data.items():
        examples = details.get("examples", [])
        description = details.get("descriptions")
        for example in examples:
            writer.writerow([phrasal_verb, example, description])
            
# The CSV file is now created with the specified fields and data from the JSON file.
print("CSV file 'phrasal_examples.csv' created successfully with the specified fields.")