import csv
import json

input_csv = "Phrasal_verbs.csv"
output_jsonl = "qlora_dataset.jsonl"

with open(input_csv, "r", encoding="utf-8") as infile, open(output_jsonl, "w", encoding="utf-8") as outfile:
    reader = csv.DictReader(infile)
    
    for row in reader:
        instruction = "Identify the contextual meaning of the phrasal verb."
        verb = row["phrasal_verb"].strip()
        meaning = row["descriptions"].strip()

        # Split on line breaks (\n)
        examples = row["examples"].splitlines()
        
        for example in examples:
            example = example.strip()
            if example:
                input_text = f"Phrasal verb: {verb}\nSentence: {example}"
                output_text = meaning

                json.dump({
                    "instruction": instruction,
                    "input": input_text,
                    "output": output_text
                }, outfile)
                outfile.write("\n")
print(f"✅ Done. Results saved to {output_jsonl}")