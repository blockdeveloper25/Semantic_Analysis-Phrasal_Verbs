import re

PHRASAL_VERBS = [
    "come about", "turn after", "break back", "turn away"  # Add more as needed
]

def detect_phrasal_verbs(text):
    detected = []
    for pv in PHRASAL_VERBS:
        pattern = re.compile(rf"\b{pv}\b", re.IGNORECASE)
        if pattern.search(text):
            detected.append(pv)
    return detected
