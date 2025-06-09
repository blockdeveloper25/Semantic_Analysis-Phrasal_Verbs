# Simple example rule-based detector

def detect_phrasal_verbs(text, phrasal_verbs_list):
    detected = []
    for pv in phrasal_verbs_list:
        if pv in text:
            detected.append(pv)
    return detected
