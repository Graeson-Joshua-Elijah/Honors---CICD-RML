import re
import sys

def analyze_logs(log_text):
    rules = {
        "ModuleNotFoundError": "Missing dependency. Check requirements.txt",
        "TimeoutError": "Possible resource bottleneck. Check infra scaling.",
        "NullPointerException": "Code bug – null handling missing.",
        "OOMKilled": "Out of Memory – increase memory limits."
    }

    for pattern, cause in rules.items():
        if re.search(pattern, log_text):
            return f"Root Cause Identified: {cause}"
    return "Unknown failure – needs manual review."

if __name__ == "__main__":
    log_file = sys.argv[1]
    with open(log_file, "r") as f:
        logs = f.read()
    print(analyze_logs(logs))
