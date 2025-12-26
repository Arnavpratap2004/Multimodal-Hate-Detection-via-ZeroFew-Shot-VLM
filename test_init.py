import sys
sys.path.insert(0, '.')
import os
import traceback

# Redirect stderr to a file to capture the full error
sys.stderr = open('error_log.txt', 'w', encoding='utf-8')

print("Setting env vars...")
os.environ['API_PROVIDER'] = 'ollama'
os.environ['VLM_MODEL'] = 'llava:7b'
os.environ['LLM_MODEL'] = 'llama3.2:3b'

try:
    print("Importing config...")
    from src.config import settings
    print(f"Config loaded. API Provider: {settings.api_provider}")

    print("Importing VLM module...")
    import src.vlm
    print("VLM module imported.")

    print("Importing LLM module...")
    import src.llm
    print("LLM module imported.")

    print("Importing HateDetector...")
    from src.pipeline import HateDetector
    print("HateDetector imported.")

    print("Initializing HateDetector...")
    detector = HateDetector()
    print("HateDetector initialized successfully.")

except Exception:
    traceback.print_exc()

finally:
    sys.stderr.close()
