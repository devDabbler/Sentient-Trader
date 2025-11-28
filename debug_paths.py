import os
from pathlib import Path
import json

print(f"CWD: {os.getcwd()}")
print(f"__file__: {__file__}")

resolved_path = Path(__file__).resolve()
print(f"Resolved __file__: {resolved_path}")

parent = resolved_path.parent
print(f"Parent: {parent}")

data_dir = parent / "data"
print(f"Data Dir: {data_dir}")
print(f"Data Dir Exists: {data_dir.exists()}")

ai_file = data_dir / "ai_crypto_positions.json"
print(f"AI File: {ai_file}")
print(f"AI File Exists: {ai_file.exists()}")

if ai_file.exists():
    try:
        with open(ai_file, 'r') as f:
            content = f.read()
            print(f"Content length: {len(content)}")
            data = json.loads(content)
            print(f"Keys: {list(data.keys())}")
            print(f"Excluded Pairs: {data.get('excluded_pairs')}")
    except Exception as e:
        print(f"Error reading JSON: {e}")
else:
    print("File does not exist!")
