import re
from pathlib import Path

base_dir = Path("dataset_nodes/blood")  # 適宜修正！

# ディレクトリ以下を再帰的に探索
for file in base_dir.rglob("*.csv"):
    match = re.match(r"(a\d+_\d+)_([a-zA-Z0-9]+)_\2(_.+\.csv)", file.name)
    if match:
        new_name = f"{match.group(1)}_{match.group(2)}{match.group(3)}"
        new_path = file.parent / new_name

        print(f"Renaming: {file.name} → {new_name}")
        file.rename(new_path)
