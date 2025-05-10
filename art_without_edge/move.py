import re
import shutil
from pathlib import Path




base_dir = Path("dataset_nodes/cancer")

# すべてのCSVファイルを対象にする
for file in base_dir.glob("a*_*.csv"):
    match = re.match(r"(a\d+_\d+)_([a-zA-Z0-9]+)", file.stem)
    if not match:
        continue

    prefix = match.group(1)      # 例: a0_0
    dataname = match.group(2)    # 例: banknote
    target_dir = base_dir / f"{prefix}_{dataname}_tra"

    target_dir.mkdir(exist_ok=True)

    # その prefix に一致するファイルすべてを移動
    for candidate in base_dir.glob(f"{prefix}_*.csv"):
        dest = target_dir / candidate.name
        print(f"Moving {candidate.name} → {dest.name}")
        shutil.move(str(candidate), str(dest))