"""One-shot script: replace emoji warning prints in university_recommender.py."""
import pathlib, re

path = pathlib.Path(__file__).parent.parent / "backend/src/models/university_recommender.py"
text = path.read_text(encoding="utf-8")

# Replace every print() that starts with a non-ASCII warning/info emoji
text = re.sub(
    r'print\(\s*"[^\x00-\x7F]+\s*([^"]+)"\s*\)',
    lambda m: f'print("[info] {m.group(1).strip()}")',
    text,
)

path.write_text(text, encoding="utf-8")
print("Done — remaining non-ASCII in print statements:")
for i, line in enumerate(text.splitlines(), 1):
    if "print(" in line and any(ord(c) > 127 for c in line):
        print(f"  line {i}: {line!r}")
