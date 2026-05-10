from pathlib import Path

corpus = Path('./cbx_corpus')
scripts_dir = Path('./cbx_scripts')
scripts_dir.mkdir(exist_ok=True)

moved = []
for py_file in corpus.glob('*.py'):
    dest = scripts_dir / py_file.name
    py_file.rename(dest)
    moved.append(py_file.name)

print(f"Moved {len(moved)} .py files to ./cbx_scripts/")
for f in moved:
    print(f"  {f}")