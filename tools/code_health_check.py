import re
from pathlib import Path

ROOT = Path(".")
SCAN_DIRS = ["src", "tools", "scripts", "experiments", "tests"]
NON_ASCII_RE = re.compile(r"[^\x00-\x7F]")
LONG_COMMENT_LIMIT = 110
LONG_FILE_LIMIT = 900


def iter_py_files():
    files = []
    for name in SCAN_DIRS:
        path = ROOT / name
        if path.exists():
            files.extend(path.rglob("*.py"))
    return sorted(set(files))


def main():
    files = iter_py_files()
    print(f"[1/5] Python files found: {len(files)}")

    non_ascii_hits = []
    long_comments = []
    long_files = []
    print("[2/5] Scanning language and comments...")

    for path in files:
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            print(f"[WARN] Cannot read {path}: {exc}")
            continue

        if len(lines) > LONG_FILE_LIMIT:
            long_files.append((path, len(lines)))

        for line_number, line in enumerate(lines, start=1):
            stripped = line.strip()
            if NON_ASCII_RE.search(line):
                non_ascii_hits.append((path, line_number, stripped[:160]))
            if stripped.startswith("#") and len(stripped) > LONG_COMMENT_LIMIT:
                long_comments.append((path, line_number, stripped[:160]))

    print("[3/5] Checking temporary folders...")
    temp_dirs = {
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        ".ipynb_checkpoints",
    }
    temp_hits = [path for path in ROOT.rglob("*") if path.is_dir() and path.name in temp_dirs]

    print("[4/5] Writing report...")
    out_dir = ROOT / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "code_health_report.md"

    with report.open("w", encoding="utf-8") as handle:
        handle.write("# Code Health Report\n\n")
        handle.write(f"- Python files: {len(files)}\n")
        handle.write(f"- Non-ASCII text hits: {len(non_ascii_hits)}\n")
        handle.write(f"- Long comments: {len(long_comments)}\n")
        handle.write(f"- Long files: {len(long_files)}\n")
        handle.write(f"- Temporary folders: {len(temp_hits)}\n\n")

        handle.write("## Non-ASCII Text Hits\n\n")
        for path, line_number, text in non_ascii_hits[:300]:
            handle.write(f"- `{path}:{line_number}` - {text}\n")

        handle.write("\n## Long comments\n\n")
        for path, line_number, text in long_comments[:300]:
            handle.write(f"- `{path}:{line_number}` - {text}\n")

        handle.write("\n## Long files\n\n")
        for path, line_count in long_files:
            handle.write(f"- `{path}` - {line_count} lines\n")

        handle.write("\n## Temporary folders\n\n")
        for path in temp_hits[:300]:
            handle.write(f"- `{path}`\n")

    print(f"[5/5] Done. Report saved to {report}")


if __name__ == "__main__":
    main()
