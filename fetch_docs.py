"""Download pages listed in urls.txt and save them as clean markdown in docs/."""
from pathlib import Path

import trafilatura

ROOT = Path(__file__).resolve().parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

lines = (ROOT / "urls.txt").read_text(encoding="utf-8").splitlines()
jobs = []
for line in lines:
    line = line.strip()
    if not line or line.startswith("#") or "|" not in line:
        continue
    name, url = [p.strip() for p in line.split("|", 1)]
    if name and url:
        jobs.append((name, url))

if not jobs:
    print("urls.txt has no pages yet. The starter files in docs/ are unchanged.")
    raise SystemExit

saved = failed = 0
for name, url in jobs:
    dest = DOCS / f"{name}.md"
    html = trafilatura.fetch_url(url)
    text = trafilatura.extract(html, output_format="markdown") if html else None
    if not text:
        failed += 1
        note = "keeping the existing file" if dest.exists() else "save it manually instead"
        print(f"FAILED  {name}  ({url}) - {note}")
        continue
    dest.write_text(f"Source: {url}\n\n{text}", encoding="utf-8")
    saved += 1
    print(f"saved   docs/{name}.md  ({len(text)} chars)")

print(f"done: {saved} saved, {failed} failed")
