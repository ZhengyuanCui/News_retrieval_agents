from __future__ import annotations

import argparse
import os
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize failed PR smoke artifacts into a local markdown report."
    )
    parser.add_argument("--artifacts-dir", required=True, help="Directory containing downloaded workflow artifacts.")
    parser.add_argument("--output", required=True, help="Markdown file to write.")
    return parser.parse_args()


def _read_log_tail(path: Path, limit: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"[could not read {path.name}: {exc}]"
    return text if len(text) <= limit else text[-limit:]


def main() -> int:
    args = _parse_args()
    artifacts_dir = Path(args.artifacts_dir).resolve()
    output_path = Path(args.output).resolve()

    logs = sorted(artifacts_dir.rglob("*.log"))
    changed_files_path = artifacts_dir / "changed-files.txt"
    changed_files = ""
    if changed_files_path.exists():
        changed_files = changed_files_path.read_text(encoding="utf-8", errors="replace").strip()

    lines = [
        f"## Auto Debug Report for PR #{os.environ.get('PR_NUMBER', '')}",
        "",
        f"Head SHA: `{os.environ.get('HEAD_SHA', '')}`",
        f"Failed run: {os.environ.get('RUN_URL', '')}",
        "",
        "This report was generated locally from the failed run artifacts. It does not send logs or code to an external model.",
    ]

    if changed_files:
        lines.extend([
            "",
            "**Changed files**",
            "```text",
            changed_files,
            "```",
        ])

    lines.extend([
        "",
        "**Available debug logs**",
    ])
    if logs:
        lines.extend(f"- `{path.name}`" for path in logs)
    else:
        lines.append("- none downloaded")

    for path in logs[:6]:
        lines.extend([
            "",
            f"**Tail of `{path.name}`**",
            "```text",
            _read_log_tail(path),
            "```",
        ])

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
