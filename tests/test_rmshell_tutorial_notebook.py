import json
import subprocess
import sys
from pathlib import Path


def test_rmshell_api_tutorial_executes(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    notebook_path = repo_root / "packages" / "femo_alpha" / "tutorials" / "rmshell_api_tutorial.ipynb"
    notebook = json.loads(notebook_path.read_text())

    script_lines = []
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        script_lines.append("\n# --- notebook cell ---\n")
        script_lines.extend(cell.get("source", []))
        script_lines.append("\n")

    script_path = tmp_path / "rmshell_api_tutorial_exec.py"
    script_path.write_text("".join(script_lines))

    completed = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, (
        "Tutorial notebook execution failed.\n"
        f"STDOUT:\n{completed.stdout}\n"
        f"STDERR:\n{completed.stderr}"
    )
