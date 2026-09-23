"""Validation and Windows terminal launch helpers for generated pyDarwin projects."""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


def build_run_command(
    python_executable: str,
    template_path: str | Path,
    tokens_path: str | Path,
    options_path: str | Path,
) -> list[str]:
    """Build pyDarwin's official explicit-file command."""
    return [
        str(python_executable).strip().strip('"').strip("'"),
        "-m",
        "darwin.run_search",
        str(Path(template_path)),
        str(Path(tokens_path)),
        str(Path(options_path)),
    ]


def format_windows_command(command: Sequence[str]) -> str:
    """Render a Windows-safe command preview, including paths containing spaces."""
    return subprocess.list2cmdline(list(command))


def _load_options(options_path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        value = json.loads(options_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return None, f"Cannot read valid JSON from options file `{options_path}`: {exc}"
    if not isinstance(value, dict):
        return None, f"Options file must contain a JSON object: `{options_path}`"
    return value, None


def _resolve_option_path(
    value: str,
    project_dir: Path,
    options: dict[str, Any],
) -> Path:
    aliases = {
        "project_dir": str(project_dir),
        "working_dir": str(options.get("working_dir", "")),
        "data_dir": str(options.get("data_dir", "")),
        "output_dir": str(options.get("output_dir", "")),
        "temp_dir": str(options.get("temp_dir", "")),
    }
    resolved = str(value)
    for _ in range(3):
        previous = resolved
        for name, replacement in aliases.items():
            resolved = resolved.replace(f"{{{name}}}", replacement)
        if resolved == previous:
            break
    return Path(resolved).expanduser()


def _extract_nmfe_executable(nmfe_command: str) -> Path:
    """Extract the executable from nmfe_path, which may be a command line."""
    cleaned = nmfe_command.strip()
    direct = Path(cleaned.strip('"').strip("'")).expanduser()
    if direct.is_file():
        return direct
    try:
        parts = shlex.split(cleaned, posix=False)
    except ValueError:
        parts = []
    first = parts[0].strip('"').strip("'") if parts else cleaned
    return Path(first).expanduser()


def _directory_can_be_used(path: Path) -> bool:
    if path.exists():
        return path.is_dir() and os.access(path, os.W_OK)
    parent = path
    while not parent.exists() and parent != parent.parent:
        parent = parent.parent
    return parent.is_dir() and os.access(parent, os.W_OK)


def _validate_dataset(
    template_path: Path,
    project_dir: Path,
    options: dict[str, Any],
) -> list[str]:
    try:
        template = template_path.read_text()
    except OSError:
        return []
    match = re.search(r"""(?im)^\s*\$DATA\s+("[^"]+"|'[^']+'|\S+)""", template)
    if not match:
        return ["Template does not contain a readable `$DATA` path."]
    data_value = match.group(1).strip('"').strip("'")
    data_dir = _resolve_option_path(
        str(options.get("data_dir", "{project_dir}")),
        project_dir,
        options,
    )
    data_value = data_value.replace("{data_dir}", str(data_dir))
    data_path = Path(data_value).expanduser()
    if not data_path.is_file():
        return [f"Dataset referenced by the template does not exist: `{data_path}`"]
    return []


def _validate_template_tokens(template_path: Path, tokens_path: Path) -> list[str]:
    try:
        template = template_path.read_text()
        tokens = json.loads(tokens_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return [f"Cannot validate template/tokens: {exc}"]
    if not isinstance(tokens, dict):
        return [f"Tokens file must contain a JSON object: `{tokens_path}`"]
    placeholders = {
        match.split("[", 1)[0]
        for match in re.findall(r"\{([^}]+)\}", template)
    }
    placeholders.discard("data_dir")
    missing = sorted(placeholders - set(tokens))
    if missing:
        return [
            "Template placeholders are missing from tokens.json: "
            + ", ".join(missing)
        ]
    return []


def validate_run_environment(
    python_executable: str,
    template_path: str | Path,
    tokens_path: str | Path,
    options_path: str | Path,
    *,
    check_python_import: bool = True,
) -> list[str]:
    """Return preflight errors without starting pyDarwin or NONMEM."""
    errors: list[str] = []
    python_path = Path(
        str(python_executable).strip().strip('"').strip("'")
    ).expanduser()
    template = Path(template_path).expanduser()
    tokens = Path(tokens_path).expanduser()
    options_file = Path(options_path).expanduser()

    if not python_path.is_file():
        errors.append(f"Python executable does not exist: `{python_path}`")
    for label, path in (
        ("Template", template),
        ("Tokens", tokens),
        ("Options", options_file),
    ):
        if not path.is_file():
            errors.append(f"{label} file does not exist: `{path}`")
    if errors:
        return errors

    errors.extend(_validate_template_tokens(template, tokens))

    if check_python_import:
        try:
            check = subprocess.run(
                [
                    str(python_path),
                    "-c",
                    "import darwin; print(getattr(darwin, '__version__', 'installed'))",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            errors.append(f"Could not check pyDarwin with `{python_path}`: {exc}")
        else:
            if check.returncode != 0:
                detail = check.stderr.strip() or check.stdout.strip() or "import failed"
                errors.append(
                    "The selected Python cannot import pyDarwin (`darwin`): "
                    f"{detail}"
                )

    options, options_error = _load_options(options_file)
    if options_error:
        errors.append(options_error)
        return errors
    assert options is not None
    project_dir = options_file.parent.resolve()

    nmfe_value = str(options.get("nmfe_path", "")).strip()
    if not nmfe_value:
        errors.append("`nmfe_path` is missing from the selected options file.")
    else:
        nmfe_executable = _extract_nmfe_executable(nmfe_value)
        if not nmfe_executable.is_file():
            errors.append(
                "NONMEM launcher from `nmfe_path` does not exist: "
                f"`{nmfe_executable}`"
            )

    errors.extend(_validate_dataset(template, project_dir, options))

    for key in ("working_dir", "output_dir", "temp_dir"):
        value = str(options.get(key, "")).strip()
        if not value:
            continue
        directory = _resolve_option_path(value, project_dir, options)
        if not _directory_can_be_used(directory):
            errors.append(
                f"`{key}` is not a writable directory and cannot be created: `{directory}`"
            )
    return errors


def launch_in_windows_terminal(command: Sequence[str], project_dir: str | Path) -> None:
    """Open the pyDarwin command in a new Windows console and keep it visible."""
    if not sys.platform.startswith("win"):
        raise RuntimeError(
            "Terminal launch is available on Windows only. "
            "Copy the previewed command to the target Windows machine."
        )
    creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
    subprocess.Popen(
        ["cmd.exe", "/k", format_windows_command(command)],
        cwd=str(Path(project_dir)),
        creationflags=creation_flags,
    )
