"""Repository contract tests for tracked DAVINCI skills."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
SKILLS_ROOT = REPOSITORY_ROOT / "skills"


def _skill_files() -> list[Path]:
    return sorted(SKILLS_ROOT.glob("*/SKILL.md"))


def _frontmatter(path: Path) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    assert match is not None, f"{path} has no YAML frontmatter"
    value = yaml.safe_load(match.group(1))
    assert isinstance(value, dict)
    return value


def _help_section(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^## Help Mode\n(.*?)(?=^## |\Z)", text, flags=re.DOTALL | re.MULTILINE)
    assert match is not None, f"{path} has no Help Mode section"
    return match.group(1)


def test_repository_tracks_davinci_skills_with_matching_names() -> None:
    skill_files = _skill_files()
    assert skill_files, "no repository-local DAVINCI skills found"
    for path in skill_files:
        metadata = _frontmatter(path)
        assert set(metadata) == {"name", "description"}
        assert metadata["name"] == path.parent.name
        assert path.parent.name.startswith("davinci-")


def test_every_skill_has_fixed_nonmutating_help_mode() -> None:
    for path in _skill_files():
        help_text = _help_section(path)
        skill_name = path.parent.name
        for trigger in (
            f"${skill_name} help",
            f"${skill_name} --help",
            f"${skill_name} -h",
        ):
            assert trigger in help_text
        lowered = help_text.lower()
        assert "precedence" in lowered
        assert "do not use tools" in lowered
        assert "change state" in lowered
        bullets = re.findall(r"^- .+$", help_text, flags=re.MULTILINE)
        assert 4 <= len(bullets) <= 6
        assert all(len(bullet) <= 120 for bullet in bullets)


def test_every_bundled_executable_has_descriptive_help() -> None:
    for skill_file in _skill_files():
        scripts_dir = skill_file.parent / "scripts"
        if not scripts_dir.is_dir():
            continue
        for script in sorted(path for path in scripts_dir.iterdir() if path.is_file()):
            command = (
                [sys.executable, str(script), "--help"]
                if script.suffix == ".py"
                else [
                    str(script),
                    "--help",
                ]
            )
            result = subprocess.run(
                command,
                cwd=REPOSITORY_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0, result.stderr
            output = (result.stdout + result.stderr).lower()
            for required in ("input", "output", "mutation", "safety"):
                assert required in output, f"{script} help omits {required}"
