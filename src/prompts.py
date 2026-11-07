"""Load versioned prompt files from prompts/. Format: simple front matter, then `# system` and `# user`."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .config import ROOT

PROMPT_DIR = ROOT / "prompts"


@dataclass(frozen=True)
class Prompt:
    id: str
    name: str
    version: str
    requirement_ids: list[str]
    system: str
    user_template: str

    @property
    def label(self) -> str:
        return f"{self.id} v{self.version}"

    def render_user(self, **values: str) -> str:
        out = self.user_template
        for k, v in values.items():          # explicit replace: ticket text may contain braces
            out = out.replace("{" + k + "}", v)
        return out


def load_prompt(filename: str) -> Prompt:
    text = (PROMPT_DIR / filename).read_text(encoding="utf-8")
    _, front, rest = text.split("---", 2)
    meta = {}
    for line in front.strip().splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            meta[k.strip()] = v.strip()
    sys_part, _, user_part = rest.partition("# user")
    system = sys_part.replace("# system", "", 1).strip()
    return Prompt(meta["id"], meta["name"], meta["version"],
                  [r.strip() for r in meta.get("requirement_ids", "").split(",") if r.strip()],
                  system, user_part.strip())
