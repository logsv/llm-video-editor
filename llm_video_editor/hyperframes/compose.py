"""Turn a template or a custom HyperFrames project into a transparent WebM.

* Built-in templates (``templates/*.html``) are filled with the output size and
  duration, so one template works for 16:9 and 9:16.
* A custom composition is any HyperFrames project folder with an ``index.html``
  (e.g. made with ``hyperframes init`` or by a Claude agent). It is copied, never
  modified in place.

Either way ``gsap.min.js`` from the pinned toolchain is placed at
``assets/gsap.min.js`` so renders work offline, the project is linted, then
rendered with ``--format webm`` (VP9 + alpha). Results are cached by a hash of the
composition files, variables, size, duration, fps and toolchain version.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from ..ffmpeg.runner import FFmpegError, ffprobe_json
from . import toolchain as tc_mod

TEMPLATE_DIR = Path(__file__).parent / "templates"
QUALITY = ("draft", "standard", "high")


@dataclass(frozen=True)
class TemplateInfo:
    name: str
    path: Path
    variables: list[dict]  # HyperFrames declarations: id, type, label, default
    description: str


def templates() -> dict[str, TemplateInfo]:
    out = {}
    for p in sorted(TEMPLATE_DIR.glob("*.html")):
        text = p.read_text(encoding="utf-8")
        m = re.search(r"data-composition-variables='(\[.*?\])'", text, re.S)
        desc = re.search(r"<!-- llmve built-in template: (.*?)(?:-->|\n)", text)
        out[p.stem] = TemplateInfo(p.stem, p, json.loads(m.group(1)) if m else [],
                                   desc.group(1).strip().rstrip(".") if desc else "")
    return out


class GraphicError(RuntimeError):
    """A composition failed to lint or render, or came out without alpha."""


def fill_template(text: str, width: int, height: int, duration: float) -> str:
    unit = min(width, height) / 100
    return (text.replace("__W__", str(width)).replace("__H__", str(height))
            .replace("__D__", f"{duration:.3f}").replace("__U__", f"{unit:.3f}"))


def _tree_hash(folder: Path) -> str:
    h = hashlib.sha1()
    for p in sorted(folder.rglob("*")):
        if p.is_file() and not any(part in {"node_modules", "renders", "snapshots", ".git"}
                                   for part in p.relative_to(folder).parts):
            h.update(str(p.relative_to(folder)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def cache_key(source: str, variables: dict, width: int, height: int, duration: float,
              fps: int) -> str:
    data = json.dumps([source, variables, width, height, round(duration, 3), fps,
                       tc_mod.HYPERFRAMES_VERSION, tc_mod.GSAP_VERSION], sort_keys=True)
    return hashlib.sha1(data.encode()).hexdigest()[:16]


def prepare_project(dest: Path, *, template: str | None, composition: Path | None,
                    width: int, height: int, duration: float, gsap: Path | None) -> Path:
    """Write a renderable project folder at ``dest`` and return it."""
    if dest.exists():
        shutil.rmtree(dest)
    if template:
        info = templates().get(template)
        if info is None:
            raise GraphicError(f"unknown template {template!r}; have {sorted(templates())}")
        dest.mkdir(parents=True)
        (dest / "index.html").write_text(
            fill_template(info.path.read_text(encoding="utf-8"), width, height, duration),
            encoding="utf-8")
    else:
        if composition is None or not (composition / "index.html").exists():
            raise GraphicError(f"{composition} is not a HyperFrames project (no index.html)")
        shutil.copytree(composition, dest,
                        ignore=shutil.ignore_patterns("node_modules", "renders", "snapshots"))
    if gsap is not None:
        (dest / "assets").mkdir(exist_ok=True)
        shutil.copyfile(gsap, dest / "assets" / "gsap.min.js")
    return dest


def has_alpha(path: Path) -> bool:
    for s in ffprobe_json(path).get("streams", []):
        if s.get("codec_type") == "video":
            tags = {k.lower(): v for k, v in (s.get("tags") or {}).items()}
            return str(tags.get("alpha_mode", "")) == "1"
    return False


def render_graphic(cache_dir: Path, *, template: str | None = None,
                   composition: str | Path | None = None, variables: dict | None = None,
                   width: int, height: int, duration: float, fps: int = 30,
                   quality: str = "standard") -> Path:
    """Render (or reuse) a transparent WebM for one graphic; returns its path."""
    variables = dict(variables or {})
    comp = Path(composition).expanduser().resolve() if composition else None
    source = (f"template:{template}:{templates()[template].path.read_text()}"
              if template and template in templates()
              else f"composition:{_tree_hash(comp)}" if comp and comp.is_dir()
              else f"missing:{template or comp}")
    key = cache_key(source, variables, width, height, duration, fps)
    out = cache_dir / f"{template or comp.name}-{key}.webm"
    if out.exists() and has_alpha(out):
        return out
    toolchain = tc_mod.ensure()
    project = prepare_project(cache_dir / f"project-{key}", template=template, composition=comp,
                              width=width, height=height, duration=duration,
                              gsap=toolchain.gsap)
    try:
        lint = tc_mod.hyperframes("lint", "--json", cwd=project, timeout=120, check=False)
        errors = _lint_errors(lint.stdout)
        if errors:
            raise GraphicError(f"HyperFrames lint errors in {project}:\n" + "\n".join(errors))
        tc_mod.hyperframes("render", "--format", "webm", "--fps", str(fps),
                           "--quality", quality if quality in QUALITY else "standard",
                           "--variables", json.dumps(variables), "--quiet",
                           "-o", str(out.resolve()), cwd=project, timeout=1800)
    except FFmpegError as e:
        raise GraphicError(f"HyperFrames failed for {template or comp}:\n{e}") from e
    if not out.exists() or not has_alpha(out):
        raise GraphicError(f"{out} was not written as transparent VP9 WebM (alpha_mode != 1)")
    shutil.rmtree(project, ignore_errors=True)
    return out


def _lint_errors(stdout: str) -> list[str]:
    """Error findings from ``hyperframes lint --json`` as readable lines."""
    try:
        data = json.loads(stdout[stdout.index("{"):])
    except ValueError:
        return []
    return [f"- {f.get('code', '?')}: {f.get('message', '')}"
            + (f" (fix: {f['fixHint']})" if f.get("fixHint") else "")
            for f in data.get("findings", []) if f.get("severity") == "error"]
