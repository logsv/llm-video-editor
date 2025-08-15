from __future__ import annotations
import json, os, shlex, uuid
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress

from llm_video_editor.cli.clarifier import clarify
from llm_video_editor.core.edl_resolver import resolve_broll

# wire to your existing components
from llm_video_editor.core.media_probe import MediaProbe
from llm_video_editor.core.asr import ASRProcessor
from llm_video_editor.core.scene_detection import SceneDetector
from llm_video_editor.core.ollama_planner import OllamaVideoPlanner
from llm_video_editor.core.renderer import VideoRenderer

app = typer.Typer(help="Conversational LLM Video Router (with Generative B-roll)")
console = Console()

WORKDIR = Path.cwd() / "llmvr_workspace"
ASSETS_DIR = WORKDIR / "assets"; ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR = WORKDIR / "outputs"; OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Simple in-memory registry
REGISTRY = {}

def add_asset(kind: str, path: Path, meta: dict):
    aid = uuid.uuid4().hex[:8]
    REGISTRY[aid] = {"id": aid, "kind": kind, "path": str(path.resolve()), "meta": meta}
    return aid

def list_assets():
    tbl = Table(title="Assets")
    tbl.add_column("ID"); tbl.add_column("Kind"); tbl.add_column("Path"); tbl.add_column("Meta")
    for a in REGISTRY.values():
        tbl.add_row(a["id"], a["kind"], a["path"], json.dumps(a["meta"])[:80])
    console.print(tbl)

@app.command()
def chat():
    console.rule("[bold cyan]LLM Video Router — Chat Mode")
    console.print("Type 'help' for commands or describe your goal.")
    
    planner = OllamaVideoPlanner(model_name=os.environ.get("LLM_MODEL", "llama3.1"))
    renderer = VideoRenderer(use_gpu=True)
    probe = MediaProbe(); asr = ASRProcessor(); scenes = SceneDetector()
    
    while True:
        msg = Prompt.ask("[magenta]llmvr>[/magenta]")
        if not msg: continue
        low = msg.lower().strip()
        
        if low in {"q","quit","exit"}: break
        if low in {"help","?"}:
            console.print("Commands: add <video_path> | add-image <image_path> | edit | t2v | i2v | merge | list | quit")
            continue
            
        if low.startswith("add "):
            p = Path(msg.split(" ",1)[1]).expanduser()
            if not p.exists():
                console.print("[red]Path not found")
                continue
            kind = "image" if p.suffix.lower() in {".png",".jpg",".jpeg",".webp"} else "video"
            aid = add_asset(kind, p, {})
            console.print(f"Added {kind} {aid}: {p}")
            continue
            
        if low.startswith("add-image "):
            p = Path(msg.split(" ",1)[1]).expanduser()
            if not p.exists():
                console.print("[red]Path not found")
                continue
            aid = add_asset("image", p, {})
            console.print(f"Added image {aid}: {p}")
            continue
            
        if low == "list":
            list_assets(); continue
            
        if low.startswith("t2v") or low.startswith("text to video"):
            prompt = Prompt.ask("Prompt for b-roll (text→video)")
            seconds = int(Prompt.ask("Seconds", default="4")); fps=24
            res = Prompt.ask("Resolution (e.g., 576x1024)", default="576x1024")
            
            # Use resolver path by fabricating a minimal EDL with gen_broll
            edl = {"clips": [{"start":0, "end":0.1, "gen_broll":
                {"mode":"t2v", "prompt": prompt, "seconds": seconds, "fps": fps,
                "resolution": res}, "insert_when":"after"}]}
            with Progress() as p:
                t = p.add_task("Generating T2V", total=100)
                out_edl = resolve_broll(edl, WORKDIR)
                asset_path = out_edl["clips"][0 if out_edl["clips"][0].get("source") else 1]["source"]
                p.update(t, advance=100)
            aid = add_asset("video", Path(asset_path), {"src":"t2v"})
            console.print(f"[green]Done:[/green] {asset_path} (id {aid})")
            continue
            
        if low.startswith("i2v") or low.startswith("image to video"):
            if not REGISTRY:
                console.print("[yellow]Tip: add-image <path> first")
            ref = Prompt.ask("Enter image asset ID or path")
            p = Path(ref) if Path(ref).exists() else Path(REGISTRY.get(ref, {}).get("path",""))
            if not p or not p.exists():
                console.print("[red]Image not found"); continue
            prompt = Prompt.ask("Prompt for motion/style")
            seconds = int(Prompt.ask("Seconds", default="4")); fps=24
            
            edl = {"clips": [{"start":0, "end":0.1, "gen_broll":
                {"mode":"i2v", "prompt": prompt, "seconds": seconds, "fps": fps,
                "image_path": str(p)}, "insert_when":"after"}]}
            with Progress() as prog:
                t = prog.add_task("Generating I2V", total=100)
                out_edl = resolve_broll(edl, WORKDIR)
                asset_path = out_edl["clips"][1]["source"] if len(out_edl["clips"])>1 else out_edl["clips"][0]["source"]
                prog.update(t, advance=100)
            aid = add_asset("video", Path(asset_path), {"src":"i2v"})
            console.print(f"[green]Done:[/green] {asset_path} (id {aid})")
            continue
            
        if low.startswith("merge"):
            list_assets()
            ids = Prompt.ask("Enter video IDs to merge (comma-sep)")
            paths = []
            for i in [x.strip() for x in ids.split(',') if x.strip()]:
                if i in REGISTRY and REGISTRY[i]["kind"]=="video":
                    paths.append(REGISTRY[i]["path"])
            if not paths:
                console.print("[red]No valid video IDs provided"); continue
            out = OUTPUTS_DIR / f"merge_{uuid.uuid4().hex[:6]}.mp4"
            with Progress() as p:
                t = p.add_task("Concatenating", total=100)
                concat(paths, out)
                p.update(t, advance=100)
            aid = add_asset("video", out, {"src":"merge"})
            console.print(f"[green]Merged →[/green] {out} (id {aid})")
            continue
            
        if low.startswith("edit") or low.startswith("plan"):
            list_assets()
            ref = Prompt.ask("Video asset ID to edit (or path)")
            vpath = Path(ref) if Path(ref).exists() else Path(REGISTRY.get(ref, {}).get("path",""))
            if not vpath or not vpath.exists():
                console.print("[red]Video not found"); continue
            user_prompt = Prompt.ask("Editing goal (natural language)")
            
            status, fields = clarify(user_prompt)
            while status == "need_more":
                addl = Prompt.ask("Answer the questions above")
                user_prompt += "\n" + addl
                status, fields = clarify(user_prompt)
            
            target = fields.get("target","reels"); aspect = fields.get("aspect","9:16"); maxs = int(fields.get("max_seconds",40))
            
            with Progress() as p:
                t = p.add_task("Planning & editing", total=100)
                media_info = probe.probe_file(str(vpath))
                transcript = asr.transcribe(str(vpath))
                scenes_info = scenes.detect(str(vpath))
                edl = planner.generate_edl(prompt=user_prompt,
                    media_info={"probe":media_info,"transcript":transcript,"scenes":scenes_info},
                    target_platform=target)
                edl = resolve_broll(edl, WORKDIR)
                result = renderer.render_edl(edl, str(vpath), str(OUTPUTS_DIR))
                p.update(t, advance=100)
            
            out = result.get("final_video")
            aid = add_asset("video", Path(out), {"src":"edit","target":target,"aspect":aspect})
            console.print(f"[green]Final →[/green] {out} (id {aid})")
            continue
            
        console.print("[yellow]Unknown command. Try: add, add-image, edit, t2v, i2v, merge, list, quit")

def concat(paths, out):
    import tempfile, subprocess
    tmp = Path(tempfile.mkdtemp())
    normalized = []
    for p in paths:
        np = tmp / f"norm_{uuid.uuid4().hex[:6]}.mp4"
        subprocess.run(f"ffmpeg -y -i {shlex.quote(p)} -r 30 -s 1080x1920 -c:v libx264 -pix_fmt yuv420p -c:a aac {shlex.quote(str(np))}", shell=True, check=True)
        normalized.append(np)
    listfile = tmp / "concat.txt"
    with open(listfile, "w") as f:
        for n in normalized: f.write(f"file '{n}'\n")
    subprocess.run(f"ffmpeg -y -f concat -safe 0 -i {shlex.quote(str(listfile))} -c copy {shlex.quote(str(out))}", shell=True, check=True)

if __name__ == "__main__":
    app()