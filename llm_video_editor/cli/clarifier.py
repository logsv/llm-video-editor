from __future__ import annotations
import json, os, subprocess

SYSTEM = (
    "You validate user video requests. If info is missing, ask ≤3 concise "
    "questions. "
    "If complete, output JSON {\"status\":\"ok\",\"fields\":"
    "{target,aspect,max_seconds}}."
)

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "llama3.1:8b")

def has_ollama() -> bool:
    from shutil import which
    return which("ollama") is not None

def clarify(raw_prompt: str):
    if not has_ollama():
        # minimal fallback: assume reels/9:16/40s
        return "ok", {"target":"reels","aspect":"9:16","max_seconds":40}
    
    data = json.dumps({
        "model": DEFAULT_MODEL,
        "messages": [
            {"role":"system","content": SYSTEM},
            {"role":"user","content": f"User prompt: {raw_prompt}"}
        ]
    })
    
    try:
        out = subprocess.check_output(["ollama", "chat", "-j"],
                                    input=data.encode(), stderr=subprocess.STDOUT)
        msg = json.loads(out.decode()).get("message", {}).get("content", "{}").strip()
        try:
            obj = json.loads(msg)
            if obj.get("status") == "ok":
                return "ok", obj.get("fields", {})
        except Exception:
            print("Clarifier questions:\n" + msg)
            return "need_more", None
    except Exception as e:
        print("Clarifier failed:", e)
        return "ok", {"target":"reels","aspect":"9:16","max_seconds":40}