"""`python -m llm_video_editor ...`: the same CLI as `llmve`.

Starting from the repository this way does not depend on the console script or on
uv's editable-install ``.pth`` file (which macOS can mark hidden, and Python 3.12+
then skips), so it is what `.mcp.json` uses to launch the tool server.
"""

from .cli.app import app

if __name__ == "__main__":
    app(prog_name="llmve")
