"""Command-line entry points.

``llmve`` (``cli.app``) is the stage-based CLI. ``main`` is the legacy
single-command router, imported lazily so the core install does not need the
``legacy`` extra.
"""


def main() -> None:
    from .main import main as legacy_main

    legacy_main()


__all__ = ["main"]
