import sys
from pathlib import Path

from streamlit.web import bootstrap


def resolve_app_path() -> str:
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        base_dir = Path(sys._MEIPASS)
    else:
        base_dir = Path(__file__).resolve().parent

    app_path = base_dir / "streamlit_app.py"
    if not app_path.exists():
        raise FileNotFoundError(f"Could not find Streamlit app: {app_path}")
    return str(app_path)


if __name__ == "__main__":
    app_path = resolve_app_path()
    bootstrap.run(
        app_path,
        "",
        [],
        {
            "server.headless": False,
            "server.port": 8501,
        },
    )
