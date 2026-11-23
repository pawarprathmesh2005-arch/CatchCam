"""Pack the CatchCam code (not your videos, results or virtual environment) into catchcam_project.zip for Google Colab.
Run it in the project folder:   python make_colab_bundle.py"""
import zipfile
from pathlib import Path

SKIP_DIRS = {"venv", ".venv", "env", "videos", "outputs", "embeddings", "matches", "relationships", ".thumbs", "__pycache__", ".git",
             "catchcam_new_features", "demo_data", "runs", ".pytest_cache", "node_modules"}
SKIP_EXT = {".zip", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".pt", ".pth", ".onnx", ".pyc", ".docx", ".pdf"}


def main(root=".", out="catchcam_project.zip"):
    root, out = Path(root).resolve(), Path(out).resolve()
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(root.rglob("*")):
            rel = f.relative_to(root)
            if f.is_file() and f != out and not (set(rel.parts[:-1]) & SKIP_DIRS) and f.suffix.lower() not in SKIP_EXT:
                z.write(f, rel)
                n += 1
    print(f"{out.name}: {n} files, {out.stat().st_size / 1e6:.1f} MB. Upload it to Google Drive (folder CatchCamColab).")
    return n


if __name__ == "__main__":
    main()
