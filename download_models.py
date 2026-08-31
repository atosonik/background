"""One-time model fetch. Run once online; afterwards the app is fully offline.

Also works as a bundler: copy the whole models/ folder to an air-gapped machine.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")
PACK = os.path.join(MODELS, "insightface", "models", "buffalo_l")
SWAP = os.path.join(MODELS, "inswapper_128.onnx")

BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
BUFFALO_NEED = ["det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx"]

# InsightFace withdrew the official release in 2023; these are community mirrors.
# They are NOT guaranteed byte-identical, so a partial download is only ever
# resumed against the mirror it was started from -- splicing bytes from two
# mirrors produces a file that passes a size check and then fails to parse.
SWAP_MIRRORS = [
    "https://huggingface.co/hacksider/inswapper_128/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx",
    "https://huggingface.co/ModelsLab/inswapper/resolve/main/inswapper_128.onnx",
    "https://huggingface.co/Patil/inswapper/resolve/main/inswapper_128.onnx",
]
MIN_SWAP_BYTES = 500_000_000
RESUMES_PER_MIRROR = 8


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def content_length(url: str) -> int:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return int(r.headers.get("Content-Length", 0))
    except Exception:
        return 0


def curl_resume(url: str, dest: str) -> int:
    """One resumed curl attempt. Returns curl's exit code (0 == finished)."""
    return subprocess.call([
        "curl", "-sL", "--fail", "-C", "-", "--max-time", "900",
        # Give up on a stalled connection rather than hanging for the full 900s.
        "--speed-limit", "5000", "--speed-time", "120",
        "-o", dest, url,
    ])


def urllib_resume(url: str, dest: str) -> int:
    """Same contract as curl_resume, for machines without curl on PATH.

    Windows 10 1803+ ships curl.exe, but a stripped PATH or an older box does
    not -- and without this the whole downloader died on a bare FileNotFoundError.
    """
    have = os.path.getsize(dest) if os.path.exists(dest) else 0
    req = urllib.request.Request(url, headers={"User-Agent": "faceswap/1.0"})
    if have:
        req.add_header("Range", f"bytes={have}-")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            # 200 to a Range request means the server ignored it; restart cleanly
            # rather than appending a second copy of the file to the first.
            mode = "ab" if (have and r.status == 206) else "wb"
            with open(dest, mode) as fh:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    fh.write(block)
        return 0
    except Exception as exc:
        print(f"  urllib: {exc}")
        return 1


def fetch_resume(url: str, dest: str) -> int:
    if shutil.which("curl"):
        return curl_resume(url, dest)
    return urllib_resume(url, dest)


def verify_onnx(path: str) -> bool:
    """A size check is not enough -- a spliced or truncated file can still be the
    right length. Actually parse it."""
    try:
        import onnx

        onnx.load(path)
        return True
    except Exception as exc:
        print(f"  model does not parse: {exc}")
        return False


def ensure_buffalo() -> None:
    if all(os.path.exists(os.path.join(PACK, f)) for f in BUFFALO_NEED):
        print("buffalo_l: present")
        return

    # Reuse the pack insightface may already have cached, rather than re-fetching.
    cached = os.path.expanduser("~/.insightface/models/buffalo_l")
    if all(os.path.exists(os.path.join(cached, f)) for f in BUFFALO_NEED):
        print(f"buffalo_l: copying from {cached}")
        os.makedirs(PACK, exist_ok=True)
        for f in os.listdir(cached):
            shutil.copy2(os.path.join(cached, f), os.path.join(PACK, f))
        return

    print("buffalo_l: downloading (~289 MB)")
    os.makedirs(PACK, exist_ok=True)
    zp = os.path.join(MODELS, "buffalo_l.zip")
    urllib.request.urlretrieve(BUFFALO_URL, zp)
    with zipfile.ZipFile(zp) as z:
        z.extractall(PACK)
    os.remove(zp)


def ensure_swapper() -> None:
    if os.path.exists(SWAP) and os.path.getsize(SWAP) > MIN_SWAP_BYTES:
        if verify_onnx(SWAP):
            print(f"inswapper_128: present  sha256={sha256(SWAP)}")
            return
        print("inswapper_128: existing file is corrupt, refetching")
        os.remove(SWAP)

    os.makedirs(MODELS, exist_ok=True)
    for url in SWAP_MIRRORS:
        host = url.split("/")[3]
        want = content_length(url)
        print(f"inswapper_128: {host}" + (f" ({want / 1e6:.0f} MB)" if want else ""))
        # A fresh mirror means a fresh file -- never resume onto another's bytes.
        if os.path.exists(SWAP):
            os.remove(SWAP)

        for attempt in range(1, RESUMES_PER_MIRROR + 1):
            rc = fetch_resume(url, SWAP)
            have = os.path.getsize(SWAP) if os.path.exists(SWAP) else 0
            print(f"  attempt {attempt}: rc={rc} {have / 1e6:.0f} MB")
            if rc == 0:
                break
        else:
            print("  giving up on this mirror")
            continue

        if os.path.getsize(SWAP) < MIN_SWAP_BYTES or not verify_onnx(SWAP):
            print("  incomplete or unparseable, trying next mirror")
            continue
        print(f"  ok  sha256={sha256(SWAP)}")
        return

    raise SystemExit(
        "All mirrors failed. Download inswapper_128.onnx (~554 MB) by hand and "
        f"place it at {SWAP}"
    )



SEGMENTATION = [
    ("u2net_human_seg.onnx", 170_000_000,
     "https://huggingface.co/tomjackson2023/rembg/resolve/main/u2net_human_seg.onnx"),
    ("birefnet_lite.onnx", 215_000_000,
     "https://huggingface.co/onnx-community/BiRefNet_lite-ONNX/resolve/main/onnx/model.onnx"),
]


def ensure_segmentation() -> None:
    """Matting models for background replacement (~400 MB, Apache-2.0 / MIT)."""
    for name, floor, url in SEGMENTATION:
        dest = os.path.join(MODELS, name)
        if os.path.exists(dest) and os.path.getsize(dest) >= floor:
            print(f"{name}: present")
            continue
        print(f"{name}: downloading")
        for attempt in range(1, RESUMES_PER_MIRROR + 1):
            if fetch_resume(url, dest) == 0 and os.path.getsize(dest) >= floor:
                break
            have = os.path.getsize(dest) if os.path.exists(dest) else 0
            print(f"  attempt {attempt}: {have / 1e6:.0f} MB")
        else:
            raise SystemExit(f"could not fetch {name} from {url}")
        if not verify_onnx(dest):
            raise SystemExit(f"{name} downloaded but does not parse")
        print(f"  ok  {os.path.getsize(dest) / 1e6:.0f} MB")


def main() -> None:
    ensure_buffalo()
    ensure_swapper()
    ensure_segmentation()
    sys.path.insert(0, HERE)
    import swapper

    gone = swapper.missing_models()
    total = sum(
        os.path.getsize(os.path.join(r, f))
        for r, _, fs in os.walk(MODELS) for f in fs
    )
    print("\nAll models present." if not gone else f"\nSTILL MISSING: {gone}")
    print(f"models/ is {total / 1e6:.0f} MB -- copy this folder to go offline.")


if __name__ == "__main__":
    main()
