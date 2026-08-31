"""Offline face-swap engine.

Every model is loaded from ./models — nothing is fetched at runtime. Run
download_models.py once on a connected machine, then this works air-gapped.
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import threading
import warnings
from dataclasses import dataclass

# Set before insightface/onnxruntime import so no library tries to phone home.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# albumentations (an insightface dependency) makes a blocking HTTPS call on import
# to check for a newer release. Offline that is a multi-second stall on every start.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import cv2
import numpy as np

# Import onnxruntime eagerly, and before anything imports PyQt5. On Windows,
# loading Qt first leaves onnxruntime's pybind11 extension unable to initialise
# ("DLL load failed ... initialization routine failed"). Verified locally:
# onnxruntime->PyQt5 works, PyQt5->onnxruntime does not. ui.py therefore imports
# this module above its PyQt5 imports, and this line is what makes that work.
try:
    import onnxruntime as _ort  # noqa: E402  (must precede Qt)
except ImportError as _exc:  # pragma: no cover - environment-specific
    if "PyQt5" in sys.modules or "PyQt6" in sys.modules:
        raise ImportError(
            "onnxruntime failed to load because Qt was imported first. Import "
            "swapper before PyQt5 (see the note at the top of ui.py)."
        ) from _exc
    raise

if getattr(sys, "frozen", False):
    # Under PyInstaller __file__ points inside the temporary _MEIPASS extraction
    # directory, so models/ must be resolved next to the .exe the user launched.
    HERE = os.path.dirname(os.path.abspath(sys.executable))
else:
    HERE = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(HERE, "models")
# insightface resolves packs as <root>/models/<name>, hence the nested path.
IF_ROOT = os.path.join(MODELS_DIR, "insightface")
PACK_DIR = os.path.join(IF_ROOT, "models", "buffalo_l")
SWAP_MODEL = os.path.join(MODELS_DIR, "inswapper_128.onnx")

# inswapper_128 was trained against buffalo_l's w600k_r50 recogniser and carries a
# baked-in 'emap' matrix fitted to that exact embedding space. Do NOT substitute
# antelopev2 or any other ArcFace pack -- the embeddings are not interchangeable
# and the swap degrades into a blurry average face.
REQUIRED = ["det_10g.onnx", "w600k_r50.onnx", "2d106det.onnx"]

_analyser = None
_swapper = None
# Model construction is one-time but the GUI can call into it from more
# than one worker thread at once; without this both would build their own
# copy of a 174 MB model and race on the globals.
_load_lock = threading.Lock()


def missing_models() -> list[str]:
    gone = [f for f in REQUIRED if not os.path.exists(os.path.join(PACK_DIR, f))]
    if not os.path.exists(SWAP_MODEL):
        gone.append("inswapper_128.onnx")
    return gone


def providers() -> list[str]:
    have = _ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in have]


# insightface calls a deprecated skimage API on every alignment. Filtering it
# once here is thread-safe; warnings.catch_warnings() around each call is not.
warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", message=r".*\bestimate\b.*deprecated.*")


@contextlib.contextmanager
def _quiet():
    """Swallow insightface's dozen 'Applied providers' lines per model.

    This swaps the process-wide sys.stdout, so it is only ever entered while
    holding _load_lock -- two threads redirecting at once restore each other's
    streams on exit and the output leaks anyway.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        yield


def load_analyser(det_size: int = 640):
    """Detector + landmarks + ArcFace. Independent of the swap model, so face
    detection still works before inswapper_128.onnx has been fetched."""
    global _analyser
    if _analyser is not None:
        return _analyser
    gone = [f for f in REQUIRED if not os.path.exists(os.path.join(PACK_DIR, f))]
    if gone:
        raise FileNotFoundError(
            "Missing buffalo_l file(s): " + ", ".join(gone)
            + "\nRun:  python download_models.py")

    from insightface.app import FaceAnalysis

    with _load_lock:
        if _analyser is not None:
            return _analyser
        with _quiet():
            _analyser = FaceAnalysis(
                name="buffalo_l",
                root=IF_ROOT,
                # genderage and the 3D 1k3d68 model are dead weight here;
                # skipping them saves ~150 MB of RAM and a second of load time.
                allowed_modules=["detection", "recognition", "landmark_2d_106"],
                providers=providers(),
            )
            # det_size drives detector input resolution: 640 finds small faces
            # in group photos, 320 is faster when the face fills the frame.
            _analyser.prepare(ctx_id=0, det_size=(det_size, det_size))
    return _analyser


def load_swapper():
    global _swapper
    if _swapper is not None:
        return _swapper
    if not os.path.exists(SWAP_MODEL):
        raise FileNotFoundError(
            f"Missing {os.path.basename(SWAP_MODEL)}"
            "\nRun:  python download_models.py")

    from insightface.model_zoo import get_model

    with _load_lock:
        if _swapper is not None:
            return _swapper
        with _quiet():
            _swapper = get_model(SWAP_MODEL, providers=providers())
    return _swapper


def load(det_size: int = 640):
    return load_analyser(det_size), load_swapper()


@dataclass
class DetectedFace:
    index: int
    bbox: tuple[int, int, int, int]
    raw: object


def detect(img_bgr: np.ndarray) -> list[DetectedFace]:
    """All faces, ordered left-to-right so indices stay stable between calls."""
    analyser = load_analyser()
    faces = sorted(analyser.get(img_bgr), key=lambda f: f.bbox[0])
    return [DetectedFace(i, tuple(int(v) for v in f.bbox), f) for i, f in enumerate(faces)]


def face_at_point(faces, x: int, y: int):
    """Face under (x, y); on overlap prefer the smallest box (the frontmost person)."""
    hits = [f for f in faces if f.bbox[0] <= x <= f.bbox[2] and f.bbox[1] <= y <= f.bbox[3]]
    if hits:
        return min(hits, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return None


def _face_mask(shape, face: DetectedFace, feather: float = 0.10) -> np.ndarray:
    """Soft float mask over the facial skin area, from the 106-point contour."""
    mask = np.zeros(shape[:2], np.uint8)
    pts = getattr(face.raw, "landmark_2d_106", None)
    x1, y1, x2, y2 = face.bbox
    if pts is None:
        cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2),
                    ((x2 - x1) // 2, (y2 - y1) // 2), 0, 0, 360, 255, -1)
    else:
        cv2.fillConvexPoly(mask, cv2.convexHull(pts.astype(np.int32)), 255)
    # Feather proportional to face size, so it looks the same at any resolution.
    k = int(max(3, (x2 - x1) * feather)) | 1
    return cv2.GaussianBlur(mask, (k, k), 0).astype(np.float32) / 255.0


def _match_tone(swapped: np.ndarray, original: np.ndarray,
                mask: np.ndarray, strength: float) -> np.ndarray:
    """Pull the swapped face's colour statistics back toward the original face.

    inswapper's 128px decoder drifts warm/light, which reads as a wrong complexion
    on Asian skin tones and leaves a visible seam against the neck. Matching mean
    and standard deviation per LAB channel, inside the face mask only, restores the
    photo's own lighting without touching identity (shape/features live in the
    high-frequency detail these first-order statistics ignore).
    """
    if strength <= 0:
        return swapped
    sel = mask > 0.5
    if sel.sum() < 50:  # too small to estimate statistics reliably
        return swapped

    src = cv2.cvtColor(swapped, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(original, cv2.COLOR_BGR2LAB).astype(np.float32)
    out = src.copy()
    for c in range(3):
        s_mu, s_sd = src[..., c][sel].mean(), src[..., c][sel].std()
        r_mu, r_sd = ref[..., c][sel].mean(), ref[..., c][sel].std()
        if s_sd < 1e-4:
            continue
        out[..., c] = (src[..., c] - s_mu) * (r_sd / s_sd) + r_mu

    corrected = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    a = (mask * float(strength))[..., None]
    return (corrected * a + swapped * (1 - a)).astype(np.uint8)


def draw_overlay(bgr, faces, selected=-1):
    """Corner brackets on every face; the selected one is thicker and accented."""
    out = bgr.copy()
    s = max(1, int(round(max(out.shape[:2]) / 900)))
    for f in faces:
        x1, y1, x2, y2 = f.bbox
        hit = f.index == selected
        col = (255, 141, 76) if hit else (150, 150, 150)  # BGR
        t = 3 * s if hit else 2 * s
        arm = max(8, (x2 - x1) // 5)
        for cx, cy, dx, dy in (
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1),
        ):
            cv2.line(out, (cx, cy), (cx + dx * arm, cy), col, t)
            cv2.line(out, (cx, cy), (cx, cy + dy * arm), col, t)
        cv2.putText(
            out, str(f.index + 1), (x1, max(14 * s, y1 - 8 * s)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7 * s, col, 2 * s, cv2.LINE_AA,
        )
    return out


def imread(path: str) -> "np.ndarray | None":
    """cv2.imread replacement. cv2 uses the ANSI API on Windows and returns None
    for any path with non-ASCII characters, so decode from bytes ourselves."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite(path: str, img: "np.ndarray") -> bool:
    """cv2.imwrite replacement, Unicode-path safe."""
    ext = os.path.splitext(path)[1] or ".png"
    params = [cv2.IMWRITE_JPEG_QUALITY, 96] if ext.lower() in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        return False
    buf.tofile(path)
    return True


def pick_source_face(source_bgr: np.ndarray) -> DetectedFace:
    """Largest face in the source image -- the subject, not a bystander behind them."""
    faces = detect(source_bgr)
    if not faces:
        raise ValueError("No face found in the replacement photo.")
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def _bounds(flags: np.ndarray, pad: int = 8):
    """Tight (y0, y1, x0, x1) box around the True pixels, or None if there are none."""
    rows, cols = np.any(flags, axis=1), np.any(flags, axis=0)
    if not rows.any():
        return None
    h, w = flags.shape
    y0, y1 = int(np.argmax(rows)), h - int(np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), w - int(np.argmax(cols[::-1]))
    return (max(0, y0 - pad), min(h, y1 + pad),
            max(0, x0 - pad), min(w, x1 + pad))


def swap_identity(target_bgr, target_face, source_bgr):
    """Expensive stage: run the network once. Returns (raw swap, mask, region).

    Split from the cosmetic stage so the UI can move the tone/blend sliders
    without paying for another forward pass.
    """
    model = load_swapper()
    src_face = pick_source_face(source_bgr)
    raw = model.get(target_bgr, target_face.raw, src_face.raw, paste_back=True)
    mask = _face_mask(target_bgr.shape, target_face)

    # The region finish() has to recomposite is everything the network actually
    # touched, which is wider than the landmark hull: paste_back writes the whole
    # aligned crop, taking in forehead and chin. Computing it from the real diff
    # (once, here) rather than guessing a margin keeps blend=0 an exact restore.
    region = _bounds(np.any(raw != target_bgr, axis=2) | (mask > 0))
    return raw, mask, region


def finish(raw, target_bgr, mask, region=None, tone_match: float = 0.7,
           blend: float = 1.0):
    """Cheap stage: colour-match and blend, over the touched region only.

    Working on the full frame made this O(image): ~1.4 s on a 22 MP photo, which
    froze the GUI on every slider tick. Restricted to the face region it is flat
    in image size, so the sliders stay live no matter how big the photo is.
    """
    if region is None:
        region = _bounds(np.any(raw != target_bgr, axis=2) | (mask > 0))
    if region is None:
        return raw.copy()

    y0, y1, x0, x1 = region
    out = raw.copy()
    sub_raw, sub_tgt = raw[y0:y1, x0:x1], target_bgr[y0:y1, x0:x1]
    sub_mask = mask[y0:y1, x0:x1]

    sub = _match_tone(sub_raw, sub_tgt, sub_mask, tone_match)
    if blend < 1.0:
        # Fade the whole touched region, not just the mask, or paste_back's
        # wider edits would survive at blend=0.
        a = float(blend)
        sub = (sub * a + sub_tgt * (1 - a)).astype(np.uint8)
    out[y0:y1, x0:x1] = sub
    return out


def swap(target_bgr, target_face, source_bgr, tone_match: float = 0.7, blend: float = 1.0):
    """Put the source person's identity onto target_face.

    tone_match: 0 keeps inswapper's raw colour, 1 fully adopts the original face's
                lighting and complexion. ~0.7 is a good default.
    blend:      0 is the untouched photo, 1 the full swap.
    """
    raw, mask, region = swap_identity(target_bgr, target_face, source_bgr)
    return finish(raw, target_bgr, mask, region, tone_match, blend)
