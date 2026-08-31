"""Person segmentation: cut the people out and put a new background behind them.

Same rules as swapper.py -- everything loads from ./models, nothing is fetched
at runtime, and onnxruntime is imported before Qt.
"""
from __future__ import annotations

import os
import threading

import cv2
import numpy as np

# Importing swapper first is deliberate and load-bearing: it sets the offline env
# vars and imports onnxruntime before anything can pull in Qt. Keep this import
# even though only these three names are used.
from swapper import MODELS_DIR, _quiet, providers

# The two matting models the app offers. Plain u2net was tried and dropped: it
# is a *salient object* detector, so on a photo of three people it masked the
# single most prominent one and returned zero for the other two.
MATTE_MODELS = {
    "human": "u2net_human_seg.onnx",        # 176 MB, Apache-2.0, 320px
    "birefnet_lite": "birefnet_lite.onnx",  # 224 MB, MIT, 1024px, hair detail
}
DEFAULT_MATTE = "human"

# U2Net's published preprocessing (rembg uses exactly this).
_U2NET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_U2NET_STD = np.array([0.229, 0.224, 0.225], np.float32)

_sessions: dict[str, object] = {}
_lock = threading.Lock()


def model_path(filename: str) -> str:
    return os.path.join(MODELS_DIR, filename)


def available() -> dict[str, bool]:
    """Which optional models are present. Nothing here is required for face swap."""
    return {k: os.path.exists(model_path(v)) for k, v in MATTE_MODELS.items()}


def _session(filename: str):
    """Cached InferenceSession, built at most once even under concurrent calls."""
    path = model_path(filename)
    if filename in _sessions:
        return _sessions[filename]
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {filename}\nRun:  python download_models.py --segmentation")
    import onnxruntime as ort

    with _lock:
        if filename not in _sessions:
            with _quiet():
                so = ort.SessionOptions()
                so.log_severity_level = 3
                _sessions[filename] = ort.InferenceSession(
                    path, sess_options=so, providers=providers())
    return _sessions[filename]


def _static_hw(sess, default=(320, 320)) -> tuple[int, int]:
    """Input H,W the model wants. Dynamic axes come back as strings or None, in
    which case the caller's default is used rather than guessing."""
    shape = sess.get_inputs()[0].shape
    if len(shape) == 4:
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int):
            return h, w
    return default


def _letterbox(rgb, ih, iw):
    """Resize into ih x iw keeping the aspect ratio, padding the rest.

    Returns (canvas, (top, left, nh, nw)) so the model output can be cropped
    back out of the padding.
    """
    h, w = rgb.shape[:2]
    s = min(ih / h, iw / w)
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    canvas = np.zeros((ih, iw, 3), rgb.dtype)
    top, left = (ih - nh) // 2, (iw - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(
        rgb, (nw, nh), interpolation=cv2.INTER_AREA)
    return canvas, (top, left, nh, nw)


def _unletterbox(a, box, out_h, out_w):
    top, left, nh, nw = box
    return cv2.resize(a[top:top + nh, left:left + nw], (out_w, out_h),
                      interpolation=cv2.INTER_LINEAR)


# ------------------------------------------------------------------- matting

def matte(img_bgr: np.ndarray, model: str = DEFAULT_MATTE) -> np.ndarray:
    """Soft foreground alpha for the whole image, float32 HxW in [0, 1].

    Covers every person in the frame as one combined mask; call person_masks()
    to split it per person.
    """
    if model not in MATTE_MODELS:
        raise ValueError(f"Unknown matte model {model!r}; have {list(MATTE_MODELS)}")
    sess = _session(MATTE_MODELS[model])
    ih, iw = _static_hw(sess, (1024, 1024) if model.startswith("birefnet")
                        else (320, 320))

    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Letterbox, never squash. rembg resizes straight to 320x320 and ignores the
    # aspect ratio; on a wide group photo that flattens the people at the edges
    # until the model stops seeing them. Measured on a 2032x900 test frame: a
    # squashed resize scored 0% coverage on the left-hand person, letterboxing
    # the same frame scored 93%.
    small, box = _letterbox(rgb, ih, iw)

    # The two models want different normalisation. Feeding one the other's input
    # produces a plausible-looking but badly degraded mask rather than an error,
    # so the branch matters.
    if model == "human":
        # rembg divides by the image max rather than 255; matching it keeps the
        # published quality on dark or low-contrast photos.
        x = small.astype(np.float32)
        x = (x / (float(x.max()) or 1.0) - _U2NET_MEAN) / _U2NET_STD
    else:
        x = (small.astype(np.float32) / 255.0 - _U2NET_MEAN) / _U2NET_STD
    x = np.transpose(x, (2, 0, 1))[None].astype(np.float32)

    out = sess.run(None, {sess.get_inputs()[0].name: x})[0]
    a = np.squeeze(out)
    if a.ndim == 3:            # some exports return a few channels; take the first
        a = a[0]
    # BiRefNet exports return raw logits. Min-max scaling those would stretch
    # whatever range this particular image happened to produce, so a photo with
    # no confident foreground would still come back with a full-contrast mask.
    if a.min() < 0.0 or a.max() > 1.0:
        a = 1.0 / (1.0 + np.exp(-np.clip(a, -30, 30)))
    else:
        lo, hi = float(a.min()), float(a.max())
        a = (a - lo) / (hi - lo) if hi > lo else np.zeros_like(a)
    return _unletterbox(a.astype(np.float32), box,
                        img_bgr.shape[0], img_bgr.shape[1])


def refine_edges(alpha: np.ndarray, radius: int = 3) -> np.ndarray:
    """Feather the alpha slightly so composites do not show a cut-out edge."""
    k = max(1, int(radius)) | 1
    return cv2.GaussianBlur(alpha, (k, k), 0)


def harden(alpha: np.ndarray, lo: float = 0.35, hi: float = 0.85,
           core: int = 15) -> np.ndarray:
    """Force confident pixels to fully opaque or fully clear.

    The matting net rarely returns a clean 1.0 across a person: on a patterned
    dress it drifts to 0.5-0.9, and at those values a replaced background shows
    straight through the clothing. Measured on a real family photo, 5.65% of
    pixels well inside the people sat below 0.95.

    Two steps: a linear remap that saturates everything above `hi`, then an
    erode that pins the interior to exactly 1.0 so stubborn patches cannot leak.
    Only a narrow band around the silhouette keeps intermediate values, which is
    where partial coverage is real (hair, motion blur).
    """
    out = np.clip((alpha - lo) / max(1e-6, hi - lo), 0.0, 1.0).astype(np.float32)
    out = fill_holes(out)
    if core > 0:
        k = np.ones((int(core) | 1,) * 2, np.uint8)
        interior = cv2.erode((out > 0.5).astype(np.uint8), k) > 0
        out[interior] = 1.0
    return out


def fill_holes(alpha: np.ndarray, max_frac: float = 0.02) -> np.ndarray:
    """Close small gaps fully enclosed by the silhouette.

    Where the matting net is unsure -- a light jacket, a highlight on a cheek --
    it can return near-zero alpha in the middle of a person, and the replaced
    background then shows through as a bright blotch on their face. Those regions
    are enclosed by foreground, so flood-filling from the frame border finds them.

    Only small holes are filled. A genuine gap, like the triangle between an arm
    and the body, is large and must stay open or the background would be painted
    over with skin.
    """
    solid = (alpha > 0.5).astype(np.uint8)
    if not solid.any():
        return alpha
    # Flood the true background inward from a border that is guaranteed empty.
    padded = cv2.copyMakeBorder(solid, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    ff = padded.copy()
    cv2.floodFill(ff, np.zeros((ff.shape[0] + 2, ff.shape[1] + 2), np.uint8), (0, 0), 1)
    enclosed = (ff[1:-1, 1:-1] == 0)
    if not enclosed.any():
        return alpha

    limit = max_frac * float(solid.sum())
    out = alpha.copy()
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        enclosed.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] <= limit:
            out[labels == i] = 1.0
    return out


def decontaminate(img_bgr: np.ndarray, alpha: np.ndarray,
                  radius: int = 25) -> np.ndarray:
    """Remove the old background's colour from partly-transparent pixels.

    A pixel at the edge of a person is a mix: observed = a*F + (1-a)*B, where B
    is the background that was there. Composite it over a new background and B's
    colour rides along -- against green foliage the people end up fringed green
    (measured: +7.1 green excess in the edge band). Estimating B locally and
    solving for F removes it.

    B is estimated by blurring the background-only pixels and normalising by the
    blurred weights, which spreads nearby background colour underneath the edge.
    """
    a = alpha[..., None].astype(np.float32)
    img = img_bgr.astype(np.float32)
    k = max(3, int(radius)) | 1

    # Blur the 2-D weight separately: GaussianBlur drops a trailing length-1
    # axis, so passing (H, W, 1) here returns (H, W) and will not broadcast
    # against the 3-channel numerator.
    w2 = (1.0 - alpha).astype(np.float32)
    num = cv2.GaussianBlur(img * w2[..., None], (k, k), 0)
    den = cv2.GaussianBlur(w2, (k, k), 0)[..., None]
    bg_est = num / np.maximum(den, 1e-3)

    fg = (img - (1.0 - a) * bg_est) / np.maximum(a, 1e-3)
    fg = np.clip(fg, 0, 255)
    # Only trust this where the pixel really is a mix; leave solid pixels alone.
    band = ((alpha > 0.02) & (alpha < 0.98))[..., None]
    return np.where(band, fg, img).astype(np.float32)


# ------------------------------------------------- per-person instance masks

def person_box(face_bbox, shape, up=1.2, down=7.0, side=3.0):
    """Generous crop around one person, estimated from their face box.

    Body proportions: a standing adult is roughly 7-8 head-heights tall and a
    little under 3 head-widths across at the shoulders, so these multiples cover
    a full-length figure and simply clamp to the frame for a head-and-shoulders
    shot.
    """
    x1, y1, x2, y2 = face_bbox
    h, w = shape[:2]
    fw, fh = x2 - x1, y2 - y1
    cx = (x1 + x2) / 2.0
    return (max(0, int(cx - side * fw)), max(0, int(y1 - up * fh)),
            min(w, int(cx + side * fw)), min(h, int(y2 + down * fh)))


def person_masks(img_bgr, faces, model: str = DEFAULT_MATTE) -> list[np.ndarray]:
    """One exclusive alpha mask per detected face, in the same order as `faces`.

    Runs the matte once per person on a crop around them, rather than once on the
    whole frame. A single 320x320 pass over a wide group photo spends almost no
    resolution on anyone at the edges: on a 2032x900 test frame the rightmost
    person came back at 0% coverage even with letterboxing. Cropping first gives
    every person the model's full input resolution.

    Costs one model run per face (~0.7 s each on CPU, far less on GPU), which is
    the right trade for a handful of people in a family photo.
    """
    if not faces:
        return []

    stack = np.zeros((len(faces),) + img_bgr.shape[:2], np.float32)
    for i, f in enumerate(faces):
        x0, y0, x1, y1 = person_box(f.bbox, img_bgr.shape)
        crop = img_bgr[y0:y1, x0:x1]
        if crop.size == 0:
            continue
        stack[i, y0:y1, x0:x1] = matte(crop, model)

    # Crops overlap, and each crop's matte covers the neighbours standing in it
    # too, so comparing raw alpha picks an arbitrary winner. On a tight group
    # that misassigned every mask -- one person's mask covered just 2.3% of their
    # own face, and all of them sat 75-114px sideways of where they belonged.
    #
    # Weight each person's alpha by how plausibly a pixel belongs to *them*
    # before choosing. People in a group photo stand side by side, so horizontal
    # distance from their own face carries most of the signal, with a looser
    # vertical term so a person keeps their legs but not their neighbour's.
    h, w = img_bgr.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    scored = np.zeros_like(stack)
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = f.bbox
        fw, fh = max(1, x2 - x1), max(1, y2 - y1)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        prior = np.exp(-0.5 * (((xs - cx) / (1.6 * fw)) ** 2
                               + ((ys - cy) / (5.0 * fh)) ** 2))
        scored[i] = stack[i] * prior

    winner = scored.argmax(axis=0)
    strongest = stack.max(axis=0)
    out = []
    for i in range(len(faces)):
        m = np.where((winner == i) & (strongest > 0.05), stack[i], 0.0)
        out.append(refine_edges(m.astype(np.float32)))
    return out


# ---------------------------------------------------------------- compositing

def composite(fg_bgr: np.ndarray, bg_bgr: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """alpha-blend fg over bg. All three must be the same size."""
    a = alpha[..., None].astype(np.float32)
    return (fg_bgr * a + bg_bgr * (1.0 - a)).astype(np.uint8)


def foreground_alpha(img_bgr, faces=None, model: str = DEFAULT_MATTE) -> np.ndarray:
    """Combined alpha over every person.

    Given the detected faces this is the union of the per-person crops, which
    catches people a single whole-frame pass misses entirely. Without faces it
    falls back to one pass over the frame.
    """
    if not faces:
        return matte(img_bgr, model)
    masks = person_masks(img_bgr, faces, model)
    return np.clip(np.max(np.stack(masks), axis=0), 0.0, 1.0)


def fit_cover(bg_bgr: np.ndarray, w: int, h: int) -> np.ndarray:
    """Scale a background to cover w x h and centre-crop it.

    Cover, not stretch: a portrait backdrop behind a landscape photo would
    otherwise be squeezed and every vertical line in it would lean.
    """
    bh, bw = bg_bgr.shape[:2]
    s = max(w / bw, h / bh)
    scaled = cv2.resize(bg_bgr, (max(w, int(round(bw * s))), max(h, int(round(bh * s)))),
                        interpolation=cv2.INTER_AREA)
    top, left = (scaled.shape[0] - h) // 2, (scaled.shape[1] - w) // 2
    return scaled[top:top + h, left:left + w]


def compose_over(img_bgr, alpha, new_bg_bgr, feather: int = 3,
                 clean: bool = True) -> np.ndarray:
    """Put a matted foreground over a new background.

    Hardening and decontamination both happen here rather than in the matte, so
    the raw alpha stays reusable and the UI can recompose from cache.
    """
    h, w = img_bgr.shape[:2]
    # Decontaminate against the RAW alpha, then harden. Doing it the other way
    # round is self-defeating: hardening drives edge pixels to 1.0 first, so the
    # partially-mixed pixels that carry the old background's colour are no
    # longer in the band decontaminate() will touch, and the fringe survives.
    fg = decontaminate(img_bgr, alpha) if clean else img_bgr.astype(np.float32)
    a = refine_edges(harden(alpha), feather)
    bg = fit_cover(new_bg_bgr, w, h).astype(np.float32)
    a3 = a[..., None]
    return np.clip(fg * a3 + bg * (1.0 - a3), 0, 255).astype(np.uint8)


def replace_background(img_bgr: np.ndarray, new_bg_bgr: np.ndarray, faces=None,
                       model: str = DEFAULT_MATTE, feather: int = 3) -> np.ndarray:
    """Keep the people, swap everything behind them."""
    return compose_over(img_bgr, foreground_alpha(img_bgr, faces, model),
                        new_bg_bgr, feather)


