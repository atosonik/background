"""Verify the install end to end. Run after download_models.py.

    python selftest.py                  # uses testdata/
    python selftest.py photo.jpg face.jpg
"""
import os
import sys
import time

import swapper  # noqa: F401  (also proves the onnxruntime import order is intact)

import numpy as np

# swapper.HERE is exe-relative when frozen, so the bundled app looks for
# testdata/ and outputs/ next to FaceSwap.exe rather than inside _MEIPASS.
HERE = swapper.HERE
FAILED = []


def check(label, ok, detail=""):
    print(f"  [{'ok' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAILED.append(label)
    return ok


def main(argv=None) -> int:
    FAILED.clear()
    print("models")
    gone = swapper.missing_models()
    if not check("all model files present", not gone, ", ".join(gone)):
        print("\nRun: python download_models.py")
        return 1

    size = os.path.getsize(swapper.SWAP_MODEL)
    check("inswapper_128.onnx size", size == 554_253_681, f"{size} bytes")

    print("\nruntime")
    prov = swapper.providers()
    check("onnxruntime provider", bool(prov), prov[0] if prov else "none")

    t = time.time()
    swapper.load_analyser()
    check("detector + arcface load", True, f"{time.time() - t:.1f}s")
    t = time.time()
    swapper.load_swapper()
    check("swap model load", True, f"{time.time() - t:.1f}s")

    args = sys.argv[1:] if argv is None else list(argv)
    tgt_p = args[0] if args else os.path.join(HERE, "testdata", "t1.jpg")
    src_p = args[1] if len(args) > 1 else os.path.join(HERE, "testdata", "t0.jpg")
    if not (os.path.exists(tgt_p) and os.path.exists(src_p)):
        print(f"\nNo test images at {tgt_p} / {src_p} - pass two paths instead.")
        return 1 if FAILED else 0

    print("\npipeline")
    tgt = swapper.imread(tgt_p)
    src = swapper.imread(src_p)
    check("unicode-safe imread", tgt is not None and src is not None)

    faces = swapper.detect(tgt)
    if not check("faces detected in target", bool(faces), f"{len(faces)} found"):
        return 1
    check("106-pt landmarks", getattr(faces[0].raw, "landmark_2d_106", None) is not None)
    check("512-d unit embedding",
          abs(float(np.linalg.norm(faces[0].raw.normed_embedding)) - 1.0) < 1e-3)

    t = time.time()
    raw, mask, region = swapper.swap_identity(tgt, faces[0], src)
    dt = time.time() - t
    check("swap ran", raw.shape == tgt.shape, f"{dt:.2f}s")

    sel = mask > 0.5
    delta = np.abs(raw.astype(int) - tgt.astype(int))
    check("face region changed", delta[sel].mean() > 5, f"mean delta {delta[sel].mean():.1f}")

    # paste_back writes the whole aligned crop, which is deliberately wider than
    # the detection box (it takes in forehead and chin), so "outside the landmark
    # hull" is the wrong reference. The invariant that actually matters is that
    # edits stay local: nothing outside a generous margin round the face may move.
    x1, y1, x2, y2 = faces[0].bbox
    mx, my = int((x2 - x1) * 0.6), int((y2 - y1) * 0.6)
    far = np.ones(tgt.shape[:2], bool)
    far[max(0, y1 - my):y2 + my, max(0, x1 - mx):x2 + mx] = False
    touched_far = int((delta.max(axis=2) > 8)[far].sum())
    check("edits stay local to the face", touched_far == 0, f"{touched_far} distant px")

    out = swapper.finish(raw, tgt, mask, region, tone_match=0.7, blend=1.0)
    check("tone match + blend", out.shape == tgt.shape)
    check("blend=0 returns the original",
          np.array_equal(swapper.finish(raw, tgt, mask, region, 0.0, 0.0), tgt))

    os.makedirs(os.path.join(HERE, "outputs"), exist_ok=True)
    dest = os.path.join(HERE, "outputs", "selftest.jpg")
    check("imwrite", swapper.imwrite(dest, out), dest)

    # The checks that actually prove the swap did its job: whose face is it now?
    # ArcFace treats cosine > ~0.4 between two embeddings as the same person.
    print("\nidentity transfer")
    e_target = faces[0].raw.normed_embedding
    e_source = swapper.pick_source_face(src).raw.normed_embedding

    def face_in(image):
        """The swapped face in a result image, matched by position not by size."""
        found = swapper.detect(image)
        if not found:
            return None
        cx, cy = (faces[0].bbox[0] + faces[0].bbox[2]) / 2, \
                 (faces[0].bbox[1] + faces[0].bbox[3]) / 2
        return min(found, key=lambda f: (cx - (f.bbox[0] + f.bbox[2]) / 2) ** 2
                   + (cy - (f.bbox[1] + f.bbox[3]) / 2) ** 2)

    got = face_in(out)
    if not check("result face still detectable", got is not None):
        return 1
    e_result = got.raw.normed_embedding
    to_source = float(np.dot(e_result, e_source))
    to_target = float(np.dot(e_result, e_target))
    check("identity moved to the source person", to_source > 0.4,
          f"cosine {to_source:.3f}")
    check("identity left the original face", to_target < to_source,
          f"cosine {to_target:.3f}")

    # Swapping a face with its own photo must give that same face back; this
    # separates "the pipeline works" from "the source photo was a bad match".
    self_raw, self_mask, self_region = swapper.swap_identity(tgt, faces[0], tgt)
    self_out = swapper.finish(self_raw, tgt, self_mask, self_region, 0.7, 1.0)
    self_face = face_in(self_out)
    if self_face is None:
        check("self-swap preserves identity", False, "face not detectable")
    else:
        keep = float(np.dot(self_face.raw.normed_embedding, e_target))
        # Short of 1.0 only because of the 128x128 bottleneck, not a bug.
        check("self-swap preserves identity", keep > 0.7, f"cosine {keep:.3f}")

    print("\nFAILED: " + ", ".join(FAILED) if FAILED else "\nAll checks passed.")
    return 1 if FAILED else 0


if __name__ == "__main__":
    raise SystemExit(main())
