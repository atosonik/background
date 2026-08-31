"""Headless face swap.

    python cli.py photo.jpg --list -o boxes.jpg        # numbered preview
    python cli.py photo.jpg new.jpg --face 2 -o out.jpg
    python cli.py photo.jpg new.jpg --point 812 430 -o out.jpg
"""
import argparse
import sys

import swapper


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("target", help="the photo containing the face to replace")
    p.add_argument("source", nargs="?", help="photo of the new person (omit with --list)")
    p.add_argument("-o", "--out", default=None,
                   help="output path (default out.jpg; --list writes nothing "
                        "unless you name one)")
    p.add_argument("--face", type=int, metavar="N",
                   help="which face to replace, numbered as --list shows them")
    p.add_argument("--point", type=int, nargs=2, metavar=("X", "Y"),
                   help="replace whichever face contains this pixel")
    p.add_argument("--list", action="store_true",
                   help="write a numbered preview of every detected face and exit")
    p.add_argument("--tone", type=float, default=0.7, metavar="0..1",
                   help="skin-tone match toward the original face (default 0.7)")
    p.add_argument("--blend", type=float, default=1.0, metavar="0..1",
                   help="swap strength (default 1.0)")
    a = p.parse_args()

    img = swapper.imread(a.target)
    if img is None:
        print(f"Cannot read {a.target}", file=sys.stderr)
        return 1

    faces = swapper.detect(img)
    if not faces:
        print("No faces detected.", file=sys.stderr)
        return 1

    if a.list:
        for f in faces:
            # 1-based to match the numbers drawn on the preview and shown in the GUI.
            print(f"{f.index + 1}  bbox={f.bbox}")
        if a.out:
            swapper.imwrite(a.out, swapper.draw_overlay(img, faces))
            print(f"Preview written to {a.out}")
        return 0

    if a.source is None:
        print("Need a source image (or pass --list).", file=sys.stderr)
        return 2

    if a.point:
        chosen = swapper.face_at_point(faces, *a.point)
        if chosen is None:
            print(f"No face contains {tuple(a.point)}. Use --list to see the boxes.",
                  file=sys.stderr)
            return 2
    elif a.face is not None:
        if not 1 <= a.face <= len(faces):
            print(f"--face must be 1..{len(faces)}", file=sys.stderr)
            return 2
        chosen = faces[a.face - 1]
    else:
        print("Pass --face N or --point X Y (use --list to see them).", file=sys.stderr)
        return 2

    src = swapper.imread(a.source)
    if src is None:
        print(f"Cannot read {a.source}", file=sys.stderr)
        return 1

    try:
        result = swapper.swap(img, chosen, src, tone_match=a.tone, blend=a.blend)
    except (ValueError, FileNotFoundError) as exc:
        print(str(exc), file=sys.stderr)
        return 1

    out_path = a.out or "out.jpg"
    if not swapper.imwrite(out_path, result):
        print(f"Could not write {out_path}", file=sys.stderr)
        return 1
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
