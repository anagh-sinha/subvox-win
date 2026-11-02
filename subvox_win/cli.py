"""Command line app: train / evaluate / decode.

Camera window keys: SPACE = start/stop mouthing a word, Q = quit.
"""
import argparse
import time
import cv2
import numpy as np
from . import dataset
from .classifier import TemplateClassifier
from .evaluation import evaluate
from .features import KEEP, segment_features


def _draw(img, lm, text, rec):
    if lm is not None:
        h, w = img.shape[:2]
        for i in KEEP:
            cv2.circle(img, (int(lm[i, 0] * w), int(lm[i, 1] * h)), 2, (0, 255, 0), -1)
    cv2.putText(img, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255) if rec else (255, 255, 255), 2)


def _capture_loop(cam, prompt, on_segment, stop_after=None):
    frames, rec, n = [], False, 0
    while True:
        img, vec, lm = cam.read()
        if img is None:
            break
        if rec and vec is not None:
            frames.append(vec)
        _draw(img, lm, ("REC " if rec else "") + prompt(), rec)
        cv2.imshow("subvox-win", img)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == 32:
            if rec:
                feats = segment_features(np.array(frames))
                if len(feats):
                    n += 1
                    on_segment(feats)
                frames = []
            rec = not rec
            if stop_after and n >= stop_after:
                break
    cv2.destroyAllWindows()


def cmd_train(a):
    from .camera import FaceCamera
    samples = dataset.load(a.data)
    cam = FaceCamera(a.camera)
    try:
        for word in a.words:
            for r in range(a.reps):
                done = []
                _capture_loop(cam, lambda: f"'{word}' rep {r + 1}/{a.reps} - SPACE to start/stop",
                              lambda f: (samples.append((word, f)), done.append(1)), stop_after=1)
                if not done:
                    dataset.save(samples, a.data)
                    return
    finally:
        cam.close()
        dataset.save(samples, a.data)
    print(f"Saved {len(samples)} samples to {a.data}")


def cmd_evaluate(a):
    s = dataset.load(a.data)
    if len({w for w, _ in s}) < 2:
        raise SystemExit("Need at least 2 words recorded (run `train` first).")
    r = evaluate(s, a.permutations)
    print(f"n={r['n']}  accuracy={r['accuracy']:.0%}  chance={r['chance']:.0%}  p={r['p_value']:.4f}"
          f"  -> {'REAL signal' if r['p_value'] < 0.05 else 'not distinguishable from chance'}")
    ws = list(r["confusion"])
    print("true\pred " + " ".join(f"{w:>8}" for w in ws))
    for t in ws:
        print(f"{t:>9} " + " ".join(f"{r['confusion'][t][p]:>8}" for p in ws))


def cmd_decode(a):
    from .camera import FaceCamera
    s = dataset.load(a.data)
    if not s:
        raise SystemExit("No dataset. Run `train` first.")
    clf = TemplateClassifier(a.max_distance, a.min_margin).fit(s)
    cam, out = FaceCamera(a.camera), []

    def on_seg(f):
        r = clf.predict(f)
        out.append(r.word or "?")
        print(f"{r.word or '?':>10}  dist={r.distance:.3f} margin={r.margin:.3f}")

    try:
        _capture_loop(cam, lambda: " ".join(out[-6:]) or "SPACE to mouth a word", on_seg)
    finally:
        cam.close()
    print("Transcript:", " ".join(out))


def main(argv=None):
    p = argparse.ArgumentParser(prog="subvox-win")
    p.add_argument("--data", default=str(dataset.DEFAULT))
    p.add_argument("--camera", type=int, default=0)
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train"); t.add_argument("words", nargs="+"); t.add_argument("--reps", type=int, default=10)
    t.set_defaults(f=cmd_train)
    e = sub.add_parser("evaluate"); e.add_argument("--permutations", type=int, default=500); e.set_defaults(f=cmd_evaluate)
    d = sub.add_parser("decode"); d.add_argument("--max-distance", type=float, default=float("inf"))
    d.add_argument("--min-margin", type=float, default=0.0); d.set_defaults(f=cmd_decode)
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
