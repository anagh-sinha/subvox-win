# subvox-win

Windows port of the idea behind [joeblau/subvox](https://github.com/joeblau/subvox): decode silently
mouthed words from motion, using **template matching, no language model**.

The original reads AirPods head motion through Apple's CoreMotion, which Windows doesn't have. This
version uses your **webcam**: MediaPipe FaceMesh tracks lip + chin landmarks, normalised for head
position, roll and distance, and each mouthed word is matched to your own recorded templates.

Pipeline (same as the original): space-toggle segmentation -> mean-centred landmark motion features ->
z-scaled 1-NN DTW (Sakoe-Chiba band) -> leave-one-out accuracy + 500x label-permutation test (p < 0.05
= real signal).

## Setup
```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

## Use
Double-click `Subvox.bat` (or run `python -m subvox_win`) for the desktop app: live camera plus
Train / Evaluate / Decode tabs. Space starts/stops a recording.

Command line:
```
python -m subvox_win train yes no stop go --reps 10   # SPACE starts / stops each word, Q quits
python -m subvox_win evaluate                          # accuracy, p-value, confusion matrix
python -m subvox_win decode --min-margin 0.05          # SPACE to mouth a word, prints transcript
```
Data lives in `data/dataset.json` (gitignored). Use `--data` / `--camera` to change.

## Tips
- Keep lighting, camera angle and seating consistent between training and decoding.
- Record 10+ reps per word; start with 3-4 clearly different words and check `evaluate` before `decode`.
- Tune `--max-distance` / `--min-margin` to reject unrecognised mouthing.

## Tests
`python -m pytest`
