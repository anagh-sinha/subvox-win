"""Desktop app: live camera + Train / Evaluate / Decode tabs (tkinter)."""
import base64
import threading
import tkinter as tk
from collections import Counter
from tkinter import messagebox, ttk

import cv2
import numpy as np

from . import dataset
from .classifier import TemplateClassifier
from .evaluation import evaluate
from .features import KEEP, segment_features

W, H = 640, 480


class App:
    def __init__(self, root, camera=0, data=dataset.DEFAULT):
        from .camera import FaceCamera
        self.root, self.data = root, data
        root.title("Subvox for Windows")
        root.resizable(False, False)
        self.samples = dataset.load(data)
        self.cam = FaceCamera(camera)
        self.rec, self.frames, self.clf = False, [], None

        left = ttk.Frame(root)
        left.grid(row=0, column=0, padx=8, pady=8)
        self.canvas = tk.Canvas(left, width=W, height=H, bg="black", highlightthickness=0)
        self.canvas.pack()
        self.status = ttk.Label(left, text="Starting camera...", font=("Segoe UI", 11))
        self.status.pack(anchor="w", pady=(6, 0))

        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="ns", padx=(0, 8), pady=8)
        self.tabs = ttk.Notebook(right)
        self.tabs.pack(fill="both", expand=True)
        self._train_tab()
        self._eval_tab()
        self._decode_tab()
        self.tabs.bind("<<NotebookTabChanged>>", lambda e: self._tab_changed())

        root.bind("<space>", self._space)
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._refresh_words()
        self.tick()

    # ---- tabs
    def _train_tab(self):
        f = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(f, text="Train")
        ttk.Label(f, text="Word to train:").pack(anchor="w")
        self.word = ttk.Entry(f, width=24)
        self.word.pack(anchor="w", pady=4)
        self.rec_btn = ttk.Button(f, text="Record (Space)", command=self.toggle, takefocus=0)
        self.rec_btn.pack(anchor="w", pady=4)
        ttk.Label(f, text="Mouth the word silently between start\nand stop. Aim for 10+ reps per word.",
                  foreground="#555").pack(anchor="w", pady=4)
        self.words_box = tk.Listbox(f, height=10, width=30, exportselection=False)
        self.words_box.pack(anchor="w", pady=6)
        self.words_box.bind("<<ListboxSelect>>", lambda e: self._pick_word())
        ttk.Button(f, text="Delete selected word", command=self.delete_word, takefocus=0).pack(anchor="w")

    def _eval_tab(self):
        f = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(f, text="Evaluate")
        ttk.Label(f, text="Is there a real signal? (leave-one-out + permutation test)",
                  wraplength=320).pack(anchor="w")
        self.eval_btn = ttk.Button(f, text="Run evaluation", command=self.run_eval, takefocus=0)
        self.eval_btn.pack(anchor="w", pady=6)
        self.eval_out = tk.Text(f, width=44, height=18, font=("Consolas", 10), state="disabled")
        self.eval_out.pack()

    def _decode_tab(self):
        f = ttk.Frame(self.tabs, padding=10)
        self.tabs.add(f, text="Decode")
        self.dec_btn = ttk.Button(f, text="Mouth a word (Space)", command=self.toggle, takefocus=0)
        self.dec_btn.pack(anchor="w", pady=4)
        row = ttk.Frame(f)
        row.pack(anchor="w", pady=4)
        ttk.Label(row, text="Min margin:").pack(side="left")
        self.margin = tk.DoubleVar(value=0.0)
        ttk.Spinbox(row, from_=0, to=2, increment=0.02, width=6, textvariable=self.margin).pack(side="left", padx=4)
        ttk.Label(row, text="Max dist:").pack(side="left", padx=(8, 0))
        self.maxd = tk.DoubleVar(value=99.0)
        ttk.Spinbox(row, from_=0, to=99, increment=0.1, width=6, textvariable=self.maxd).pack(side="left", padx=4)
        self.big = ttk.Label(f, text="", font=("Segoe UI", 28, "bold"))
        self.big.pack(anchor="w", pady=6)
        self.transcript = tk.Text(f, width=44, height=10, font=("Segoe UI", 12), state="disabled")
        self.transcript.pack()
        ttk.Button(f, text="Clear", command=lambda: self._set(self.transcript, ""), takefocus=0).pack(anchor="w", pady=4)

    # ---- helpers
    def _set(self, box, text, append=False):
        box.config(state="normal")
        if not append:
            box.delete("1.0", "end")
        box.insert("end", text)
        box.see("end")
        box.config(state="disabled")

    def _mode(self):
        return self.tabs.tab(self.tabs.select(), "text")

    def _tab_changed(self):
        if self.rec:
            self.toggle(cancel=True)
        if self._mode() == "Decode":
            self._build_clf()

    def _build_clf(self):
        if len({w for w, _ in self.samples}) < 2:
            self.clf = None
            self.status.config(text="Train at least 2 words first.")
        else:
            self.clf = TemplateClassifier().fit(self.samples)

    def _refresh_words(self):
        c = Counter(w for w, _ in self.samples)
        self.words_box.delete(0, "end")
        for w, n in sorted(c.items()):
            self.words_box.insert("end", f"{w}  ({n} samples)")

    def _pick_word(self):
        sel = self.words_box.curselection()
        if sel:
            self.word.delete(0, "end")
            self.word.insert(0, self.words_box.get(sel[0]).rsplit("  (", 1)[0])

    def delete_word(self):
        sel = self.words_box.curselection()
        if not sel:
            return
        w = self.words_box.get(sel[0]).rsplit("  (", 1)[0]
        if messagebox.askyesno("Delete", f"Delete all samples of '{w}'?"):
            self.samples = [(a, b) for a, b in self.samples if a != w]
            dataset.save(self.samples, self.data)
            self._refresh_words()

    # ---- recording
    def _space(self, e):
        if e.widget.winfo_class() in ("Entry", "TEntry", "Text", "TSpinbox", "Spinbox"):
            return
        self.toggle()

    def toggle(self, cancel=False):
        mode = self._mode()
        if not self.rec:
            if mode == "Evaluate":
                return
            if mode == "Train" and not self.word.get().strip():
                self.status.config(text="Type the word to train first.")
                return
            if mode == "Decode" and self.clf is None:
                self._build_clf()
                if self.clf is None:
                    return
            self.rec, self.frames = True, []
        else:
            self.rec = False
            if not cancel:
                feats = segment_features(np.array(self.frames)) if self.frames else []
                if len(feats) < 4:
                    self.status.config(text="Too short / no face tracked - try again.")
                elif mode == "Train":
                    self.samples.append((self.word.get().strip(), feats))
                    dataset.save(self.samples, self.data)
                    self._refresh_words()
                    self.status.config(text=f"Saved sample of '{self.word.get().strip()}'.")
                elif mode == "Decode":
                    self.on_decode(feats)
        for b, t in ((self.rec_btn, "Record (Space)"), (self.dec_btn, "Mouth a word (Space)")):
            b.config(text="Stop (Space)" if self.rec else t)

    def on_decode(self, feats):
        self.clf.max_distance, self.clf.min_margin = self.maxd.get(), self.margin.get()
        r = self.clf.predict(feats)
        self.big.config(text=r.word or "?")
        self._set(self.transcript, (r.word or "?") + " ", append=True)
        self.status.config(text=f"dist={r.distance:.3f}  margin={r.margin:.3f}")

    def run_eval(self):
        if len({w for w, _ in self.samples}) < 2:
            self._set(self.eval_out, "Train at least 2 words first.")
            return
        self.eval_btn.config(state="disabled")
        self._set(self.eval_out, "Running...")
        data = list(self.samples)

        def work():
            try:
                r = evaluate(data, 500)
                ws = list(r["confusion"])
                lines = [f"n={r['n']}  accuracy={r['accuracy']:.0%}  chance={r['chance']:.0%}",
                         f"p-value={r['p_value']:.4f}  ->  "
                         + ("REAL signal" if r["p_value"] < 0.05 else "not distinguishable from chance"),
                         "", "true\\pred " + " ".join(f"{w[:7]:>7}" for w in ws)]
                lines += [f"{t[:9]:>9} " + " ".join(f"{r['confusion'][t][p]:>7}" for p in ws) for t in ws]
                out = "\n".join(lines)
            except Exception as ex:
                out = f"Error: {ex}"
            self.root.after(0, lambda: (self._set(self.eval_out, out), self.eval_btn.config(state="normal")))

        threading.Thread(target=work, daemon=True).start()

    # ---- camera loop
    def tick(self):
        img, vec, lm = self.cam.read()
        if img is not None:
            if lm is not None:
                h, w = img.shape[:2]
                for i in KEEP:
                    cv2.circle(img, (int(lm[i, 0] * w), int(lm[i, 1] * h)), 2, (0, 255, 0), -1)
            if self.rec:
                cv2.circle(img, (30, 30), 12, (0, 0, 255), -1)
                if vec is not None:
                    self.frames.append(vec)
            img = cv2.resize(img, (W, H))
            ok, png = cv2.imencode(".png", img)
            self.photo = tk.PhotoImage(data=base64.b64encode(png.tobytes()))
            self.canvas.create_image(0, 0, anchor="nw", image=self.photo)
            if not self.rec and self.status.cget("text") in ("Starting camera...", "No face detected", "Ready"):
                self.status.config(text="Ready" if lm is not None else "No face detected")
        self.root.after(15, self.tick)

    def close(self):
        self.cam.close()
        self.root.destroy()


def main():
    root = tk.Tk()
    try:
        App(root)
    except RuntimeError as e:
        messagebox.showerror("Subvox", str(e))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
