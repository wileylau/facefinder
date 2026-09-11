import os
import sys
import shutil
import threading
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import cv2
import numpy as np

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

IMAGE_EXTS = ('.jpg', '.jpeg', '.png', '.webp', '.bmp')


def bundled_model_path():
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, 'buffalo_s')
    return None


def ensure_model():
    model_dir = os.path.join(os.path.expanduser('~'), '.insightface', 'models', 'buffalo_s')
    if os.path.isdir(model_dir) and os.listdir(model_dir):
        return
    src = bundled_model_path()
    if src is None or not os.path.isdir(src):
        return
    os.makedirs(model_dir, exist_ok=True)
    for f in os.listdir(src):
        shutil.copy2(os.path.join(src, f), os.path.join(model_dir, f))


class FaceFinderApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Finder")
        self.geometry("620x520")
        self.resizable(False, False)
        self._build_ui()

    def _build_ui(self):
        pad = {"padx": 16, "pady": (10, 0)}

        # --- Folder picks ---
        ctk.CTkLabel(self, text="Source Folder").pack(anchor="w", **pad)
        self.source_var = ctk.StringVar()
        src_row = ctk.CTkFrame(self, fg_color="transparent")
        src_row.pack(fill="x", **pad)
        ctk.CTkEntry(src_row, textvariable=self.source_var, placeholder_text="Folder with photos to scan").pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(src_row, text="Browse", width=80, command=self._pick_source).pack(side="right")

        ctk.CTkLabel(self, text="Output Folder").pack(anchor="w", **pad)
        self.out_var = ctk.StringVar()
        out_row = ctk.CTkFrame(self, fg_color="transparent")
        out_row.pack(fill="x", **pad)
        ctk.CTkEntry(out_row, textvariable=self.out_var, placeholder_text="Where matched photos go").pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(out_row, text="Browse", width=80, command=self._pick_output).pack(side="right")

        ctk.CTkLabel(self, text="Reference Photo (face to find)").pack(anchor="w", **pad)
        ref_row = ctk.CTkFrame(self, fg_color="transparent")
        ref_row.pack(fill="x", **pad)
        self.ref_var = ctk.StringVar()
        ctk.CTkEntry(ref_row, textvariable=self.ref_var, placeholder_text="Photo of the person").pack(side="left", fill="x", expand=True, padx=(0, 6))
        ctk.CTkButton(ref_row, text="Browse", width=80, command=self._pick_ref).pack(side="right")

        # --- Advanced (collapsible) ---
        self.adv_open = False
        self.adv_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.adv_frame.pack(fill="x", **pad)
        self.adv_btn = ctk.CTkButton(self.adv_frame, text="Advanced \u25bc", width=100, fg_color="#555", command=self._toggle_advanced)
        self.adv_btn.pack(anchor="w")

        self.adv_content = ctk.CTkFrame(self.adv_frame, fg_color="transparent")
        self.threshold_var = ctk.DoubleVar(value=0.40)
        self.workers_var = ctk.IntVar(value=8)

        thr_row = ctk.CTkFrame(self.adv_content, fg_color="transparent")
        thr_row.pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(thr_row, text="Similarity threshold:").pack(side="left")
        self.thr_label = ctk.CTkLabel(thr_row, text="0.40")
        self.thr_label.pack(side="right")
        ctk.CTkSlider(self.adv_content, from_=0.20, to=0.80, number_of_steps=60,
                       variable=self.threshold_var, command=self._on_thr_change).pack(fill="x", pady=(2, 0))

        w_row = ctk.CTkFrame(self.adv_content, fg_color="transparent")
        w_row.pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(w_row, text="Worker threads:").pack(side="left")
        ctk.CTkEntry(w_row, textvariable=self.workers_var, width=50).pack(side="right")

        # --- Options ---
        self.autoopen_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self, text="Open output folder when done", variable=self.autoopen_var).pack(anchor="w", **pad)

        # --- Actions ---
        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack(fill="x", pady=(14, 0), padx=16)
        self.start_btn = ctk.CTkButton(btn_row, text="Start Scanning", height=36, command=self._start)
        self.start_btn.pack(side="left", expand=True, fill="x", padx=(0, 6))
        self.stop_btn = ctk.CTkButton(btn_row, text="Stop", height=36, fg_color="#c0392b", hover_color="#e74c3c",
                                       state="disabled", command=self._stop)
        self.stop_btn.pack(side="right", expand=True, fill="x")

        # --- Progress ---
        self.progress = ctk.CTkProgressBar(self)
        self.progress.pack(fill="x", padx=16, pady=(12, 0))
        self.progress.set(0)

        self.status_label = ctk.CTkLabel(self, text="Ready", text_color="#aaa")
        self.status_label.pack(anchor="w", padx=16, pady=(4, 0))

        # --- Log ---
        self.log = ctk.CTkTextbox(self, state="disabled", height=120, font=("Consolas", 12))
        self.log.pack(fill="both", expand=True, padx=16, pady=(6, 12))

        self._stop_event = threading.Event()

    # -- Folder/file pickers --
    def _pick_source(self):
        d = filedialog.askdirectory(title="Select source folder")
        if d:
            self.source_var.set(d)

    def _pick_output(self):
        d = filedialog.askdirectory(title="Select output folder")
        if d:
            self.out_var.set(d)

    def _pick_ref(self):
        f = filedialog.askopenfilename(title="Select reference photo",
                                        filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
        if f:
            self.ref_var.set(f)

    # -- Advanced toggle --
    def _toggle_advanced(self):
        self.adv_open = not self.adv_open
        if self.adv_open:
            self.adv_content.pack(fill="x")
            self.adv_btn.configure(text="Advanced \u25b2")
        else:
            self.adv_content.pack_forget()
            self.adv_btn.configure(text="Advanced \u25bc")

    def _on_thr_change(self, val):
        self.thr_label.configure(text=f"{val:.2f}")

    # -- Logging --
    def _log(self, msg):
        self.log.configure(state="normal")
        self.log.insert("end", msg + "\n")
        self.log.see("end")
        self.log.configure(state="disabled")

    def _set_status(self, text):
        self.status_label.configure(text=text)

    # -- Start / Stop --
    def _start(self):
        src = self.source_var.get().strip()
        out = self.out_var.get().strip()
        ref = self.ref_var.get().strip()

        if not all([src, out, ref]):
            messagebox.showwarning("Missing", "Please select source folder, output folder, and reference photo.")
            return
        if not os.path.isdir(src):
            messagebox.showerror("Error", "Source folder does not exist.")
            return
        if not os.path.isfile(ref):
            messagebox.showerror("Error", "Reference photo not found.")
            return

        os.makedirs(out, exist_ok=True)
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self._stop_event.clear()
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress.set(0)

        threading.Thread(target=self._run_scan, args=(src, out, ref), daemon=True).start()

    def _stop(self):
        self._stop_event.set()
        self._set_status("Stopping...")

    # -- Core scan --
    def _run_scan(self, src, out, ref):
        try:
            import cv2
            import numpy as np
            from insightface.app import FaceAnalysis
            from concurrent.futures import ThreadPoolExecutor, as_completed
        except ImportError as e:
            self.after(0, lambda: messagebox.showerror("Missing dependency", str(e)))
            self.after(0, self._reset_buttons)
            return

        def load_image(path):
            try:
                pil_img = Image.open(path).convert('RGB')
                return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            except Exception:
                return None

        # Collect images
        all_files = []
        for root, _, files in os.walk(src):
            for f in files:
                if f.lower().endswith(IMAGE_EXTS):
                    all_files.append(os.path.join(root, f))

        total = len(all_files)
        self.after(0, lambda: self._log(f"Found {total} images"))
        if total == 0:
            self.after(0, lambda: self._set_status("No images found."))
            self.after(0, self._reset_buttons)
            return

        # Load model
        self.after(0, lambda: self._set_status("Loading face model..."))
        ensure_model()
        app = FaceAnalysis(name='buffalo_s', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Load reference
        ref_img = load_image(ref)
        if ref_img is None:
            self.after(0, lambda: messagebox.showerror("Error", "Could not load reference image."))
            self.after(0, self._reset_buttons)
            return

        ref_faces = app.get(ref_img)
        if not ref_faces:
            self.after(0, lambda: messagebox.showerror("Error", "No face detected in reference photo."))
            self.after(0, self._reset_buttons)
            return

        ref_emb = ref_faces[0].embedding
        self.after(0, lambda: self._log("Reference face loaded. Scanning...\n"))

        threshold = self.threshold_var.get()
        workers = self.workers_var.get()

        def process_one(file_path):
            if self._stop_event.is_set():
                return None
            img = load_image(file_path)
            if img is None:
                return None
            faces = app.get(img)
            filename = os.path.basename(file_path)
            for face in faces:
                sim = np.dot(ref_emb, face.embedding) / (
                    np.linalg.norm(ref_emb) * np.linalg.norm(face.embedding)
                )
                if sim > threshold:
                    dest = os.path.join(out, filename)
                    if os.path.exists(dest):
                        dest = os.path.join(out, f"{hash(file_path)}_{filename}")
                    shutil.copy(file_path, dest)
                    return (filename, sim)
            return None

        matched = 0
        done = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(process_one, fp): fp for fp in all_files}
            for future in as_completed(futures):
                if self._stop_event.is_set():
                    pool.shutdown(wait=False, cancel_futures=True)
                    break
                done += 1
                result = future.result()
                if result:
                    matched += 1
                    fname, sim = result
                    self.after(0, lambda f=fname, s=sim: self._log(f"[MATCH] {f}  (sim: {s:.2f})"))
                if done % 25 == 0 or done == total:
                    pct = done / total
                    self.after(0, lambda p=pct, d=done: (
                        self.progress.set(p),
                        self._set_status(f"Scanned {d}/{total}")
                    ))

        summary = f"Done. {matched} matches out of {total} photos."
        self.after(0, lambda: self._log(f"\n{summary}"))
        self.after(0, lambda: self._set_status(summary))

        if self.autoopen_var.get() and matched > 0:
            os.startfile(out)

        self.after(0, self._reset_buttons)

    def _reset_buttons(self):
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")


if __name__ == "__main__":
    app = FaceFinderApp()
    app.mainloop()
