import json
import os
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from concurrent.futures import ProcessPoolExecutor, as_completed

def _check_wechat_runtime(model_dir: Path) -> tuple[bool, str]:
    required = [
        "detect.prototxt",
        "detect.caffemodel",
        "sr.prototxt",
        "sr.caffemodel",
    ]
    missing = [name for name in required if not (model_dir / name).exists()]
    if missing:
        return False, f"WeChat模型缺失: {', '.join(missing)}"
    if not hasattr(cv2, "wechat_qrcode_WeChatQRCode"):
        return False, "当前OpenCV不支持WeChatQRCode（请安装 opencv-contrib-python）"
    return True, ""

def _prepare_wechat_model_dir(model_dir: Path) -> Path:
    required = ["detect.prototxt", "detect.caffemodel", "sr.prototxt", "sr.caffemodel"]
    try:
        str(model_dir).encode("ascii")
        return model_dir
    except UnicodeEncodeError:
        pass

    compat_dir = Path(tempfile.gettempdir()) / "pdfqr_wechat_models"
    compat_dir.mkdir(parents=True, exist_ok=True)
    for name in required:
        src = model_dir / name
        dst = compat_dir / name
        if src.exists():
            shutil.copy2(src, dst)
    return compat_dir

_opencv_detector = None
_wechat_detector = None

def _init_detectors(model_dir_str, use_wechat):
    global _opencv_detector, _wechat_detector
    _opencv_detector = cv2.QRCodeDetector()
    _wechat_detector = None
    if not use_wechat:
        return
    try:
        model_dir = Path(model_dir_str)
        actual_model_dir = _prepare_wechat_model_dir(model_dir)
        _wechat_detector = cv2.wechat_qrcode_WeChatQRCode(
            str(actual_model_dir / "detect.prototxt"),
            str(actual_model_dir / "detect.caffemodel"),
            str(actual_model_dir / "sr.prototxt"),
            str(actual_model_dir / "sr.caffemodel")
        )
    except Exception:
        _wechat_detector = None

def _render_page_to_bgr(page: fitz.Page, zoom: float) -> np.ndarray:
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    if pix.n == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if pix.n == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
    if pix.n == 1:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"Unsupported pixmap channels: {pix.n}")

def _decode_qr_opencv(detector, bgr: np.ndarray) -> list[str]:
    texts = []
    try:
        t, points, _ = detector.detectAndDecode(bgr)
        if t and t.strip() and points is not None and len(points) >= 4:
            texts.append(t.strip())
    except Exception:
        pass
    return texts

def _decode_qr_wechat(detector, bgr: np.ndarray) -> list[str]:
    try:
        result, points = detector.detectAndDecode(bgr)
        if result:
            if isinstance(result, (list, tuple)):
                return [r.strip() for r in result if r and r.strip()]
            elif isinstance(result, str) and result.strip():
                return [result.strip()]
    except Exception:
        pass
    return []

def _process_single_page(args):
    pdf_path_str, page_index, zoom, try_rotations, model_dir, use_wechat = args
    
    global _opencv_detector, _wechat_detector
    if _opencv_detector is None:
        _init_detectors(str(model_dir), use_wechat)
    
    page_texts = set()
    
    try:
        with fitz.open(pdf_path_str) as doc:
            page = doc.load_page(page_index - 1)
            bgr = _render_page_to_bgr(page, zoom=zoom)
            
            images_to_try = [bgr]
            
            if try_rotations:
                images_to_try.extend([
                    cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE),
                    cv2.rotate(bgr, cv2.ROTATE_180),
                    cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
                ])
            
            for img in images_to_try:
                for text in _decode_qr_opencv(_opencv_detector, img):
                    page_texts.add(text)
                
                if _wechat_detector:
                    for text in _decode_qr_wechat(_wechat_detector, img):
                        page_texts.add(text)
    
    except Exception:
        pass
    
    return page_index, list(page_texts)

def extract_pdf_qr(pdf_path, model_dir, zoom, try_rotations, use_wechat, workers, progress_cb, log_cb, count_cb=None):
    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
    
    progress_cb(0, total_pages)
    log_cb(f"PDF页数: {total_pages} | 缩放: {zoom} | 旋转: {try_rotations} | WeChat: {use_wechat} | 进程: {workers}")
    
    tasks = []
    for i in range(1, total_pages + 1):
        tasks.append((str(pdf_path), i, zoom, try_rotations, str(model_dir), use_wechat))
    
    all_results = {}
    seen_texts = set()
    done = 0
    
    if workers <= 1:
        _init_detectors(str(model_dir), use_wechat)
        for task in tasks:
            page_idx, texts = _process_single_page(task)
            if texts:
                all_results[page_idx] = texts
                for t in texts:
                    seen_texts.add(t)
                if count_cb:
                    count_cb(len(seen_texts))
            done += 1
            progress_cb(done, total_pages)
            if done % 100 == 0:
                log_cb(f"[{done}/{total_pages}] 已处理 {done} 页")
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_detectors,
            initargs=(str(model_dir), use_wechat),
        ) as executor:
            future_to_page = {executor.submit(_process_single_page, t): t[1] for t in tasks}
            
            for future in as_completed(future_to_page):
                page_idx = future_to_page[future]
                try:
                    _, texts = future.result()
                    if texts:
                        all_results[page_idx] = texts
                        for t in texts:
                            seen_texts.add(t)
                        if count_cb:
                            count_cb(len(seen_texts))
                except Exception:
                    pass
                
                done += 1
                progress_cb(done, total_pages)
                if done % 100 == 0:
                    log_cb(f"[{done}/{total_pages}] 已处理 {done} 页")
    
    final_results = []
    for page in sorted(all_results.keys()):
        for content in all_results[page]:
            final_results.append({"page": page, "content": content})
    
    seen = set()
    unique_results = []
    for item in final_results:
        key = item['content']
        if key not in seen:
            seen.add(key)
            unique_results.append(item)
    
    return unique_results

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("增强版二维码识别器")
        self.minsize(860, 320)
        
        self._q = queue.Queue()
        self._stop_flag = threading.Event()
        self._running = False
        
        self.pdf_var = tk.StringVar(value=str(Path.cwd() / "qr_doc.pdf"))
        default_pdf = Path(self.pdf_var.get())
        self.out_var = tk.StringVar(value=str(self._build_output_path(default_pdf)))
        self.zoom_var = tk.StringVar(value="3.0")
        self.workers_var = tk.StringVar(value="8")
        self.wechat_var = tk.BooleanVar(value=True)
        self.rotate_var = tk.BooleanVar(value=True)
        self.count_var = tk.StringVar(value="0")
        self.total_var = tk.StringVar(value="0")
        self.time_var = tk.StringVar(value="")
        
        self._build_ui()
        self.update_idletasks()
        self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")
        self.after(80, self._poll_queue)
    
    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.X, expand=False)
        
        row = 0
        ttk.Label(frm, text="PDF 文件").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.pdf_var, width=60).grid(row=row, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(frm, text="选择…", command=self._pick_pdf).grid(row=row, column=2, sticky="ew")
        
        row += 1
        ttk.Label(frm, text="输出文件（自动）").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(frm, textvariable=self.out_var, width=60, state="readonly").grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Label(frm, text="随PDF自动生成").grid(row=row, column=2, sticky="w", pady=(8, 0))
        
        row += 1
        opts = ttk.Frame(frm)
        opts.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        ttk.Label(opts, text="缩放倍数").grid(row=0, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.zoom_var, width=8).grid(row=0, column=1, sticky="w", padx=(6, 16))
        
        ttk.Label(opts, text="进程数").grid(row=0, column=2, sticky="w")
        ttk.Entry(opts, textvariable=self.workers_var, width=8).grid(row=0, column=3, sticky="w", padx=(6, 16))
        
        ttk.Checkbutton(opts, text="使用WeChat识别", variable=self.wechat_var).grid(row=0, column=4, sticky="w", padx=(0, 12))
        ttk.Checkbutton(opts, text="尝试旋转", variable=self.rotate_var).grid(row=0, column=5, sticky="w", padx=(0, 12))
        
        opts.columnconfigure(6, weight=1)
        
        row += 1
        btns = ttk.Frame(frm)
        btns.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        self.start_btn = ttk.Button(btns, text="开始识别", command=self._start)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btns, text="停止", command=self._stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        
        self.prog = ttk.Progressbar(btns, mode="determinate")
        self.prog.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(12, 0))
        
        row += 1
        self.status = ttk.Label(frm, text="就绪")
        self.status.grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 0))
        
        row += 1
        stats_row = ttk.Frame(frm)
        stats_row.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        
        ttk.Label(stats_row, text="PDF页数:").pack(side=tk.LEFT)
        ttk.Label(stats_row, textvariable=self.total_var, font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=(4, 16))
        
        ttk.Label(stats_row, text="已识别:").pack(side=tk.LEFT)
        ttk.Label(stats_row, textvariable=self.count_var, font=('Arial', 12, 'bold'), foreground='green').pack(side=tk.LEFT, padx=(4, 16))
        
        ttk.Label(stats_row, text="耗时:").pack(side=tk.LEFT)
        ttk.Label(stats_row, textvariable=self.time_var, font=('Arial', 12, 'bold')).pack(side=tk.LEFT, padx=(4, 0))
        
        stats_row.columnconfigure(1, weight=1)
        
        frm.columnconfigure(1, weight=1)
    
    def _build_output_path(self, pdf_path: Path) -> Path:
        pdf_abs = pdf_path.expanduser().resolve()
        script_dir = Path(__file__).resolve().parent
        out_dir = script_dir / "output"
        return out_dir / f"{pdf_abs.stem}_result.json"

    def _pick_pdf(self):
        p = filedialog.askopenfilename(title="选择PDF文件", filetypes=[("PDF", "*.pdf"), ("All files", "*.*")])
        if p:
            pdf_path = Path(p)
            self.pdf_var.set(str(pdf_path))
            self.out_var.set(str(self._build_output_path(pdf_path)))
    
    def _pick_out(self):
        p = filedialog.asksaveasfilename(title="保存结果", defaultextension=".json", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if p:
            self.out_var.set(p)
    
    def _set_running(self, running):
        self._running = running
        self.start_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)
    
    def _start(self):
        if self._running:
            messagebox.showinfo("提示", "任务正在运行中")
            return
        
        pdf = Path(self.pdf_var.get()).expanduser()
        out = self._build_output_path(pdf)
        self.out_var.set(str(out))
        
        if not pdf.exists():
            messagebox.showerror("错误", f"PDF文件不存在: {pdf}")
            return
        
        try:
            zoom = float(self.zoom_var.get())
        except ValueError:
            messagebox.showerror("错误", "缩放倍数请输入数字")
            return
        
        try:
            workers = int(self.workers_var.get())
        except ValueError:
            messagebox.showerror("错误", "进程数请输入整数")
            return
        
        self._stop_flag.clear()
        self.prog.configure(maximum=100, value=0)
        self.status.configure(text="运行中…")
        self.count_var.set("0")
        self.total_var.set("0")
        self.time_var.set("")
        self._set_running(True)

        if self.wechat_var.get():
            model_dir = Path(__file__).resolve().parent / "opencv_3rdparty"
            ok, reason = _check_wechat_runtime(model_dir)
            if not ok:
                self.wechat_var.set(False)
                messagebox.showwarning(
                    "WeChat识别不可用",
                    f"{reason}\n\n已自动关闭“使用WeChat识别”，将仅使用OpenCV原生识别。"
                )
        
        t = threading.Thread(
            target=self._run_job,
            args=(pdf, out, zoom, max(1, workers), self.rotate_var.get(), self.wechat_var.get()),
            daemon=True
        )
        t.start()
    
    def _stop(self):
        self._stop_flag.set()
        self.status.configure(text="正在停止…")
    
    def _run_job(self, pdf, out, zoom, workers, rotate, use_wechat):
        def progress_cb(cur, total):
            if self._stop_flag.is_set():
                return
            self._q.put(("progress", (cur, total)))
            elapsed = time.time() - t0
            self._q.put(("time", f"{elapsed:.2f}s"))
        
        def count_cb(count):
            if self._stop_flag.is_set():
                return
            self._q.put(("count", str(count)))
        
        try:
            t0 = time.time()
            
            with fitz.open(pdf) as doc:
                total_pages = doc.page_count
            self._q.put(("total", str(total_pages)))
            
            results = extract_pdf_qr(
                pdf_path=pdf,
                model_dir=(Path(__file__).resolve().parent / "opencv_3rdparty"),
                zoom=zoom,
                try_rotations=rotate,
                use_wechat=use_wechat,
                workers=workers,
                progress_cb=progress_cb,
                log_cb=lambda msg: None,
                count_cb=count_cb,
            )
            elapsed = time.time() - t0
            
            if self._stop_flag.is_set():
                self._q.put(("status", "已停止"))
                self._q.put(("count", str(len(results))))
                self._q.put(("time", f"{elapsed:.2f}s"))
                self._q.put(("done", None))
                return
            
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

            out_txt = out.with_name(f"{out.stem}_pages.txt")
            lines = [str(item.get("content", "")).strip() for item in sorted(results, key=lambda x: int(x.get("page", 0)))]
            lines = [line for line in lines if line]
            out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            
            self._q.put(("status", "完成"))
            self._q.put(("count", str(len(results))))
            self._q.put(("time", f"{elapsed:.2f}s"))
            self._q.put(("done", None))
        
        except Exception as e:
            self._q.put(("status", f"失败: {e}"))
            self._q.put(("count", "0"))
            self._q.put(("done", None))
    
    def _poll_queue(self):
        try:
            while True:
                typ, payload = self._q.get_nowait()
                if typ == "status":
                    self.status.configure(text=str(payload))
                elif typ == "count":
                    self.count_var.set(str(payload))
                elif typ == "total":
                    self.total_var.set(str(payload))
                elif typ == "time":
                    self.time_var.set(str(payload))
                elif typ == "progress":
                    cur, total = payload
                    if total > 0:
                        self.prog.configure(maximum=total, value=cur)
                elif typ == "done":
                    self._set_running(False)
        except queue.Empty:
            pass
        finally:
            self.after(80, self._poll_queue)

def main():
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()