import json
import multiprocessing
import os
import queue
import shutil
import sys
import tempfile
import threading
import time
import ctypes
from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

def _get_base_dir() -> Path:
    if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent

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


def _get_recommended_workers_limit() -> int:
    cpu = os.cpu_count() or 1
    cpu_cap = max(1, cpu * 2)
    # Estimate memory-based cap to avoid overcommitting workers on low-memory machines.
    mem_cap = cpu_cap
    if os.name == "nt":
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = MEMORYSTATUSEX()
        stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            # Reserve ~1GB per worker as a conservative default for PDF rendering workload.
            mem_cap = max(1, int(stat.ullAvailPhys // (1024 ** 3)))
    return max(1, min(cpu_cap, mem_cap))

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

def _chunked(items, chunk_size):
    if chunk_size <= 0:
        chunk_size = 1
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]

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
    texts = set()
    try:
        ok, decoded_info, points, _ = detector.detectAndDecodeMulti(bgr)
        if ok and decoded_info is not None:
            for text in decoded_info:
                if text and str(text).strip():
                    texts.add(str(text).strip())
    except Exception:
        pass
    if texts:
        return list(texts)

    try:
        t, points, _ = detector.detectAndDecode(bgr)
        if t and t.strip() and points is not None and len(points) >= 4:
            texts.add(t.strip())
    except Exception:
        pass
    return list(texts)

def _decode_qr_wechat(detector, bgr: np.ndarray) -> list[str]:
    try:
        result, _ = detector.detectAndDecode(bgr)
        if result:
            if isinstance(result, (list, tuple)):
                return [r.strip() for r in result if r and r.strip()]
            elif isinstance(result, str) and result.strip():
                return [result.strip()]
    except Exception:
        pass
    return []

def _process_page_image(detectors, bgr: np.ndarray, try_rotations: bool) -> list[str]:
    opencv_detector, wechat_detector = detectors
    page_texts = set()

    images_to_try = [bgr]
    if try_rotations:
        images_to_try.extend([
            cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(bgr, cv2.ROTATE_180),
            cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
        ])

    for img in images_to_try:
        for text in _decode_qr_opencv(opencv_detector, img):
            page_texts.add(text)
        if wechat_detector:
            for text in _decode_qr_wechat(wechat_detector, img):
                page_texts.add(text)
    return list(page_texts)

def _process_page_batch(args):
    pdf_path_str, page_indexes, zoom, try_rotations, model_dir, use_wechat = args
    
    global _opencv_detector, _wechat_detector
    if _opencv_detector is None:
        _init_detectors(str(model_dir), use_wechat)
    
    batch_results = []
    errors = []
    
    try:
        with fitz.open(pdf_path_str) as doc:
            for page_index in page_indexes:
                try:
                    page = doc.load_page(page_index - 1)
                    bgr = _render_page_to_bgr(page, zoom=zoom)
                    texts = _process_page_image((_opencv_detector, _wechat_detector), bgr, try_rotations)
                    batch_results.append((page_index, texts))
                except Exception as e:
                    errors.append((page_index, str(e)))
                    batch_results.append((page_index, []))
    except Exception as e:
        err_msg = f"批处理失败: {e}"
        for page_index in page_indexes:
            errors.append((page_index, err_msg))
            batch_results.append((page_index, []))

    return batch_results, errors

def extract_pdf_qr(pdf_path, model_dir, zoom, try_rotations, use_wechat, workers, progress_cb, log_cb, count_cb=None, should_stop=None):
    with fitz.open(pdf_path) as doc:
        total_pages = doc.page_count
    
    progress_cb(0, total_pages)
    log_cb(f"PDF页数: {total_pages} | 缩放: {zoom} | 旋转: {try_rotations} | WeChat: {use_wechat} | 进程: {workers}")
    
    pages = list(range(1, total_pages + 1))
    
    all_results = {}
    seen_texts = set()
    error_count = 0
    error_samples = []
    done = 0

    def _is_cancelled():
        return bool(should_stop and should_stop())
    
    if workers <= 1:
        _init_detectors(str(model_dir), use_wechat)
        try:
            with fitz.open(pdf_path) as doc:
                for page_idx in pages:
                    if _is_cancelled():
                        log_cb("检测到停止请求：已中止后续页面处理")
                        break
                    try:
                        page = doc.load_page(page_idx - 1)
                        bgr = _render_page_to_bgr(page, zoom=zoom)
                        texts = _process_page_image((_opencv_detector, _wechat_detector), bgr, try_rotations)
                    except Exception as e:
                        texts = []
                        error_count += 1
                        if len(error_samples) < 20:
                            error_samples.append((page_idx, str(e)))
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
        except Exception as e:
            raise RuntimeError(f"单进程处理失败: {e}") from e
    else:
        batch_size = max(4, min(32, total_pages // max(workers * 4, 1) + 1))
        max_inflight = max(workers * 2, 1)
        page_batches = list(_chunked(pages, batch_size))

        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_detectors,
            initargs=(str(model_dir), use_wechat),
        ) as executor:
            pending = {}
            batch_idx = 0
            cancelled = False

            while batch_idx < len(page_batches) and len(pending) < max_inflight:
                page_batch = page_batches[batch_idx]
                task = (str(pdf_path), page_batch, zoom, try_rotations, str(model_dir), use_wechat)
                future = executor.submit(_process_page_batch, task)
                pending[future] = page_batch
                batch_idx += 1

            while pending:
                if _is_cancelled() and not cancelled:
                    cancelled = True
                    log_cb("检测到停止请求：停止提交新任务，并取消未开始批次")
                    for f in list(pending.keys()):
                        if f.cancel():
                            pages_cancelled = len(pending[f])
                            done += pages_cancelled
                            progress_cb(done, total_pages)
                            del pending[f]

                done_futures, _ = wait(list(pending.keys()), timeout=0.2, return_when=FIRST_COMPLETED)
                if not done_futures:
                    continue
                for future in done_futures:
                    page_batch = pending.pop(future, [])
                    if future.cancelled():
                        continue
                    try:
                        batch_results, batch_errors = future.result()
                        if batch_errors:
                            error_count += len(batch_errors)
                            for err in batch_errors:
                                if len(error_samples) < 20:
                                    error_samples.append(err)
                        for page_idx, texts in batch_results:
                            if texts:
                                all_results[page_idx] = texts
                                for t in texts:
                                    seen_texts.add(t)
                                if count_cb:
                                    count_cb(len(seen_texts))
                    except Exception as e:
                        error_count += len(page_batch)
                        for page_idx in page_batch:
                            if len(error_samples) < 20:
                                error_samples.append((page_idx, str(e)))
                    done += len(page_batch)
                    progress_cb(done, total_pages)
                    if done % 100 == 0:
                        log_cb(f"[{done}/{total_pages}] 已处理 {done} 页")

                    if not cancelled:
                        while batch_idx < len(page_batches) and len(pending) < max_inflight:
                            next_batch = page_batches[batch_idx]
                            task = (str(pdf_path), next_batch, zoom, try_rotations, str(model_dir), use_wechat)
                            new_future = executor.submit(_process_page_batch, task)
                            pending[new_future] = next_batch
                            batch_idx += 1
                if _is_cancelled() and not pending:
                    break

    if error_count > 0:
        log_cb(f"处理完成，失败页数: {error_count}")
        for page_idx, err in error_samples[:5]:
            log_cb(f"页 {page_idx} 失败: {err}")
    
    flat_results = []
    for page in sorted(all_results.keys()):
        for content in all_results[page]:
            flat_results.append({"page": page, "content": content})

    return flat_results

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("增强版二维码识别器")
        self.minsize(860, 320)
        
        self._q = queue.Queue()
        self._stop_flag = threading.Event()
        self._running = False
        self._timer_stop_flag = threading.Event()
        self._timer_thread = None
        self._run_started_at = None

        default_pdf = self._build_default_pdf_path()
        self.pdf_var = tk.StringVar(value=str(default_pdf))
        self.out_var = tk.StringVar(value=str(self._build_output_path(default_pdf)))
        self.zoom_var = tk.StringVar(value="3.0")
        self.workers_limit = _get_recommended_workers_limit()
        default_workers = min(max(1, (os.cpu_count() or 2) - 1), self.workers_limit)
        self.workers_var = tk.StringVar(value=str(default_workers))
        self.wechat_var = tk.BooleanVar(value=True)
        self.rotate_var = tk.BooleanVar(value=True)
        self.count_var = tk.StringVar(value="0")
        self.total_var = tk.StringVar(value="0")
        self.time_var = tk.StringVar(value="")
        
        self._build_ui()
        self.update_idletasks()
        self.geometry(f"{self.winfo_reqwidth()}x{self.winfo_reqheight()}")
        self.after(80, self._poll_queue)

    def _build_default_pdf_path(self) -> Path:
        if getattr(sys, 'frozen', False):
            base_dir = Path.cwd()
        else:
            base_dir = _get_base_dir()
        preferred = base_dir / "qr_doc.pdf"
        if preferred.exists():
            return preferred
        candidates = sorted(base_dir.glob("*.pdf"))
        if candidates:
            return candidates[0]
        return preferred
    
    def _build_ui(self):
        frm = ttk.Frame(self, padding=12)
        frm.pack(fill=tk.X, expand=False)
        
        row = 0
        ttk.Label(frm, text="PDF 文件").grid(row=row, column=0, sticky="w")
        ttk.Entry(frm, textvariable=self.pdf_var, width=60).grid(row=row, column=1, sticky="ew", padx=(8, 8))
        ttk.Button(frm, text="选择…", command=self._pick_pdf).grid(row=row, column=2, sticky="ew")
        
        row += 1
        ttk.Label(frm, text="输出文件（自动）").grid(row=row, column=0, sticky="w", pady=(8, 0))
        ttk.Label(frm, textvariable=self.out_var, anchor="w").grid(row=row, column=1, sticky="ew", padx=(8, 8), pady=(8, 0))
        ttk.Label(frm, text="随PDF自动生成").grid(row=row, column=2, sticky="w", pady=(8, 0))
        
        row += 1
        opts = ttk.Frame(frm)
        opts.grid(row=row, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        
        ttk.Label(opts, text="缩放倍数").grid(row=0, column=0, sticky="w")
        ttk.Entry(opts, textvariable=self.zoom_var, width=8).grid(row=0, column=1, sticky="w", padx=(6, 16))
        
        ttk.Label(opts, text="进程数").grid(row=0, column=2, sticky="w")
        ttk.Entry(opts, textvariable=self.workers_var, width=8).grid(row=0, column=3, sticky="w", padx=(6, 16))
        ttk.Label(opts, text=f"建议上限: {self.workers_limit}", foreground="gray").grid(row=0, column=4, sticky="w", padx=(0, 16))
        
        ttk.Checkbutton(opts, text="使用WeChat识别", variable=self.wechat_var).grid(row=0, column=5, sticky="w", padx=(0, 12))
        ttk.Checkbutton(opts, text="尝试旋转", variable=self.rotate_var).grid(row=0, column=6, sticky="w", padx=(0, 12))
        
        opts.columnconfigure(7, weight=1)
        
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
        if getattr(sys, 'frozen', False):
            out_root = Path.cwd()
        else:
            out_root = Path(__file__).resolve().parent
        out_dir = out_root / "output"
        return out_dir / f"{pdf_abs.stem}_result.json"

    def _pick_pdf(self):
        p = filedialog.askopenfilename(title="选择PDF文件", filetypes=[("PDF", "*.pdf"), ("All files", "*.*")])
        if p:
            pdf_path = Path(p)
            self.pdf_var.set(str(pdf_path))
            self.out_var.set(str(self._build_output_path(pdf_path)))
    
    def _set_running(self, running):
        self._running = running
        self.start_btn.configure(state=tk.DISABLED if running else tk.NORMAL)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)

    def _start_timer_thread(self):
        self._timer_stop_flag.clear()
        self._run_started_at = time.time()

        def _timer_loop():
            while not self._timer_stop_flag.is_set():
                if self._run_started_at is not None:
                    elapsed = time.time() - self._run_started_at
                    self._q.put(("time", self._format_elapsed(elapsed)))
                time.sleep(0.2)

        self._timer_thread = threading.Thread(target=_timer_loop, daemon=True)
        self._timer_thread.start()

    def _stop_timer_thread(self):
        self._timer_stop_flag.set()

    @staticmethod
    def _format_elapsed(elapsed_seconds: float) -> str:
        total = max(0, int(elapsed_seconds))
        minutes = total // 60
        seconds = total % 60
        return f"{minutes:02d}:{seconds:02d}"
    
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
        if workers > self.workers_limit:
            workers = self.workers_limit
            self.workers_var.set(str(workers))
            messagebox.showinfo("提示", f"已按建议上限自动调整为 {workers}")
        
        self._stop_flag.clear()
        self.prog.configure(maximum=100, value=0)
        self.status.configure(text="运行中…")
        self.count_var.set("0")
        self.total_var.set("0")
        self.time_var.set("")
        self._set_running(True)
        self._start_timer_thread()

        if self.wechat_var.get():
            model_dir = _get_base_dir() / "opencv_3rdparty"
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
                model_dir=(_get_base_dir() / "opencv_3rdparty"),
                zoom=zoom,
                try_rotations=rotate,
                use_wechat=use_wechat,
                workers=workers,
                progress_cb=progress_cb,
                log_cb=lambda msg: self._q.put(("status", msg)),
                count_cb=count_cb,
                should_stop=self._stop_flag.is_set,
            )
            elapsed = time.time() - t0
            lines = [str(item.get("content", "")).strip() for item in sorted(results, key=lambda x: int(x.get("page", 0)))]
            lines = [line for line in lines if line]
            
            if self._stop_flag.is_set():
                self._q.put(("status", "已停止"))
                self._q.put(("count", str(len(results))))
                self._q.put(("time", self._format_elapsed(elapsed)))
                self._q.put(("show_results", {"lines": lines, "count": len(results), "save_error": None}))
                self._q.put(("done", None))
                return

            save_error = None
            try:
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

                out_txt = out.with_name(f"{out.stem}_pages.txt")
                out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            except (PermissionError, OSError) as e:
                save_error = str(e)
                self._q.put(("status", f"完成（保存失败）: {e}"))

            if save_error is None:
                self._q.put(("status", "完成"))
            self._q.put(("count", str(len(results))))
            self._q.put(("time", self._format_elapsed(elapsed)))
            self._q.put(("show_results", {"lines": lines, "count": len(results), "save_error": save_error}))
            self._q.put(("done", None))
        
        except Exception as e:
            self._q.put(("status", f"失败: {e}"))
            self._q.put(("count", "0"))
            self._q.put(("done", None))

    def _show_results_dialog(self, lines, count, save_error=None):
        win = tk.Toplevel(self)
        win.title("识别结果")
        win.geometry("760x500")
        win.transient(self)
        win.grab_set()

        top = ttk.Frame(win, padding=(10, 10, 10, 6))
        top.pack(fill=tk.X)
        ttk.Label(top, text=f"识别数量: {count}", font=('Arial', 11, 'bold')).pack(side=tk.LEFT)
        if save_error:
            ttk.Label(top, text="（文件保存失败，可直接复制下方内容）", foreground="red").pack(side=tk.LEFT, padx=(10, 0))

        txt = ScrolledText(win, wrap=tk.WORD, font=('Consolas', 10))
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        content = "\n".join(lines)
        txt.insert("1.0", content)
        txt.focus_set()

        btns = ttk.Frame(win, padding=(10, 0, 10, 10))
        btns.pack(fill=tk.X)

        def _copy_all():
            win.clipboard_clear()
            win.clipboard_append(content)
            self.status.configure(text="结果已复制到剪贴板")

        ttk.Button(btns, text="复制全部", command=_copy_all).pack(side=tk.LEFT)
        ttk.Button(btns, text="关闭", command=win.destroy).pack(side=tk.RIGHT)
    
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
                elif typ == "show_results":
                    lines = payload.get("lines", [])
                    count = payload.get("count", 0)
                    save_error = payload.get("save_error")
                    self._show_results_dialog(lines, count, save_error)
                elif typ == "done":
                    self._stop_timer_thread()
                    if self._run_started_at is not None:
                        elapsed = time.time() - self._run_started_at
                        self.time_var.set(self._format_elapsed(elapsed))
                    self._run_started_at = None
                    self._set_running(False)
        except queue.Empty:
            pass
        finally:
            self.after(80, self._poll_queue)

def main():
    multiprocessing.freeze_support()
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    app = App()
    app.mainloop()

if __name__ == "__main__":
    main()