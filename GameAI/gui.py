import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText

import catch
import gpt
import log
from action import ActionQueue, ActionRunner
from decide import DecideRunner
from mind import HandContext, MindRunner
from quick import QuickRunner


PROJECT_DIR = Path(__file__).resolve().parent
EXPECTED_CONFIGS = [
    "L_mind",
    "R_mind",
    "L_decide",
    "R_decide",
    "L_quick",
    "R_quick",
]


class RunnerState:
    def __init__(self):
        self.workers: dict[str, threading.Thread | None] = {}
        self.stop_event = threading.Event()
        self.l_action_queue = ActionQueue()
        self.r_action_queue = ActionQueue()
        self.l_context = HandContext("L")
        self.r_context = HandContext("R")


def safe_int(s: str, default: int = 0) -> int:
    try:
        return int(float(str(s).strip()))
    except Exception:
        return default


def run_app():
    state = RunnerState()
    root = tk.Tk()
    root.title("Dual-Hand Generic Mind + Decide + Quick + Action Runner v6.0")

    w, h = 1460, 700
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    x = (screen_w - w) // 2
    y = (screen_h - h) // 2
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.resizable(False, False)

    main_frame = ttk.Frame(root, padding=10)
    main_frame.pack(fill="both", expand=True)
    main_frame.columnconfigure(0, weight=0)
    main_frame.columnconfigure(1, weight=1)
    main_frame.rowconfigure(0, weight=1)

    left_container = tk.Frame(main_frame, bd=2, relief="groove", bg="#f2f2f2")
    left_container.grid(row=0, column=0, sticky="ns", padx=(0, 10))
    left_container.configure(width=340)
    left_container.grid_propagate(False)
    for c in range(4):
        left_container.columnconfigure(c, weight=1)

    title = tk.Label(left_container, text="Setting", bg="#f2f2f2", font=("Arial", 12, "bold"))
    title.grid(row=0, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 10))

    tk.Label(left_container, text="left:", bg="#f2f2f2").grid(row=1, column=0, sticky="e", padx=(10, 6), pady=6)
    ent_left = ttk.Entry(left_container, width=10)
    ent_left.grid(row=1, column=1, sticky="w", padx=(0, 10), pady=6)

    tk.Label(left_container, text="top:", bg="#f2f2f2").grid(row=1, column=2, sticky="e", padx=(10, 6), pady=6)
    ent_top = ttk.Entry(left_container, width=10)
    ent_top.grid(row=1, column=3, sticky="w", padx=(0, 10), pady=6)

    tk.Label(left_container, text="width:", bg="#f2f2f2").grid(row=2, column=0, sticky="e", padx=(10, 6), pady=6)
    ent_w = ttk.Entry(left_container, width=10)
    ent_w.grid(row=2, column=1, sticky="w", padx=(0, 10), pady=6)

    tk.Label(left_container, text="height:", bg="#f2f2f2").grid(row=2, column=2, sticky="e", padx=(10, 6), pady=6)
    ent_h = ttk.Entry(left_container, width=10)
    ent_h.grid(row=2, column=3, sticky="w", padx=(0, 10), pady=6)

    ent_left.insert(0, "100")
    ent_top.insert(0, "80")
    ent_w.insert(0, "1710")
    ent_h.insert(0, "790")
    catch.set_position(100, 80, 1710, 790)

    right_container = ttk.Frame(main_frame)
    right_container.grid(row=0, column=1, sticky="nsew")
    right_container.columnconfigure(0, weight=1)
    right_container.columnconfigure(1, weight=1)
    right_container.rowconfigure(1, weight=1)

    ttk.Label(right_container, text="L hand series", font=("Arial", 11, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))
    ttk.Label(right_container, text="R hand series", font=("Arial", 11, "bold")).grid(row=0, column=1, sticky="w", padx=(10, 0), pady=(0, 6))

    left_log = ScrolledText(right_container, bg="white", fg="black", wrap="word")
    left_log.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
    right_log = ScrolledText(right_container, bg="white", fg="black", wrap="word")
    right_log.grid(row=1, column=1, sticky="nsew", padx=(5, 0))
    log.init(left_log, right_log)

    status_var = tk.StringVar(value="Idle")
    folder_var = tk.StringVar(value=str(gpt.get_default_profile_dir()))

    def get_region() -> tuple[int, int, int, int]:
        return (
            safe_int(ent_left.get(), 0),
            safe_int(ent_top.get(), 0),
            safe_int(ent_w.get(), 0),
            safe_int(ent_h.get(), 0),
        )

    def set_region(left: int, top: int, width: int, height: int) -> None:
        for entry, value in ((ent_left, left), (ent_top, top), (ent_w, width), (ent_h, height)):
            entry.delete(0, tk.END)
            entry.insert(0, str(int(value)))
        catch.set_position(left, top, width, height)

    def validate_folder(path_str: str) -> Path:
        path = Path(path_str).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Folder not found: {path}")
        found = False
        for name in EXPECTED_CONFIGS:
            try:
                gpt.resolve_config_path(name, profile_dir=path)
                found = True
                break
            except FileNotFoundError:
                continue
        if not found:
            raise FileNotFoundError(f"No stage config found in {path}")
        return path

    def on_preview():
        left, top, width, height = get_region()
        catch.set_position(left, top, width, height)
        log.log(f"Preview: left={left}, top={top}, width={width}, height={height}")
        threading.Thread(target=catch.catch_view, args=(left, top, width, height), daemon=True).start()

    def on_select_area():
        result = catch.select_region(get_region())
        if result is None:
            return
        set_region(*result)

    def choose_folder():
        path = filedialog.askdirectory(initialdir=folder_var.get() or str(PROJECT_DIR))
        if path:
            folder_var.set(path)

    def workers_alive() -> bool:
        return any(worker is not None and worker.is_alive() for worker in state.workers.values())

    def update_buttons() -> None:
        alive = workers_alive()
        btn_start.config(state=("disabled" if alive else "normal"))
        btn_stop.config(state=("normal" if alive else "disabled"))
        status_var.set("Running" if alive else "Idle")

    def _mark_worker_done(name: str) -> None:
        state.workers[name] = None
        root.after(0, update_buttons)

    def _make_entry(name: str, runner):
        def entry() -> None:
            try:
                runner.run()
            finally:
                _mark_worker_done(name)
        return entry

    def maybe_add_thread(name: str, runner) -> None:
        state.workers[name] = threading.Thread(target=_make_entry(name, runner), daemon=True, name=f"{name}-thread")

    def on_start():
        if workers_alive():
            return
        left, top, width, height = get_region()
        if width <= 0 or height <= 0:
            return
        try:
            profile_dir = validate_folder(folder_var.get())
        except Exception as e:
            log.log(f"[ERROR] {e}")
            return

        gpt.set_profile_dir(profile_dir)
        region = (left, top, width, height)
        origin = (left, top)
        state.stop_event = threading.Event()
        state.workers = {}
        state.l_action_queue.clear()
        state.r_action_queue.clear()
        state.l_context = HandContext("L")
        state.r_context = HandContext("R")

        hand_map = {
            "L": (state.l_context, state.l_action_queue, state.r_action_queue),
            "R": (state.r_context, state.r_action_queue, state.l_action_queue),
        }

        any_stage_enabled = False
        for hand_key, (context, primary_queue, partner_queue) in hand_map.items():
            stage_enabled = False
            for stage_name, runner_cls in (("mind", MindRunner), ("decide", DecideRunner), ("quick", QuickRunner)):
                config_name = f"{hand_key}_{stage_name}"
                try:
                    settings = gpt.get_stage_settings(config_name, profile_dir=profile_dir)
                except FileNotFoundError:
                    continue
                if not settings["enabled"]:
                    continue
                stage_enabled = True
                any_stage_enabled = True
                runner = runner_cls(
                    hand_key=hand_key,
                    profile_dir=profile_dir,
                    region=region,
                    context=context,
                    primary_queue=primary_queue,
                    partner_queue=partner_queue,
                    stop_event=state.stop_event,
                    log_fn=log.log,
                )
                maybe_add_thread(f"{hand_key}_{stage_name}", runner)

            if stage_enabled:
                action_runner = ActionRunner(
                    hand_key=hand_key,
                    action_queue=primary_queue,
                    stop_event=state.stop_event,
                    origin=origin,
                    execute_enabled=True,
                    log_fn=log.log,
                )
                maybe_add_thread(f"{hand_key}_action", action_runner)

        if not any_stage_enabled:
            log.log("[ERROR] no enabled stage found")
            return

        for worker in state.workers.values():
            if worker is not None:
                worker.start()
        update_buttons()

    def on_stop():
        state.stop_event.set()
        state.l_action_queue.clear()
        state.r_action_queue.clear()
        update_buttons()

    def on_close():
        state.stop_event.set()
        state.l_action_queue.clear()
        state.r_action_queue.clear()
        root.destroy()

    ttk.Button(left_container, text="Preview", command=on_preview).grid(row=3, column=0, columnspan=4, pady=(10, 6), padx=10, sticky="ew")
    ttk.Button(left_container, text="Select Area", command=on_select_area).grid(row=4, column=0, columnspan=4, pady=(0, 6), padx=10, sticky="ew")

    tk.Label(left_container, text="Dataset Folder:", bg="#f2f2f2").grid(row=5, column=0, columnspan=4, sticky="w", padx=10, pady=(10, 4))
    ttk.Button(left_container, text="Browse", command=choose_folder).grid(row=6, column=0, columnspan=4, pady=(0, 6), padx=10, sticky="ew")
    tk.Label(left_container, textvariable=folder_var, bg="#f2f2f2", anchor="w", justify="left", wraplength=300).grid(row=7, column=0, columnspan=4, padx=10, pady=(0, 10), sticky="ew")

    btn_start = ttk.Button(left_container, text="Start", command=on_start)
    btn_start.grid(row=8, column=0, columnspan=2, pady=(10, 6), padx=(10, 5), sticky="ew")
    btn_stop = ttk.Button(left_container, text="Stop", command=on_stop, state="disabled")
    btn_stop.grid(row=8, column=2, columnspan=2, pady=(10, 6), padx=(5, 10), sticky="ew")

    tk.Label(left_container, text="Status:", bg="#f2f2f2").grid(row=9, column=0, sticky="w", padx=10, pady=(4, 10))
    tk.Label(left_container, textvariable=status_var, bg="#f2f2f2", fg="#005a9c", font=("Arial", 10, "bold")).grid(row=9, column=1, columnspan=3, sticky="w", padx=(0, 10), pady=(4, 10))

    root.protocol("WM_DELETE_WINDOW", on_close)
    update_buttons()
    root.mainloop()
