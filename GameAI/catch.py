import tkinter as tk
import time

POSITION = [100, 200, 300, 400]  # [left, top, width, height]


def set_position(left: int, top: int, width: int, height: int) -> None:
    POSITION[:] = [int(left), int(top), int(width), int(height)]


def catch_view(left=None, top=None, width=None, height=None):
    if left is None or top is None or width is None or height is None:
        left, top, width, height = POSITION

    root = tk.Tk()
    root.withdraw()

    overlay = tk.Toplevel(root)
    overlay.overrideredirect(True)
    overlay.attributes("-topmost", True)

    sw = overlay.winfo_screenwidth()
    sh = overlay.winfo_screenheight()
    overlay.geometry(f"{sw}x{sh}+0+0")

    transparent_color = "magenta"
    overlay.configure(bg=transparent_color)

    try:
        overlay.attributes("-transparentcolor", transparent_color)
    except tk.TclError:
        overlay.attributes("-alpha", 0.25)

    canvas = tk.Canvas(overlay, bg=transparent_color, highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    x1, y1 = int(left), int(top)
    x2, y2 = x1 + int(width), y1 + int(height)
    canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=3, dash=(8, 6))

    root.update_idletasks()
    root.update()
    time.sleep(2)
    root.destroy()


def select_region(initial=None):
    result = {"rect": None}
    start = {"x": 0, "y": 0}
    temp = {"rect_id": None, "label_id": None}

    root = tk.Tk()
    root.attributes("-fullscreen", True)
    root.attributes("-topmost", True)
    root.configure(bg="black")
    try:
        root.attributes("-alpha", 0.25)
    except tk.TclError:
        pass

    canvas = tk.Canvas(root, bg="black", highlightthickness=0, cursor="cross")
    canvas.pack(fill="both", expand=True)
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()

    hint = (
        "滑鼠拖曳框選範圍，放開後按 Enter 確認，按 Esc 取消。"
        "\n如果要重選，直接再拖一次即可。"
    )
    canvas.create_text(
        20,
        20,
        anchor="nw",
        text=hint,
        fill="white",
        font=("Arial", 16, "bold"),
    )

    if initial and len(initial) == 4:
        left, top, width, height = [int(v) for v in initial]
        temp["rect_id"] = canvas.create_rectangle(
            left,
            top,
            left + width,
            top + height,
            outline="red",
            width=3,
        )
        temp["label_id"] = canvas.create_text(
            left,
            max(20, top - 10),
            anchor="sw",
            text=f"{width} x {height}",
            fill="yellow",
            font=("Arial", 14, "bold"),
        )
        result["rect"] = (left, top, width, height)

    def update_rect(x1, y1, x2, y2):
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        if temp["rect_id"] is None:
            temp["rect_id"] = canvas.create_rectangle(left, top, left + width, top + height, outline="red", width=3)
        else:
            canvas.coords(temp["rect_id"], left, top, left + width, top + height)

        label_y = max(20, top - 10)
        if temp["label_id"] is None:
            temp["label_id"] = canvas.create_text(
                left,
                label_y,
                anchor="sw",
                text=f"{width} x {height}",
                fill="yellow",
                font=("Arial", 14, "bold"),
            )
        else:
            canvas.coords(temp["label_id"], left, label_y)
            canvas.itemconfig(temp["label_id"], text=f"{width} x {height}")

        result["rect"] = (left, top, width, height)

    def on_press(event):
        start["x"] = event.x
        start["y"] = event.y
        update_rect(event.x, event.y, event.x, event.y)

    def on_drag(event):
        update_rect(start["x"], start["y"], event.x, event.y)

    def on_release(event):
        update_rect(start["x"], start["y"], event.x, event.y)

    def confirm(_event=None):
        rect = result["rect"]
        if not rect:
            return
        left, top, width, height = rect
        if width <= 0 or height <= 0:
            return
        set_position(left, top, width, height)
        root.quit()

    def cancel(_event=None):
        result["rect"] = None
        root.quit()

    canvas.bind("<ButtonPress-1>", on_press)
    canvas.bind("<B1-Motion>", on_drag)
    canvas.bind("<ButtonRelease-1>", on_release)
    root.bind("<Return>", confirm)
    root.bind("<Escape>", cancel)
    root.protocol("WM_DELETE_WINDOW", cancel)

    root.mainloop()
    root.destroy()
    return result["rect"]


if __name__ == "__main__":
    print(select_region())
