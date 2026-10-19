from __future__ import annotations

import json
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np

from .audio_io import play_audio, stop_audio, write_wav
from .debug_audio import write_debug_bundle
from .frontend import MandarinFrontend
from .gesture_planner import PlannerConfig, plan_gestures
from .models import GestureTimeline, PhonemeToken
from .optional_backends import component_status, export_vtl_bridge_payload, probe_vocaltractlab
from .synthesizer import synthesize_timeline
from .voice_profile import VoiceProfile, captured_reference_settings, waveform_preset
from .widgets import TimelineCanvas, WaveformEditor


APP_TITLE = "DigiVocal 數位發聲構造實驗室 v0.1"


class DigiVocalApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("1440x960")
        self.root.minsize(1040, 700)

        self.tokens: list[PhonemeToken] = []
        self.timeline: GestureTimeline | None = None
        self.audio: np.ndarray | None = None
        self._busy = False

        default_profile = VoiceProfile.wave_analysis_clone(-5.5)
        self.backend_var = tk.StringVar(value="pypinyin")
        self.tone_sandhi_var = tk.BooleanVar(value=True)
        self.rate_var = tk.DoubleVar(value=3.7)
        self.base_f0_var = tk.DoubleVar(value=default_profile.base_f0_hz)
        self.f0_range_var = tk.DoubleVar(value=default_profile.f0_range_semitones)
        self.formant_scale_var = tk.DoubleVar(value=default_profile.formant_scale)
        self.f1_scale_var = tk.DoubleVar(value=default_profile.f1_scale)
        self.f2_scale_var = tk.DoubleVar(value=default_profile.f2_scale)
        self.f3_scale_var = tk.DoubleVar(value=default_profile.f3_scale)
        self.f1_gain_var = tk.DoubleVar(value=default_profile.f1_gain)
        self.f2_gain_var = tk.DoubleVar(value=default_profile.f2_gain)
        self.f3_gain_var = tk.DoubleVar(value=default_profile.f3_gain)
        self.f1_bandwidth_var = tk.DoubleVar(value=default_profile.f1_bandwidth_scale)
        self.f2_bandwidth_var = tk.DoubleVar(value=default_profile.f2_bandwidth_scale)
        self.f3_bandwidth_var = tk.DoubleVar(value=default_profile.f3_bandwidth_scale)
        self.output_gain_var = tk.DoubleVar(value=default_profile.output_gain * 100.0)
        self.breath_var = tk.DoubleVar(value=default_profile.breathiness * 100.0)
        self.jitter_var = tk.DoubleVar(value=default_profile.jitter_percent)
        self.shimmer_var = tk.DoubleVar(value=default_profile.shimmer_percent)
        self.tilt_var = tk.DoubleVar(value=default_profile.spectral_tilt_db_per_octave)
        self.cycle_morph_var = tk.DoubleVar(value=default_profile.cycle_morph_percent)
        self.cleanup_strength_var = tk.DoubleVar(
            value=default_profile.source_residual_cleanup_strength * 100.0
        )
        self.source_preservation_var = tk.BooleanVar(
            value=default_profile.source_preservation
        )
        self.vowel_source_mix_var = tk.DoubleVar(
            value=default_profile.vowel_source_mix * 100.0
        )
        self.source_direct_mix_var = tk.DoubleVar(
            value=default_profile.source_direct_mix * 100.0
        )
        self.profile_name_var = tk.StringVar(value=default_profile.name)
        self.source_info_var = tk.StringVar(value="")
        self._profile_template = default_profile
        self._wave_dirty = False
        self.compute_backend_var = tk.StringVar(value="auto")
        self.precision_guard_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(
            value="輸入中文後先按『分析動作』；輸出是研究用構音合成，會比神經 TTS 更像電子聲。"
        )

        self._build_style()
        self._build_ui(default_profile)

    def _build_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 18, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Microsoft JhengHei UI", 11, "bold"))
        style.configure("Treeview", rowheight=24)

    def _build_ui(self, profile: VoiceProfile) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 8))
        ttk.Label(header, text=APP_TITLE, style="Title.TLabel").pack(side="left")
        ttk.Label(
            header,
            text="中文文字 → 拼音/聲調 → 發聲構造時間軸 → 聲源濾波合成",
        ).pack(side="right", pady=(8, 0))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)

        synth_tab = ttk.Frame(notebook, padding=8)
        profile_tab = ttk.Frame(notebook, padding=8)
        components_tab = ttk.Frame(notebook, padding=8)
        notebook.add(synth_tab, text="文字與發聲動作")
        notebook.add(profile_tab, text="基調聲源／聲紋輪廓")
        notebook.add(components_tab, text="開源元件")

        self._build_synth_tab(synth_tab)
        self._build_profile_tab(profile_tab, profile)
        self._build_components_tab(components_tab)

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(8, 0))
        ttk.Label(footer, textvariable=self.status_var).pack(side="left", fill="x", expand=True)
        ttk.Button(footer, text="停止聲音", command=self.stop).pack(side="right")

    def _build_synth_tab(self, master: ttk.Frame) -> None:
        text_frame = ttk.LabelFrame(
            master,
            text="1. 中文文字",
            style="Section.TLabelframe",
            padding=8,
        )
        text_frame.pack(fill="x")
        self.text_input = tk.Text(
            text_frame,
            height=4,
            wrap="word",
            font=("Microsoft JhengHei UI", 12),
        )
        self.text_input.pack(fill="x", expand=True)
        self.text_input.insert(
            "1.0",
            "你好，我是數位聲音。今天想測試聲調、嘴型與呼吸的變化。",
        )

        controls = ttk.Frame(text_frame)
        controls.pack(fill="x", pady=(8, 0))
        ttk.Label(controls, text="讀音後端").pack(side="left")
        ttk.Combobox(
            controls,
            textvariable=self.backend_var,
            values=("pypinyin", "auto", "g2pw"),
            state="readonly",
            width=11,
        ).pack(side="left", padx=(5, 12))
        ttk.Checkbutton(
            controls,
            text="套用變調（pypinyin）",
            variable=self.tone_sandhi_var,
        ).pack(side="left")
        ttk.Label(controls, text="每秒音節").pack(side="left", padx=(18, 5))
        ttk.Spinbox(
            controls,
            from_=1.0,
            to=10.0,
            increment=0.1,
            textvariable=self.rate_var,
            width=6,
        ).pack(side="left")
        ttk.Label(controls, text="精算後端").pack(side="left", padx=(14, 5))
        ttk.Combobox(
            controls,
            textvariable=self.compute_backend_var,
            values=("auto", "gpu", "cpu"),
            state="readonly",
            width=6,
        ).pack(side="left")
        ttk.Checkbutton(
            controls,
            text="精算品質最佳化",
            variable=self.precision_guard_var,
        ).pack(side="left", padx=(8, 0))

        self.analyze_button = ttk.Button(controls, text="分析動作", command=self.analyze)
        self.analyze_button.pack(side="right", padx=(5, 0))
        self.render_button = ttk.Button(controls, text="合成並播放", command=self.render_and_play)
        self.render_button.pack(side="right", padx=(5, 0))
        ttk.Button(controls, text="只合成", command=self.render_only).pack(side="right", padx=(5, 0))

        token_frame = ttk.LabelFrame(
            master,
            text="2. 拼音、音素與時間",
            style="Section.TLabelframe",
            padding=6,
        )
        token_frame.pack(fill="x", pady=(8, 0))
        columns = ("text", "pinyin", "initial", "final", "tone", "start", "end")
        self.token_tree = ttk.Treeview(token_frame, columns=columns, show="headings", height=8)
        headings = {
            "text": "文字",
            "pinyin": "拼音",
            "initial": "聲母",
            "final": "韻母",
            "tone": "聲調",
            "start": "開始 s",
            "end": "結束 s",
        }
        widths = {"text": 80, "pinyin": 110, "initial": 75, "final": 90, "tone": 60, "start": 85, "end": 85}
        for column in columns:
            self.token_tree.heading(column, text=headings[column])
            self.token_tree.column(column, width=widths[column], anchor="center")
        scrollbar = ttk.Scrollbar(token_frame, orient="vertical", command=self.token_tree.yview)
        self.token_tree.configure(yscrollcommand=scrollbar.set)
        self.token_tree.pack(side="left", fill="x", expand=True)
        scrollbar.pack(side="right", fill="y")

        timeline_frame = ttk.LabelFrame(
            master,
            text="3. 發聲構造時間軸",
            style="Section.TLabelframe",
            padding=6,
        )
        timeline_frame.pack(fill="both", expand=True, pady=(8, 0))
        self.timeline_canvas = TimelineCanvas(timeline_frame, height=280)
        self.timeline_canvas.pack(fill="both", expand=True)

        output_row = ttk.Frame(master)
        output_row.pack(fill="x", pady=(8, 0))
        ttk.Button(output_row, text="匯出 WAV", command=self.export_wav).pack(side="left")
        ttk.Button(output_row, text="匯出動作 JSON", command=self.export_gestures).pack(side="left", padx=6)
        ttk.Button(output_row, text="匯出 VTL 橋接 JSON", command=self.export_vtl_bridge).pack(side="left")
        self.debug_button = ttk.Button(
            output_row,
            text="合成 + 爆音 Debug",
            command=self.render_debug,
        )
        self.debug_button.pack(side="left", padx=(12, 0))
        ttk.Label(
            output_row,
            text="VTL 橋接檔不是 .ges；仍需依 speaker model 校正參數映射。",
        ).pack(side="right")

    def _build_profile_tab(self, master: ttk.Frame, profile: VoiceProfile) -> None:
        left = ttk.LabelFrame(
            master,
            text="1. 高解析聲源週期／Wave Bank",
            style="Section.TLabelframe",
            padding=8,
        )
        left.pack(side="left", fill="both", expand=True, padx=(0, 8))

        preset_row = ttk.Frame(left)
        preset_row.pack(fill="x", pady=(0, 6))
        for label, key in (
            ("理想聲線 -5.5", "wave_analysis_bright"),
            ("理想聲線 -7.0", "wave_analysis_balanced"),
            ("柔和聲門", "soft_glottal"),
            ("正弦", "sine"),
            ("三角", "triangle"),
            ("鋸齒", "saw"),
            ("緊聲", "pressed"),
            ("氣聲", "breathy"),
        ):
            ttk.Button(
                preset_row,
                text=label,
                command=lambda value=key: self.set_wave_preset(value),
            ).pack(side="left", padx=2)

        self.wave_editor = WaveformEditor(
            left,
            profile.wave_points,
            on_change=self._mark_wave_dirty,
            height=420,
        )
        self.wave_editor.pack(fill="both", expand=True)

        ttk.Label(
            left,
            textvariable=self.source_info_var,
            wraplength=760,
        ).pack(fill="x", pady=(8, 0))

        right = ttk.LabelFrame(
            master,
            text="2. 聲線輪廓",
            style="Section.TLabelframe",
            padding=8,
        )
        right.pack(side="right", fill="y")

        settings = ttk.Notebook(right)
        settings.pack(fill="both", expand=True)
        basic = ttk.Frame(settings, padding=8)
        formants = ttk.Frame(settings, padding=8)
        source = ttk.Frame(settings, padding=8)
        settings.add(basic, text="基本")
        settings.add(formants, text="F1/F2/F3")
        settings.add(source, text="聲源動態")

        name_row = ttk.Frame(basic)
        name_row.pack(fill="x", pady=(0, 6))
        ttk.Label(name_row, text="名稱").pack(side="left")
        ttk.Entry(name_row, textvariable=self.profile_name_var, width=22).pack(side="right")
        self._add_scale(basic, "基調 F0 (Hz)", self.base_f0_var, 50.0, 700.0)
        self._add_scale(basic, "聲調音域 (半音)", self.f0_range_var, 0.0, 24.0)
        self._add_scale(basic, "輸出音量／峰值上限 (%)", self.output_gain_var, 5.0, 90.0)
        self._add_scale(basic, "氣聲比例 (%)", self.breath_var, 0.0, 70.0)
        self._add_scale(basic, "Jitter (%)", self.jitter_var, 0.0, 3.0)
        self._add_scale(basic, "Shimmer (%)", self.shimmer_var, 0.0, 10.0)
        self._add_scale(basic, "頻譜傾斜 dB/oct", self.tilt_var, -18.0, 2.0)

        self._add_scale(formants, "整體共振峰倍率", self.formant_scale_var, 0.65, 1.50)
        self._add_scale(formants, "F1 頻率倍率", self.f1_scale_var, 0.75, 1.35)
        self._add_scale(formants, "F2 頻率倍率", self.f2_scale_var, 0.75, 1.40)
        self._add_scale(formants, "F3 頻率倍率", self.f3_scale_var, 0.75, 1.45)
        self._add_scale(formants, "F1 能量倍率", self.f1_gain_var, 0.20, 2.00)
        self._add_scale(formants, "F2 能量倍率", self.f2_gain_var, 0.20, 2.00)
        self._add_scale(formants, "F3 能量倍率", self.f3_gain_var, 0.20, 2.50)
        self._add_scale(formants, "F1 頻寬倍率", self.f1_bandwidth_var, 0.55, 1.60)
        self._add_scale(formants, "F2 頻寬倍率", self.f2_bandwidth_var, 0.55, 1.60)
        self._add_scale(formants, "F3 頻寬倍率", self.f3_bandwidth_var, 0.55, 1.60)
        ttk.Label(
            formants,
            text="目前理想聲線預設：F1 稍弱、F2/F3 提高並上移，用來減少厚、圓、含糊感。",
            wraplength=310,
        ).pack(fill="x", pady=(10, 0))

        ttk.Checkbutton(
            source,
            text="Digital Throat Source 保真模式",
            variable=self.source_preservation_var,
        ).pack(fill="x", pady=(0, 6))
        self._add_scale(source, "逐週期 Wave Bank morph (%)", self.cycle_morph_var, 0.0, 60.0)
        self._add_scale(source, "母音 Source 差異混合 (%)", self.vowel_source_mix_var, 0.0, 80.0)
        self._add_scale(source, "聲道直通諧波 (%)", self.source_direct_mix_var, 0.0, 90.0)
        self._add_scale(source, "Residual 清理強度 (%)", self.cleanup_strength_var, 0.0, 100.0)
        ttk.Button(
            source,
            text="套用理想聲線 -5.5（亮／像）",
            command=lambda: self.set_target_voice(-5.5),
        ).pack(fill="x", pady=(12, 3))
        ttk.Button(
            source,
            text="套用理想聲線 -7.0（較平衡）",
            command=lambda: self.set_target_voice(-7.0),
        ).pack(fill="x", pady=3)
        ttk.Label(
            source,
            text=(
                "Digital Throat 模式會保留擷取 Source 的相位與頻譜傾斜，依目前韻母選擇 "
                "vowel_a/e/i/o/u，並輸出逐層 Source Debug WAV。Residual 清理只供舊式／手動畫波形使用。"
            ),
            wraplength=310,
        ).pack(fill="x", pady=(10, 0))

        ttk.Separator(right).pack(fill="x", pady=8)
        ttk.Button(right, text="儲存聲音輪廓 JSON", command=self.save_profile).pack(fill="x", pady=3)
        ttk.Button(right, text="載入聲音輪廓 JSON", command=self.load_profile).pack(fill="x", pady=3)
        ttk.Button(right, text="用目前輪廓重新合成", command=self.render_and_play).pack(
            fill="x", pady=(10, 3)
        )

        self._update_source_info(profile)

    def _mark_wave_dirty(self) -> None:
        self._wave_dirty = True
        self.source_info_var.set(
            f"已手動修改 {len(self.wave_editor.values)}-point 主表；下次合成會由它重建 Wave Bank。"
        )

    def _update_source_info(self, profile: VoiceProfile) -> None:
        metadata = profile.source_metadata or {}
        cleanup = metadata.get("residual_cleanup", {})
        leakage = cleanup.get("leakage_score_after_db")
        leakage_text = f"，清理後 residual score {float(leakage):.2f} dB" if leakage is not None else ""
        vowel_banks = [
            name for name in profile.wave_bank_names
            if name in {"vowel_a", "vowel_e", "vowel_i", "vowel_o", "vowel_u"}
        ]
        schema = str(
            metadata.get("origin_profile_schema")
            or metadata.get("profile_schema")
            or "unknown"
        )
        mode = "Digital Throat 保真" if profile.is_digital_throat_profile else "DigiVocal 傳統"
        vowel_text = "/".join(name.removeprefix("vowel_") for name in vowel_banks) or "無"
        self.source_info_var.set(
            f"{mode}｜schema={schema}｜原生週期 {len(profile.native_cycle)} samples → "
            f"主表 {len(profile.wave_points)} points；Wave Bank {len(profile.wave_bank)} 組；"
            f"母音 Bank {vowel_text}；母音混合 {profile.vowel_source_mix * 100.0:.1f}%；"
            f"直通諧波 {profile.source_direct_mix * 100.0:.1f}%；"
            f"逐週期 morph {profile.cycle_morph_percent:.1f}%{leakage_text}。"
        )

    def _apply_profile_to_ui(self, profile: VoiceProfile, *, status: str | None = None) -> None:
        profile.validate()
        self._profile_template = profile
        self.wave_editor.set_values(profile.wave_points)
        self.profile_name_var.set(profile.name)
        self.base_f0_var.set(profile.base_f0_hz)
        self.f0_range_var.set(profile.f0_range_semitones)
        self.formant_scale_var.set(profile.formant_scale)
        self.f1_scale_var.set(profile.f1_scale)
        self.f2_scale_var.set(profile.f2_scale)
        self.f3_scale_var.set(profile.f3_scale)
        self.f1_gain_var.set(profile.f1_gain)
        self.f2_gain_var.set(profile.f2_gain)
        self.f3_gain_var.set(profile.f3_gain)
        self.f1_bandwidth_var.set(profile.f1_bandwidth_scale)
        self.f2_bandwidth_var.set(profile.f2_bandwidth_scale)
        self.f3_bandwidth_var.set(profile.f3_bandwidth_scale)
        self.output_gain_var.set(profile.output_gain * 100.0)
        self.breath_var.set(profile.breathiness * 100.0)
        self.jitter_var.set(profile.jitter_percent)
        self.shimmer_var.set(profile.shimmer_percent)
        self.tilt_var.set(profile.spectral_tilt_db_per_octave)
        self.cycle_morph_var.set(profile.cycle_morph_percent)
        self.cleanup_strength_var.set(profile.source_residual_cleanup_strength * 100.0)
        self.source_preservation_var.set(profile.source_preservation)
        self.vowel_source_mix_var.set(profile.vowel_source_mix * 100.0)
        self.source_direct_mix_var.set(profile.source_direct_mix * 100.0)
        self._wave_dirty = False
        self._update_source_info(profile)
        if status:
            self.status_var.set(status)

    def set_target_voice(self, tilt: float) -> None:
        profile = VoiceProfile.wave_analysis_clone(tilt)
        label = "亮／像" if tilt > -6.2 else "較平衡"
        self._apply_profile_to_ui(
            profile,
            status=f"已套用 wave_analysis 理想聲線：tilt {tilt:.1f} dB/oct（{label}）。",
        )

    def _build_components_tab(self, master: ttk.Frame) -> None:
        toolbar = ttk.Frame(master)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="重新檢查", command=self.refresh_components).pack(side="left")
        ttk.Button(toolbar, text="查看 VTL API 探測", command=self.show_vtl_probe).pack(side="left", padx=6)
        ttk.Label(
            toolbar,
            text="第三方程式碼沒有直接包進 ZIP；可用 scripts/fetch_open_source_components.py 下載。",
        ).pack(side="right")

        self.components_text = tk.Text(
            master,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
        )
        self.components_text.pack(fill="both", expand=True, pady=(8, 0))
        self.refresh_components()

    def _add_scale(
        self,
        master: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        start: float,
        end: float,
    ) -> None:
        row = ttk.Frame(master)
        row.pack(fill="x", pady=(7, 0))
        ttk.Label(row, text=label).pack(side="left")
        value_label = ttk.Label(row, width=8, anchor="e")
        value_label.pack(side="right")

        def update(_value=None) -> None:
            value_label.configure(text=f"{variable.get():.2f}")

        ttk.Scale(master, from_=start, to=end, variable=variable, command=update).pack(fill="x")
        variable.trace_add("write", lambda *_args: update())
        update()

    def current_profile(self) -> VoiceProfile:
        template = self._profile_template.to_dict()
        previous_cleanup = float(template.get("source_residual_cleanup_strength", 0.0))
        preservation_requested = bool(self.source_preservation_var.get())
        requested_cleanup = (
            0.0
            if preservation_requested
            else float(self.cleanup_strength_var.get()) / 100.0
        )
        template.update(
            {
                "name": self.profile_name_var.get().strip() or "unnamed",
                "sample_rate": 48_000,
                "base_f0_hz": float(self.base_f0_var.get()),
                "f0_range_semitones": float(self.f0_range_var.get()),
                "formant_scale": float(self.formant_scale_var.get()),
                "f1_scale": float(self.f1_scale_var.get()),
                "f2_scale": float(self.f2_scale_var.get()),
                "f3_scale": float(self.f3_scale_var.get()),
                "f1_gain": float(self.f1_gain_var.get()),
                "f2_gain": float(self.f2_gain_var.get()),
                "f3_gain": float(self.f3_gain_var.get()),
                "f1_bandwidth_scale": float(self.f1_bandwidth_var.get()),
                "f2_bandwidth_scale": float(self.f2_bandwidth_var.get()),
                "f3_bandwidth_scale": float(self.f3_bandwidth_var.get()),
                "output_gain": float(self.output_gain_var.get()) / 100.0,
                "breathiness": float(self.breath_var.get()) / 100.0,
                "jitter_percent": float(self.jitter_var.get()),
                "shimmer_percent": float(self.shimmer_var.get()),
                "spectral_tilt_db_per_octave": float(self.tilt_var.get()),
                "cycle_morph_percent": float(self.cycle_morph_var.get()),
                "source_residual_cleanup_strength": requested_cleanup,
                "source_preservation": preservation_requested,
                "vowel_source_mix": float(self.vowel_source_mix_var.get()) / 100.0,
                "source_direct_mix": float(self.source_direct_mix_var.get()) / 100.0,
                "source_table_points": len(self.wave_editor.values),
                "wave_points": [float(value) for value in self.wave_editor.values],
            }
        )
        cleanup_changed = abs(requested_cleanup - previous_cleanup) > 1e-6
        if self._wave_dirty:
            template["native_cycle"] = list(template["wave_points"])
            template["wave_bank"] = []
            template["wave_bank_names"] = []
            template["wave_bank_base_weights"] = []
            template["source_preservation"] = False
            template["vowel_source_mix"] = 0.0
            template["source_direct_mix"] = 0.0
            metadata = dict(template.get("source_metadata") or {})
            metadata.pop("origin_profile_schema", None)
            metadata["profile_schema"] = "digivocal-voice-profile-v2"
            metadata["wave_points_are_cleaned"] = True
            metadata["manual_wave_edit"] = True
            metadata["reference"] = "manual_wave_edit"
            template["source_metadata"] = metadata
        elif cleanup_changed:
            # Re-run residual de-coloring from the preserved native cycle rather
            # than repeatedly processing the already-cleaned 512-point table.
            template["wave_points"] = list(template.get("native_cycle") or template["wave_points"])
            template["wave_bank"] = []
            template["wave_bank_names"] = []
            template["wave_bank_base_weights"] = []
            metadata = dict(template.get("source_metadata") or {})
            metadata["wave_points_are_cleaned"] = False
            metadata["cleanup_recomputed_from_native"] = True
            template["source_metadata"] = metadata
        profile = VoiceProfile.from_dict(template)
        self._profile_template = profile
        self._wave_dirty = False
        self._update_source_info(profile)
        return profile

    def current_config(self) -> PlannerConfig:
        return PlannerConfig(syllables_per_second=float(self.rate_var.get()))

    def set_wave_preset(self, name: str) -> None:
        key = name.strip().lower()
        if key in {"captured", "captured_reference", "wave_analysis", "wave_analysis_bright", "截取聲線", "理想聲線"}:
            self.set_target_voice(-5.5)
            return
        if key in {"wave_analysis_balanced", "理想聲線-7"}:
            self.set_target_voice(-7.0)
            return
        points = waveform_preset(name, len(self.wave_editor.values))
        self.wave_editor.set_values(points)
        self._wave_dirty = True
        self.status_var.set(f"已套用 {len(points)}-point 聲源波形：{name}；Wave Bank 將自動重建。")

    def _set_busy(self, busy: bool, text: str | None = None) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        self.analyze_button.configure(state=state)
        self.render_button.configure(state=state)
        self.debug_button.configure(state=state)
        if text:
            self.status_var.set(text)

    def _run_async(self, busy_text: str, worker, done) -> None:
        if self._busy:
            return
        self._set_busy(True, busy_text)

        def target() -> None:
            try:
                result = worker()
            except Exception as exc:
                self.root.after(0, lambda: self._async_error(exc))
                return
            self.root.after(0, lambda: self._async_done(result, done))

        threading.Thread(target=target, daemon=True).start()

    def _async_error(self, exc: Exception) -> None:
        self._set_busy(False)
        messagebox.showerror("處理失敗", str(exc), parent=self.root)
        self.status_var.set(f"失敗：{exc}")

    def _async_done(self, result, callback) -> None:
        self._set_busy(False)
        callback(result)

    def _capture_request(self):
        """Read all Tk state on the main thread before starting a worker."""
        return {
            "text": self.text_input.get("1.0", "end").strip(),
            "profile": self.current_profile(),
            "config": self.current_config(),
            "backend": self.backend_var.get(),
            "tone_sandhi": bool(self.tone_sandhi_var.get()),
            "compute_backend": self.compute_backend_var.get(),
            "precision_guard": bool(self.precision_guard_var.get()),
        }

    @staticmethod
    def _analyze_request(request):
        text = request["text"]
        profile = request["profile"]
        frontend = MandarinFrontend(
            request["backend"],
            tone_sandhi=request["tone_sandhi"],
        )
        result = frontend.convert(text)
        timeline = plan_gestures(
            result.tokens,
            profile,
            request["config"],
            metadata={"text": text, "frontend": result.backend_used},
        )
        return result, timeline, profile

    def analyze(self) -> None:
        try:
            request = self._capture_request()
        except Exception as exc:
            messagebox.showerror("參數錯誤", str(exc), parent=self.root)
            return
        self._run_async(
            "正在把中文轉成拼音與發聲構造……",
            lambda: self._analyze_request(request),
            self._accept_analysis,
        )

    def _accept_analysis(self, payload) -> None:
        result, timeline, _profile = payload
        self.tokens = result.tokens
        self.timeline = timeline
        self.audio = None
        self._update_token_tree()
        self.timeline_canvas.set_timeline(timeline)
        suffix = f"；{result.warning}" if result.warning else ""
        self.status_var.set(
            f"分析完成：{len(result.tokens)} 個文字單位，{timeline.duration_s:.2f} 秒，後端={result.backend_used}{suffix}"
        )

    @classmethod
    def _render_request(cls, request):
        result, timeline, profile = cls._analyze_request(request)
        guard_report: dict[str, object] = {}
        audio = synthesize_timeline(
            timeline,
            profile,
            compute_backend=request["compute_backend"],
            precision_guard=request["precision_guard"],
            precision_guard_report=guard_report,
        )
        return result, timeline, profile, audio, guard_report

    def render_only(self) -> None:
        try:
            request = self._capture_request()
        except Exception as exc:
            messagebox.showerror("參數錯誤", str(exc), parent=self.root)
            return
        self._run_async(
            "正在產生聲音波形……",
            lambda: self._render_request(request),
            self._accept_render,
        )

    def render_and_play(self) -> None:
        try:
            request = self._capture_request()
        except Exception as exc:
            messagebox.showerror("參數錯誤", str(exc), parent=self.root)
            return
        self._run_async(
            "正在產生聲音波形……",
            lambda: self._render_request(request),
            self._accept_render_and_play,
        )

    def _accept_render(self, payload) -> None:
        result, timeline, _profile, audio, guard_report = payload
        self.tokens = result.tokens
        self.timeline = timeline
        self.audio = audio
        self._update_token_tree()
        self.timeline_canvas.set_timeline(timeline)
        backend = guard_report.get("backend", {}) if isinstance(guard_report, dict) else {}
        backend_name = backend.get("active", "off") if isinstance(backend, dict) else "off"
        corrected = guard_report.get("corrected_samples", 0) if isinstance(guard_report, dict) else 0
        self.status_var.set(
            f"合成完成：{len(audio) / 48000.0:.2f} 秒；精算={backend_name}；守門修正={corrected} samples。"
        )

    def _accept_render_and_play(self, payload) -> None:
        self._accept_render(payload)
        try:
            play_audio(self.audio, 48_000)
            self.status_var.set(self.status_var.get() + " 正在播放。")
        except Exception as exc:
            messagebox.showwarning("播放失敗", str(exc), parent=self.root)

    def render_debug(self) -> None:
        try:
            request = self._capture_request()
        except Exception as exc:
            messagebox.showerror("參數錯誤", str(exc), parent=self.root)
            return

        project_root = Path(__file__).resolve().parents[2]
        debug_root = project_root / "Debug_log"
        debug_root.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        destination = debug_root / stamp
        suffix = 1
        while destination.exists():
            destination = debug_root / f"{stamp}_{suffix:02d}"
            suffix += 1
        request["debug_dir"] = str(destination)

        def worker():
            result, timeline, profile = self._analyze_request(request)
            trace: dict[str, np.ndarray] = {}
            guard_report: dict[str, object] = {}
            audio = synthesize_timeline(
                timeline,
                profile,
                debug_trace=trace,
                compute_backend=request["compute_backend"],
                precision_guard=request["precision_guard"],
                precision_guard_report=guard_report,
            )
            report_path, report = write_debug_bundle(
                request["debug_dir"],
                trace,
                profile.sample_rate,
                timeline=timeline,
                profile=profile,
                seed=7,
                run_ablations=True,
                compute_backend=request["compute_backend"],
                precision_guard=request["precision_guard"],
                precision_guard_report=guard_report,
            )
            return result, timeline, profile, audio, guard_report, report_path, report

        self._run_async(
            "正在合成並追蹤每個聲源、濾波與爆音候選……",
            worker,
            self._accept_debug,
        )

    def _accept_debug(self, payload) -> None:
        result, timeline, profile, audio, guard_report, report_path, report = payload
        self._accept_render((result, timeline, profile, audio, guard_report))
        event_count = int(report["analysis"]["event_count"])
        precision = report.get("precision_audit", {})
        backend = precision.get("backend", {}).get("active", "cpu")
        precision_events = precision.get("summary", {}).get("event_count", 0)
        source_stage_count = len(report.get("source_stage_files", []))
        self.status_var.set(
            f"Debug 完成：一般候選={event_count}、精算候選={precision_events}、"
            f"Source 階段={source_stage_count}、後端={backend}；報告：{report_path}"
        )
        try:
            webbrowser.open(report_path.resolve().as_uri())
        except Exception:
            pass
        messagebox.showinfo(
            "爆音 Debug 完成",
            f"一般候選 {event_count} 個；精算後端 {backend}；Source 階段 {source_stage_count} 個。"
            f"\n\n逐層 WAV：\n{report_path.parent / 'source_stages'}"
            f"\n\n完整輸出：\n{report_path.parent}",
            parent=self.root,
        )

    def _update_token_tree(self) -> None:
        self.token_tree.delete(*self.token_tree.get_children())
        if self.timeline is None:
            return
        for span in self.timeline.spans:
            self.token_tree.insert(
                "",
                "end",
                values=(
                    span.text.replace("\n", "\\n"),
                    span.pinyin or "",
                    span.initial,
                    span.final,
                    span.tone if span.kind == "syllable" else "",
                    f"{span.start_s:.3f}",
                    f"{span.end_s:.3f}",
                ),
            )

    def stop(self) -> None:
        stop_audio()
        self.status_var.set("已停止播放。")

    def export_wav(self) -> None:
        if self.audio is None:
            messagebox.showinfo("尚未合成", "請先按『只合成』或『合成並播放』。", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            title="匯出 WAV",
            defaultextension=".wav",
            filetypes=[("WAV audio", "*.wav")],
            initialfile="digivocal.wav",
        )
        if path:
            write_wav(path, self.audio, 48_000)
            self.status_var.set(f"已匯出 WAV：{path}")

    def export_gestures(self) -> None:
        if self.timeline is None:
            messagebox.showinfo("尚未分析", "請先按『分析動作』。", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            title="匯出發聲動作",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="gestures.json",
        )
        if path:
            Path(path).write_text(
                json.dumps(self.timeline.to_dict(), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            self.status_var.set(f"已匯出動作 JSON：{path}")

    def export_vtl_bridge(self) -> None:
        if self.timeline is None:
            messagebox.showinfo("尚未分析", "請先按『分析動作』。", parent=self.root)
            return
        path = filedialog.asksaveasfilename(
            title="匯出 VTL 橋接 JSON",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="vtl_bridge.json",
        )
        if path:
            export_vtl_bridge_payload(self.timeline, path)
            self.status_var.set(f"已匯出 VTL 橋接資料：{path}")

    def save_profile(self) -> None:
        path = filedialog.asksaveasfilename(
            title="儲存聲音輪廓",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialfile="voice_profile.json",
        )
        if path:
            self.current_profile().save(path)
            self.status_var.set(f"已儲存聲音輪廓：{path}")

    def load_profile(self) -> None:
        path = filedialog.askopenfilename(title="載入聲音輪廓", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            profile = VoiceProfile.load(path)
            self._apply_profile_to_ui(profile, status=f"已載入聲音輪廓：{path}")
        except Exception as exc:
            messagebox.showerror("載入失敗", str(exc), parent=self.root)

    def refresh_components(self) -> None:
        lines = [
            "DigiVocal core\n",
            "  - 自製簡化構音規劃器與 source-filter renderer：已包含\n",
            "  - pypinyin：核心中文前端依賴\n",
            "\n可選開源元件\n",
        ]
        for name, info in component_status().items():
            mark = "YES" if info["installed"] else "NO "
            lines.append(f"  [{mark}] {name:<24} {info['license']:<10} {info['role']}\n")
        lines.extend(
            [
                "\n安裝提示\n",
                "  g2pW: 雙擊 install_g2pw_windows.bat；首次使用可能下載模型。\n",
                "  SPARC: 安裝 pyproject.toml 的 sparc 選用依賴；依賴很大。\n",
                "  VocalTractLab: 建議獨立環境安裝；目前只輸出橋接 JSON。\n",
                "\n授權\n",
                "  本專案核心為 MIT。VocalTractLab 為 GPL-3.0，散布整合版本前請自行評估授權義務。\n",
            ]
        )
        self.components_text.configure(state="normal")
        self.components_text.delete("1.0", "end")
        self.components_text.insert("1.0", "".join(lines))
        self.components_text.configure(state="disabled")

    def show_vtl_probe(self) -> None:
        messagebox.showinfo(
            "VocalTractLab API 探測",
            json.dumps(probe_vocaltractlab(), ensure_ascii=False, indent=2),
            parent=self.root,
        )


def run() -> None:
    root = tk.Tk()
    app = DigiVocalApp(root)

    def close() -> None:
        app.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()
