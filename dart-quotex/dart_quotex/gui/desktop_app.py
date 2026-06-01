"""
dart_quotex/gui/desktop_app.py
Full-featured CustomTkinter desktop application.

Panels
------
1. Header bar       — live balance, session P&L, connection status
2. Signal panel     — current AI signal, confidence meter, regime badge
3. Chart panel      — embedded candlestick chart (matplotlib)
4. Metrics panel    — real-time win rate, profit factor, drawdown gauges
5. MTF panel        — multi-timeframe alignment matrix
6. Trade log        — scrollable live trade history table
7. AI debug panel   — ensemble weights, SAC action, uncertainty scores
8. Control panel    — start/stop, asset selector, duration, risk override
9. News sidebar     — latest sentiment headlines

Launch
------
    from dart_quotex.gui.desktop_app import launch
    launch()                             # blocking
    # or non-blocking:
    import threading; threading.Thread(target=launch, daemon=True).start()
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── CustomTkinter import guard ────────────────────────────────────────────────
try:
    import customtkinter as ctk
    from tkinter import ttk, messagebox
    import tkinter as tk
    _CTK = True
except ImportError:
    _CTK = False
    logger.warning("customtkinter not installed — GUI unavailable. "
                   "Install: pip install customtkinter")

# ── Matplotlib for chart ──────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.patches as mpatches
    _MPL = True
except ImportError:
    _MPL = False

# ──────────────────────────────────────────────────────────────────────────────
# Colour palette (dark glassmorphism theme)
# ──────────────────────────────────────────────────────────────────────────────

COLORS = {
    "bg":          "#0d0f14",
    "panel":       "#151820",
    "panel_alt":   "#1a1d26",
    "border":      "#2a2d3a",
    "accent":      "#00d4aa",       # teal
    "accent2":     "#7c5cbf",       # purple
    "call":        "#00e676",       # green
    "put":         "#ff5252",       # red
    "hold":        "#ffa726",       # amber
    "text_primary": "#e8eaf6",
    "text_muted":   "#7986cb",
    "win":          "#00e676",
    "loss":         "#ff5252",
    "neutral":      "#b0bec5",
    "chart_bg":     "#0d0f14",
    "chart_grid":   "#1e2130",
    "candle_up":    "#00e676",
    "candle_dn":    "#ff5252",
}


# ──────────────────────────────────────────────────────────────────────────────
# State container (shared between async loop and Tkinter main thread)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class UIState:
    # Connection
    connected: bool = False
    account_mode: str = "DEMO"

    # Balance
    balance: float = 0.0
    session_pnl: float = 0.0
    start_balance: float = 0.0

    # Current signal
    signal_direction: str = "—"
    signal_confidence: float = 0.0
    signal_asset: str = ""

    # Regime
    regime_name: str = "DETECTING"
    regime_probs: List[float] = field(default_factory=lambda: [0.0] * 7)

    # Session stats
    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    sortino: float = 0.0
    sharpe: float = 0.0

    # Trade log (recent 50)
    trades: List[dict] = field(default_factory=list)

    # AI debug
    ensemble_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    sac_direction: float = 0.0
    sac_conviction: float = 0.0
    uncertainty_score: float = 0.0
    ece: float = 0.0

    # MTF alignment
    mtf_data: Dict[str, dict] = field(default_factory=dict)
    mtf_confluence: float = 0.0
    mtf_direction: str = "—"

    # Candles for chart (list of dicts)
    candles: List[dict] = field(default_factory=list)

    # News
    news_items: List[dict] = field(default_factory=list)
    news_sentiment: float = 0.0

    # Status message
    status: str = "Ready"
    error: str = ""


# ──────────────────────────────────────────────────────────────────────────────
# GUI Application
# ──────────────────────────────────────────────────────────────────────────────

class DARTApp:
    """
    Main CustomTkinter application window.
    """

    REFRESH_MS = 500    # UI refresh interval (ms)

    def __init__(self) -> None:
        if not _CTK:
            raise RuntimeError("customtkinter is required for the GUI")

        self.state      = UIState()
        self._update_q: queue.Queue = queue.Queue()
        self._running   = False
        self._advisor   = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._async_thread: Optional[threading.Thread] = None

        # Build window
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")

        self.root = ctk.CTk()
        self.root.title("DART-Quotex  ·  AI Trading Dashboard")
        self.root.geometry("1600×950")
        self.root.minsize(1200, 700)
        self.root.configure(fg_color=COLORS["bg"])
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._build_ui()
        self.root.after(self.REFRESH_MS, self._refresh_ui)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=3)
        self.root.columnconfigure(2, weight=1)
        self.root.rowconfigure(1, weight=1)

        self._build_header()
        self._build_left_panel()
        self._build_center_panel()
        self._build_right_panel()
        self._build_status_bar()

    def _build_header(self) -> None:
        hdr = ctk.CTkFrame(self.root, height=60, fg_color=COLORS["panel"], corner_radius=0)
        hdr.grid(row=0, column=0, columnspan=3, sticky="ew", padx=0, pady=0)
        hdr.columnconfigure(3, weight=1)

        # Logo
        ctk.CTkLabel(
            hdr, text="◈  DART-QUOTEX",
            font=ctk.CTkFont("Consolas", 20, "bold"),
            text_color=COLORS["accent"],
        ).grid(row=0, column=0, padx=20, pady=10)

        # Balance
        self._lbl_balance = ctk.CTkLabel(
            hdr, text="Balance: $—",
            font=ctk.CTkFont("Consolas", 14, "bold"),
            text_color=COLORS["text_primary"],
        )
        self._lbl_balance.grid(row=0, column=1, padx=20)

        # Session P&L
        self._lbl_pnl = ctk.CTkLabel(
            hdr, text="Session P&L: —",
            font=ctk.CTkFont("Consolas", 14),
            text_color=COLORS["text_muted"],
        )
        self._lbl_pnl.grid(row=0, column=2, padx=20)

        # Connection indicator
        self._lbl_conn = ctk.CTkLabel(
            hdr, text="● DISCONNECTED",
            font=ctk.CTkFont("Consolas", 13),
            text_color=COLORS["loss"],
        )
        self._lbl_conn.grid(row=0, column=4, padx=20)

        # Account mode badge
        self._lbl_mode = ctk.CTkLabel(
            hdr, text="[ DEMO ]",
            font=ctk.CTkFont("Consolas", 12),
            text_color=COLORS["hold"],
        )
        self._lbl_mode.grid(row=0, column=5, padx=10)

    def _build_left_panel(self) -> None:
        left = ctk.CTkFrame(self.root, fg_color=COLORS["panel"], corner_radius=8)
        left.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        left.columnconfigure(0, weight=1)

        # ── Signal panel ──────────────────────────────────────────────────────
        self._build_signal_panel(left, row=0)

        # ── Regime panel ─────────────────────────────────────────────────────
        self._build_regime_panel(left, row=1)

        # ── MTF alignment ─────────────────────────────────────────────────────
        self._build_mtf_panel(left, row=2)

        # ── Control panel ─────────────────────────────────────────────────────
        self._build_control_panel(left, row=3)

    def _build_signal_panel(self, parent, row: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color=COLORS["panel_alt"], corner_radius=8)
        frame.grid(row=row, column=0, sticky="ew", padx=8, pady=(8, 4))
        frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="AI SIGNAL",
            font=ctk.CTkFont("Consolas", 11),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 0))

        self._lbl_signal = ctk.CTkLabel(
            frame, text="—",
            font=ctk.CTkFont("Consolas", 36, "bold"),
            text_color=COLORS["text_primary"],
        )
        self._lbl_signal.grid(row=1, column=0, pady=4)

        # Confidence bar
        ctk.CTkLabel(
            frame, text="Confidence",
            font=ctk.CTkFont("Consolas", 10),
            text_color=COLORS["text_muted"],
        ).grid(row=2, column=0, sticky="w", padx=12)

        self._prog_conf = ctk.CTkProgressBar(
            frame, width=200, height=14,
            progress_color=COLORS["accent"],
            fg_color=COLORS["border"],
        )
        self._prog_conf.set(0)
        self._prog_conf.grid(row=3, column=0, padx=12, pady=(2, 4))

        self._lbl_conf_val = ctk.CTkLabel(
            frame, text="0.0%",
            font=ctk.CTkFont("Consolas", 12),
            text_color=COLORS["accent"],
        )
        self._lbl_conf_val.grid(row=4, column=0, pady=(0, 8))

    def _build_regime_panel(self, parent, row: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color=COLORS["panel_alt"], corner_radius=8)
        frame.grid(row=row, column=0, sticky="ew", padx=8, pady=4)
        frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="MARKET REGIME",
            font=ctk.CTkFont("Consolas", 11),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 2))

        self._lbl_regime = ctk.CTkLabel(
            frame, text="DETECTING…",
            font=ctk.CTkFont("Consolas", 16, "bold"),
            text_color=COLORS["hold"],
        )
        self._lbl_regime.grid(row=1, column=0, padx=12, pady=(0, 4))

        # 7 regime probability bars
        REGIME_NAMES = [
            "TREND ↑", "TREND ↓", "RANGING", "VOLATILE",
            "BREAKOUT", "REVERSAL", "CHOPPY",
        ]
        REGIME_COLORS = [
            COLORS["call"], COLORS["put"], COLORS["accent2"],
            COLORS["loss"], COLORS["call"], COLORS["hold"], COLORS["text_muted"],
        ]
        self._regime_bars: List[ctk.CTkProgressBar] = []
        self._regime_lbls: List[ctk.CTkLabel] = []

        for i, (name, color) in enumerate(zip(REGIME_NAMES, REGIME_COLORS)):
            r = i + 2
            ctk.CTkLabel(
                frame, text=name,
                font=ctk.CTkFont("Consolas", 9),
                text_color=COLORS["text_muted"],
                width=70, anchor="w",
            ).grid(row=r, column=0, sticky="w", padx=12)

            pb = ctk.CTkProgressBar(
                frame, width=130, height=8,
                progress_color=color, fg_color=COLORS["border"],
            )
            pb.set(0)
            pb.grid(row=r, column=0, padx=(88, 12), sticky="e")
            self._regime_bars.append(pb)

            lbl = ctk.CTkLabel(
                frame, text="0%",
                font=ctk.CTkFont("Consolas", 8),
                text_color=color, width=28, anchor="e",
            )
            lbl.grid(row=r, column=0, sticky="e", padx=12)
            self._regime_lbls.append(lbl)

        frame.rowconfigure(9, minsize=8)

    def _build_mtf_panel(self, parent, row: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color=COLORS["panel_alt"], corner_radius=8)
        frame.grid(row=row, column=0, sticky="ew", padx=8, pady=4)
        frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="MULTI-TIMEFRAME",
            font=ctk.CTkFont("Consolas", 11),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, columnspan=3, sticky="w", padx=12, pady=(8, 2))

        TFS = ["1m", "5m", "15m", "1h"]
        COLS = ["TF", "Trend", "RSI", "Mom"]

        for ci, col in enumerate(COLS):
            ctk.CTkLabel(
                frame, text=col,
                font=ctk.CTkFont("Consolas", 9, "bold"),
                text_color=COLORS["text_muted"],
            ).grid(row=1, column=ci, padx=6, pady=2)

        self._mtf_cells: Dict[str, List[ctk.CTkLabel]] = {}
        for ri, tf in enumerate(TFS):
            ctk.CTkLabel(
                frame, text=tf,
                font=ctk.CTkFont("Consolas", 10, "bold"),
                text_color=COLORS["accent"],
            ).grid(row=ri + 2, column=0, padx=8, pady=2)

            cells = []
            for ci in range(3):
                lbl = ctk.CTkLabel(
                    frame, text="—",
                    font=ctk.CTkFont("Consolas", 10),
                    text_color=COLORS["text_primary"],
                )
                lbl.grid(row=ri + 2, column=ci + 1, padx=4, pady=2)
                cells.append(lbl)
            self._mtf_cells[tf] = cells

        # Confluence label
        self._lbl_confluence = ctk.CTkLabel(
            frame, text="Confluence: —",
            font=ctk.CTkFont("Consolas", 11, "bold"),
            text_color=COLORS["accent"],
        )
        self._lbl_confluence.grid(
            row=len(TFS) + 2, column=0, columnspan=4,
            padx=12, pady=(4, 8),
        )

    def _build_control_panel(self, parent, row: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color=COLORS["panel_alt"], corner_radius=8)
        frame.grid(row=row, column=0, sticky="ew", padx=8, pady=(4, 8))
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="CONTROLS",
            font=ctk.CTkFont("Consolas", 11),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=12, pady=(8, 4))

        # Asset dropdown
        ctk.CTkLabel(
            frame, text="Asset:",
            font=ctk.CTkFont("Consolas", 10),
            text_color=COLORS["text_muted"],
        ).grid(row=1, column=0, sticky="w", padx=12, pady=2)

        self._cmb_asset = ctk.CTkComboBox(
            frame,
            values=[
                "EURUSD_OTC", "GBPUSD_OTC", "USDJPY_OTC",
                "AUDUSD_OTC", "USDCAD_OTC", "EURJPY_OTC",
                "GBPJPY_OTC", "EURGBP_OTC",
            ],
            font=ctk.CTkFont("Consolas", 11),
            width=160,
        )
        self._cmb_asset.set("EURUSD_OTC")
        self._cmb_asset.grid(row=1, column=1, padx=8, pady=2)

        # Duration
        ctk.CTkLabel(
            frame, text="Duration:",
            font=ctk.CTkFont("Consolas", 10),
            text_color=COLORS["text_muted"],
        ).grid(row=2, column=0, sticky="w", padx=12, pady=2)

        self._cmb_duration = ctk.CTkComboBox(
            frame,
            values=["30", "60", "120", "180", "300"],
            font=ctk.CTkFont("Consolas", 11),
            width=80,
        )
        self._cmb_duration.set("60")
        self._cmb_duration.grid(row=2, column=1, padx=8, pady=2, sticky="w")

        # Risk override
        ctk.CTkLabel(
            frame, text="Min Conf:",
            font=ctk.CTkFont("Consolas", 10),
            text_color=COLORS["text_muted"],
        ).grid(row=3, column=0, sticky="w", padx=12, pady=2)

        self._slider_conf = ctk.CTkSlider(
            frame, from_=0.5, to=0.9, width=120,
            button_color=COLORS["accent"],
            progress_color=COLORS["accent"],
        )
        self._slider_conf.set(0.62)
        self._slider_conf.grid(row=3, column=1, padx=8, pady=4)

        self._lbl_slider_val = ctk.CTkLabel(
            frame, text="0.62",
            font=ctk.CTkFont("Consolas", 10),
            text_color=COLORS["accent"],
        )
        self._lbl_slider_val.grid(row=4, column=1, sticky="w", padx=8)
        self._slider_conf.configure(command=lambda v: self._lbl_slider_val.configure(text=f"{v:.2f}"))

        # Start / Stop buttons
        self._btn_start = ctk.CTkButton(
            frame, text="▶  START",
            font=ctk.CTkFont("Consolas", 13, "bold"),
            fg_color=COLORS["call"], hover_color="#00c853",
            text_color=COLORS["bg"],
            command=self._on_start,
            width=120, height=36,
        )
        self._btn_start.grid(row=5, column=0, padx=8, pady=(8, 4))

        self._btn_stop = ctk.CTkButton(
            frame, text="■  STOP",
            font=ctk.CTkFont("Consolas", 13, "bold"),
            fg_color=COLORS["put"], hover_color="#d32f2f",
            text_color=COLORS["bg"],
            command=self._on_stop,
            state="disabled",
            width=120, height=36,
        )
        self._btn_stop.grid(row=5, column=1, padx=8, pady=(8, 4))

        frame.rowconfigure(6, minsize=8)

    def _build_center_panel(self) -> None:
        center = ctk.CTkFrame(self.root, fg_color=COLORS["panel"], corner_radius=8)
        center.grid(row=1, column=1, sticky="nsew", padx=4, pady=8)
        center.columnconfigure(0, weight=1)
        center.rowconfigure(0, weight=2)
        center.rowconfigure(1, weight=1)
        center.rowconfigure(2, weight=1)

        # ── Chart ─────────────────────────────────────────────────────────────
        self._build_chart_panel(center, row=0)

        # ── Session metrics ───────────────────────────────────────────────────
        self._build_metrics_panel(center, row=1)

        # ── Trade log ─────────────────────────────────────────────────────────
        self._build_trade_log(center, row=2)

    def _build_chart_panel(self, parent, row: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color=COLORS["chart_bg"], corner_radius=8)
        frame.grid(row=row, column=0, sticky="nsew", padx=8, pady=(8, 4))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="PRICE CHART",
            font=ctk.CTkFont("Consolas", 11),
            text_color=COLORS["text_muted"],
        ).pack(anchor="nw", padx=12, pady=(6, 0))

        if _MPL:
            self._fig = Figure(
                figsize=(8, 3.5), dpi=100,
                facecolor=COLORS["chart_bg"],
            )
            self._ax = self._fig.add_subplot(111)
            self._ax.set_facecolor(COLORS["chart_bg"])
            self._ax.tick_params(colors=COLORS["text_muted"], labelsize=8)
            for spine in self._ax.spines.values():
                spine.set_color(COLORS["chart_grid"])

            self._canvas = FigureCanvasTkAgg(self._fig, master=frame)
            self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=4)
        else:
            ctk.CTkLabel(
                frame,
                text="Install matplotlib for live charts\npip install matplotlib",
                text_color=COLORS["text_muted"],
                font=ctk.CTkFont("Consolas", 12),
            ).pack(expand=True)

    def _build_metrics_panel(self, parent, row: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color=COLORS["panel_alt"], corner_radius=8)
        frame.grid(row=row, column=0, sticky="ew", padx=8, pady=4)

        metrics = [
            ("Win Rate",     "_lbl_wr",    "—"),
            ("Profit Factor","_lbl_pf",    "—"),
            ("Max DD",       "_lbl_dd",    "—"),
            ("Sharpe",       "_lbl_sh",    "—"),
            ("Sortino",      "_lbl_sort",  "—"),
            ("Trades",       "_lbl_ntrd",  "0"),
        ]

        for i, (label, attr, default) in enumerate(metrics):
            col_frame = ctk.CTkFrame(frame, fg_color="transparent")
            col_frame.grid(row=0, column=i, padx=16, pady=10)

            ctk.CTkLabel(
                col_frame, text=label,
                font=ctk.CTkFont("Consolas", 9),
                text_color=COLORS["text_muted"],
            ).pack()

            lbl = ctk.CTkLabel(
                col_frame, text=default,
                font=ctk.CTkFont("Consolas", 16, "bold"),
                text_color=COLORS["text_primary"],
            )
            lbl.pack()
            setattr(self, attr, lbl)

    def _build_trade_log(self, parent, row: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color=COLORS["panel_alt"], corner_radius=8)
        frame.grid(row=row, column=0, sticky="nsew", padx=8, pady=(4, 8))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ctk.CTkLabel(
            frame, text="TRADE LOG",
            font=ctk.CTkFont("Consolas", 11),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(6, 0))

        # Treeview
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.Treeview",
            background=COLORS["panel_alt"],
            foreground=COLORS["text_primary"],
            fieldbackground=COLORS["panel_alt"],
            rowheight=22,
            font=("Consolas", 10),
        )
        style.configure("Dark.Treeview.Heading",
            background=COLORS["panel"],
            foreground=COLORS["text_muted"],
            font=("Consolas", 9, "bold"),
        )
        style.map("Dark.Treeview",
            background=[("selected", COLORS["border"])],
        )

        cols = ("#", "Asset", "Dir", "Stake", "Conf", "Result", "P&L", "Bal")
        self._trade_tree = ttk.Treeview(
            frame, columns=cols, show="headings",
            style="Dark.Treeview", height=7,
        )
        col_widths = [35, 90, 48, 60, 55, 60, 70, 80]
        for col, w in zip(cols, col_widths):
            self._trade_tree.heading(col, text=col)
            self._trade_tree.column(col, width=w, anchor="center")

        sb = ttk.Scrollbar(frame, orient="vertical",
                           command=self._trade_tree.yview)
        self._trade_tree.configure(yscrollcommand=sb.set)
        self._trade_tree.grid(row=1, column=0, sticky="nsew", padx=(8, 0), pady=(4, 8))
        sb.grid(row=1, column=1, sticky="ns", pady=(4, 8))

    def _build_right_panel(self) -> None:
        right = ctk.CTkFrame(self.root, fg_color=COLORS["panel"], corner_radius=8)
        right.grid(row=1, column=2, sticky="nsew", padx=(4, 8), pady=8)
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # ── AI Debug ─────────────────────────────────────────────────────────
        self._build_ai_debug_panel(right, row=0)

        # ── News sidebar ─────────────────────────────────────────────────────
        self._build_news_panel(right, row=1)

    def _build_ai_debug_panel(self, parent, row: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color=COLORS["panel_alt"], corner_radius=8)
        frame.grid(row=row, column=0, sticky="ew", padx=8, pady=(8, 4))
        frame.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            frame, text="AI INTERNALS",
            font=ctk.CTkFont("Consolas", 11),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 4))

        # Ensemble weights
        ctk.CTkLabel(frame, text="Ensemble Weights",
                     font=ctk.CTkFont("Consolas", 9), text_color=COLORS["text_muted"]
                     ).grid(row=1, column=0, sticky="w", padx=12)

        wt_frame = ctk.CTkFrame(frame, fg_color="transparent")
        wt_frame.grid(row=2, column=0, sticky="ew", padx=12, pady=2)
        self._ens_weight_labels: List[ctk.CTkLabel] = []
        for i, name in enumerate(["RF", "GB", "SGD"]):
            ctk.CTkLabel(wt_frame, text=name,
                         font=ctk.CTkFont("Consolas", 9), text_color=COLORS["text_muted"]
                         ).grid(row=0, column=i * 2, padx=(0, 2))
            lbl = ctk.CTkLabel(wt_frame, text="1.00",
                               font=ctk.CTkFont("Consolas", 9, "bold"),
                               text_color=COLORS["accent"])
            lbl.grid(row=0, column=i * 2 + 1, padx=(0, 8))
            self._ens_weight_labels.append(lbl)

        # SAC action
        rows_debug = [
            ("SAC Dir",   "_lbl_sac_dir",  "—"),
            ("SAC Conv",  "_lbl_sac_conv", "—"),
            ("Uncertainty","_lbl_unc",     "—"),
            ("Calib ECE", "_lbl_ece",      "—"),
            ("ICM Novelty","_lbl_icm",     "—"),
        ]
        for ri, (label, attr, default) in enumerate(rows_debug, start=3):
            ctk.CTkLabel(
                frame, text=f"{label}:",
                font=ctk.CTkFont("Consolas", 9),
                text_color=COLORS["text_muted"],
            ).grid(row=ri, column=0, sticky="w", padx=12, pady=1)
            lbl = ctk.CTkLabel(
                frame, text=default,
                font=ctk.CTkFont("Consolas", 10, "bold"),
                text_color=COLORS["text_primary"],
            )
            lbl.grid(row=ri, column=0, sticky="e", padx=12)
            setattr(self, attr, lbl)

        frame.rowconfigure(9, minsize=6)

    def _build_news_panel(self, parent, row: int) -> None:
        frame = ctk.CTkFrame(parent, fg_color=COLORS["panel_alt"], corner_radius=8)
        frame.grid(row=row, column=0, sticky="nsew", padx=8, pady=(4, 8))
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(2, weight=1)

        ctk.CTkLabel(
            frame, text="NEWS SENTIMENT",
            font=ctk.CTkFont("Consolas", 11),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, sticky="w", padx=12, pady=(8, 2))

        self._lbl_sent_score = ctk.CTkLabel(
            frame, text="Sentiment: —",
            font=ctk.CTkFont("Consolas", 12, "bold"),
            text_color=COLORS["text_primary"],
        )
        self._lbl_sent_score.grid(row=1, column=0, sticky="w", padx=12, pady=2)

        # Scrollable news items
        self._news_box = ctk.CTkScrollableFrame(
            frame, fg_color=COLORS["bg"], corner_radius=4,
        )
        self._news_box.grid(row=2, column=0, sticky="nsew", padx=8, pady=(2, 8))
        self._news_item_labels: List[ctk.CTkLabel] = []

    def _build_status_bar(self) -> None:
        bar = ctk.CTkFrame(self.root, height=24, fg_color=COLORS["panel"], corner_radius=0)
        bar.grid(row=2, column=0, columnspan=3, sticky="ew")
        bar.columnconfigure(0, weight=1)

        self._lbl_status = ctk.CTkLabel(
            bar, text="Ready",
            font=ctk.CTkFont("Consolas", 10),
            text_color=COLORS["text_muted"],
            anchor="w",
        )
        self._lbl_status.grid(row=0, column=0, sticky="w", padx=12)

    # ── UI refresh ────────────────────────────────────────────────────────────

    def _refresh_ui(self) -> None:
        """Called every REFRESH_MS ms from Tkinter main thread."""
        try:
            # Drain update queue
            while True:
                try:
                    update = self._update_q.get_nowait()
                    self._apply_update(update)
                except queue.Empty:
                    break

            self._update_widgets()
        except Exception as exc:
            logger.error("UI refresh error: %s", exc)
        finally:
            self.root.after(self.REFRESH_MS, self._refresh_ui)

    def _apply_update(self, update: dict) -> None:
        """Apply a state update dict from async thread."""
        for key, value in update.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)

    def _update_widgets(self) -> None:
        s = self.state

        # Header
        self._lbl_balance.configure(text=f"Balance: ₹{s.balance:,.2f}")
        pnl_color = COLORS["win"] if s.session_pnl >= 0 else COLORS["loss"]
        self._lbl_pnl.configure(
            text=f"P&L: {'+' if s.session_pnl >= 0 else ''}{s.session_pnl:,.2f}",
            text_color=pnl_color,
        )
        if s.connected:
            self._lbl_conn.configure(text=f"● {s.account_mode}", text_color=COLORS["win"])
        else:
            self._lbl_conn.configure(text="● DISCONNECTED", text_color=COLORS["loss"])

        # Signal
        sig_color = (
            COLORS["call"] if s.signal_direction == "CALL" else
            COLORS["put"]  if s.signal_direction == "PUT"  else
            COLORS["text_muted"]
        )
        self._lbl_signal.configure(
            text=s.signal_direction if s.signal_direction else "—",
            text_color=sig_color,
        )
        self._prog_conf.set(s.signal_confidence)
        self._prog_conf.configure(
            progress_color=sig_color if s.signal_confidence > 0 else COLORS["border"]
        )
        self._lbl_conf_val.configure(
            text=f"{s.signal_confidence:.1%}",
            text_color=sig_color,
        )

        # Regime
        regime_color = {
            "TRENDING_UP":   COLORS["call"],
            "TRENDING_DOWN": COLORS["put"],
            "VOLATILE":      COLORS["loss"],
            "RANGING":       COLORS["accent2"],
            "BREAKOUT":      COLORS["call"],
            "REVERSAL":      COLORS["hold"],
            "CHOPPY":        COLORS["text_muted"],
        }.get(s.regime_name, COLORS["text_primary"])
        self._lbl_regime.configure(text=s.regime_name, text_color=regime_color)

        for i, (pb, lbl) in enumerate(zip(self._regime_bars, self._regime_lbls)):
            prob = s.regime_probs[i] if i < len(s.regime_probs) else 0.0
            pb.set(float(prob))
            lbl.configure(text=f"{prob:.0%}")

        # MTF
        for tf, cells in self._mtf_cells.items():
            data = s.mtf_data.get(tf, {})
            trend = data.get("trend", "—")
            rsi   = data.get("rsi", 0.0)
            mom   = data.get("momentum", 0.0)
            cells[0].configure(
                text=trend,
                text_color=COLORS["call"] if trend == "UP" else COLORS["put"] if trend == "DOWN" else COLORS["text_muted"],
            )
            cells[1].configure(text=f"{rsi:.0f}")
            cells[2].configure(
                text=f"{mom:+.3f}",
                text_color=COLORS["call"] if mom > 0 else COLORS["put"] if mom < 0 else COLORS["text_muted"],
            )
        self._lbl_confluence.configure(
            text=f"Confluence: {s.mtf_direction} {s.mtf_confluence:.0%}",
            text_color=COLORS["call"] if s.mtf_direction == "CALL" else
                       COLORS["put"] if s.mtf_direction == "PUT" else
                       COLORS["text_muted"],
        )

        # Metrics
        wr_color = COLORS["win"] if s.win_rate >= 0.55 else COLORS["loss"]
        self._lbl_wr.configure(text=f"{s.win_rate:.1%}", text_color=wr_color)
        self._lbl_pf.configure(text=f"{s.profit_factor:.2f}")
        dd_color = COLORS["loss"] if s.max_drawdown > 0.07 else COLORS["text_primary"]
        self._lbl_dd.configure(text=f"{s.max_drawdown:.1%}", text_color=dd_color)
        self._lbl_sh.configure(text=f"{s.sharpe:.2f}")
        self._lbl_sort.configure(text=f"{s.sortino:.2f}")
        self._lbl_ntrd.configure(text=str(s.n_trades))

        # AI debug
        for i, lbl in enumerate(self._ens_weight_labels):
            w = s.ensemble_weights[i] if i < len(s.ensemble_weights) else 0.0
            lbl.configure(text=f"{w:.2f}")
        sac_dir_str = f"{s.sac_direction:+.2f}"
        self._lbl_sac_dir.configure(
            text=sac_dir_str,
            text_color=COLORS["call"] if s.sac_direction > 0.1 else
                       COLORS["put"] if s.sac_direction < -0.1 else
                       COLORS["text_muted"],
        )
        self._lbl_sac_conv.configure(text=f"{s.sac_conviction:.2f}")
        unc_color = COLORS["loss"] if s.uncertainty_score > 0.6 else COLORS["text_primary"]
        self._lbl_unc.configure(text=f"{s.uncertainty_score:.2f}", text_color=unc_color)
        self._lbl_ece.configure(text=f"{s.ece:.3f}")

        # Trade log — add new rows
        existing = len(self._trade_tree.get_children())
        for i, t in enumerate(s.trades[existing:], start=existing + 1):
            result_str = "WIN" if t.get("won") else "LOSS"
            row_color  = "win_row" if t.get("won") else "loss_row"
            self._trade_tree.insert(
                "", 0,   # insert at top
                values=(
                    i,
                    t.get("asset", ""),
                    t.get("direction", "").upper(),
                    f"{t.get('stake', 0):.0f}",
                    f"{t.get('confidence', 0):.0%}",
                    result_str,
                    f"{t.get('payout', 0):+.2f}",
                    f"{t.get('balance', 0):.2f}",
                ),
                tags=(row_color,),
            )
        self._trade_tree.tag_configure("win_row",  foreground=COLORS["win"])
        self._trade_tree.tag_configure("loss_row", foreground=COLORS["loss"])

        # Chart
        if _MPL and s.candles:
            self._draw_chart(s.candles[-60:])

        # News
        sent_color = (
            COLORS["call"] if s.news_sentiment > 0.1 else
            COLORS["put"]  if s.news_sentiment < -0.1 else
            COLORS["text_muted"]
        )
        self._lbl_sent_score.configure(
            text=f"Sentiment: {s.news_sentiment:+.2f}",
            text_color=sent_color,
        )
        # Refresh news items
        for lbl in self._news_item_labels:
            lbl.destroy()
        self._news_item_labels.clear()
        for item in s.news_items[:12]:
            color = (
                COLORS["call"] if item.get("sentiment", 0) > 0.1 else
                COLORS["put"]  if item.get("sentiment", 0) < -0.1 else
                COLORS["text_muted"]
            )
            lbl = ctk.CTkLabel(
                self._news_box,
                text=item.get("title", "")[:50] + ("…" if len(item.get("title", "")) > 50 else ""),
                font=ctk.CTkFont("Consolas", 9),
                text_color=color,
                anchor="w",
                wraplength=200,
            )
            lbl.pack(anchor="w", padx=4, pady=2)
            self._news_item_labels.append(lbl)

        # Status
        self._lbl_status.configure(text=s.status)

    def _draw_chart(self, candles: List[dict]) -> None:
        """Draw candlestick chart using matplotlib."""
        self._ax.clear()
        self._ax.set_facecolor(COLORS["chart_bg"])
        self._ax.grid(True, color=COLORS["chart_grid"], linewidth=0.5, alpha=0.5)
        self._ax.tick_params(colors=COLORS["text_muted"], labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_color(COLORS["chart_grid"])

        xs = list(range(len(candles)))
        for i, c in enumerate(candles):
            o, h, l, cl = c["open"], c["high"], c["low"], c["close"]
            color = COLORS["candle_up"] if cl >= o else COLORS["candle_dn"]
            # Wick
            self._ax.plot([i, i], [l, h], color=color, linewidth=0.8)
            # Body
            body_h = abs(cl - o)
            body_y = min(cl, o)
            self._ax.bar(i, body_h, bottom=body_y, width=0.7,
                         color=color, alpha=0.85)

        self._ax.set_xlim(-1, len(candles))
        self._ax.yaxis.tick_right()
        self._fig.tight_layout(pad=0.3)
        self._canvas.draw_idle()

    # ── event handlers ────────────────────────────────────────────────────────

    def _on_start(self) -> None:
        asset    = self._cmb_asset.get()
        duration = int(self._cmb_duration.get())
        min_conf = self._slider_conf.get()

        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")
        self.push_update({"status": f"Starting {asset}…"})
        self._start_async_trader(asset, duration, min_conf)

    def _on_stop(self) -> None:
        self._running = False
        self._btn_start.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self.push_update({"status": "Stopping…"})
        if self._loop:
            asyncio.run_coroutine_threadsafe(
                self._stop_advisor(), self._loop
            )

    def _on_close(self) -> None:
        self._on_stop()
        self.root.after(800, self.root.destroy)

    def _start_async_trader(
        self, asset: str, duration: int, min_conf: float
    ) -> None:
        """Start async trading loop in background thread."""
        def _thread() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(
                self._trading_loop(asset, duration, min_conf)
            )
            self._loop.close()

        self._async_thread = threading.Thread(target=_thread, daemon=True)
        self._running = True
        self._async_thread.start()

    async def _trading_loop(
        self, asset: str, duration: int, min_conf: float
    ) -> None:
        from dart_quotex.advisor import AIAdvisor

        self._advisor = AIAdvisor()
        try:
            await self._advisor.connect()
            balance = await self._advisor.client.get_balance()
            self.push_update({
                "connected":    True,
                "balance":      balance,
                "start_balance": balance,
                "status":       f"Connected — trading {asset}",
            })

            while self._running:
                try:
                    result = await self._advisor.trade(asset=asset, duration=duration)
                    if result:
                        s = self.state
                        trades = list(s.trades) + [result]
                        wins   = sum(1 for t in trades if t.get("won"))
                        n      = len(trades)
                        pnl    = result["balance"] - s.start_balance
                        self.push_update({
                            "balance":      result["balance"],
                            "session_pnl":  pnl,
                            "n_trades":     n,
                            "n_wins":       wins,
                            "win_rate":     wins / n,
                            "trades":       trades[-50:],
                            "status":       f"{'WIN' if result['won'] else 'LOSS'} — ₹{result['payout']:+.2f}",
                        })

                    # Refresh chart candles
                    df  = self._advisor.db.get_candles(asset, 60, limit=60)
                    raw = [
                        {"open": r.open, "high": r.high, "low": r.low,
                         "close": r.close, "volume": r.volume}
                        for _, r in df.iterrows()
                    ]
                    self.push_update({"candles": raw})

                    await asyncio.sleep(5)

                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    logger.error("Trading loop error: %s", exc)
                    self.push_update({"status": f"Error: {exc}", "error": str(exc)})
                    await asyncio.sleep(10)

        finally:
            if self._advisor:
                await self._advisor.disconnect()
            self.push_update({"connected": False, "status": "Disconnected"})

    async def _stop_advisor(self) -> None:
        self._running = False
        if self._advisor:
            try:
                await self._advisor.disconnect()
            except Exception:
                pass

    # ── public helpers ────────────────────────────────────────────────────────

    def push_update(self, update: dict) -> None:
        """Thread-safe state update from async or other threads."""
        self._update_q.put(update)

    def run(self) -> None:
        """Start the Tkinter main loop (blocking)."""
        self.root.mainloop()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def launch() -> None:
    """Launch the DART-Quotex desktop GUI."""
    if not _CTK:
        print(
            "\nCustomTkinter is required for the desktop GUI.\n"
            "Install it with:  pip install customtkinter\n"
        )
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    app = DARTApp()
    app.run()


if __name__ == "__main__":
    launch()
