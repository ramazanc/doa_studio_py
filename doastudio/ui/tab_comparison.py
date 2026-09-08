"""Comparison tab: a descriptive reference, with no simulation of any kind.

The page renders the static content in ``doastudio.content.comparison`` into a
scrollable document with a section navigator.  Tables are embedded Tk frames
rather than monospace art, so they re-flow with the window instead of
breaking.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import List, Tuple

from ..content.comparison import INTRO, NAV_LABELS, SECTIONS
from . import theme

TABLE_HEADER_BG = "#e7edf4"
TABLE_ROW_BG = "#ffffff"
TABLE_ALT_BG = "#f7f9fb"
NOTE_BG = "#f2f7fc"
DOC_PAD = 26


class ComparisonTab(ttk.Frame):
    def __init__(self, master, app):
        super().__init__(master, padding=8)
        self.app = app
        self.f = app.f

        # Embedded tables, kept so their cell wraplengths can follow a resize
        # without re-rendering (which would lose the scroll position).
        self._tables: List[Tuple[tk.Frame, List[List[tk.Label]], List[float]]] = []
        self._rules: List[tk.Frame] = []
        self._last_width = 0
        self._resize_job = None

        self.columnconfigure(0, minsize=210, weight=0)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._build_nav()
        self._build_doc()
        self._render()

    # =================================================================
    # Layout
    # =================================================================
    def _build_nav(self):
        wrap = ttk.LabelFrame(self, text=" Contents ", padding=(6, 6, 6, 8))
        wrap.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)

        self.nav = tk.Listbox(
            wrap, activestyle="none", exportselection=False,
            font=self.f["doc_nav"], relief="flat", borderwidth=0,
            highlightthickness=0, background=theme.PANEL_BG,
            selectbackground=theme.ACCENT_LIGHT, selectforeground=theme.NAVY,
            width=24)
        self.nav.grid(row=0, column=0, sticky="nsew")
        for _sid, label in NAV_LABELS:
            self.nav.insert("end", "  " + label)
        self.nav.bind("<<ListboxSelect>>", self._on_nav)

        ttk.Label(wrap, style="Muted.TLabel", wraplength=180, justify="left",
                  text="Background reference only.\nNothing on this page is "
                       "simulated or computed."
                  ).grid(row=1, column=0, sticky="w", pady=(10, 0))

    def _build_doc(self):
        wrap = ttk.LabelFrame(
            self, text=" Direction Finding Algorithms - Comparative Reference ",
            padding=(2, 4, 2, 4))
        wrap.grid(row=0, column=1, sticky="nsew")
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=1)

        self.text = tk.Text(
            wrap, wrap="word", relief="flat", borderwidth=0,
            highlightthickness=0, background=theme.PANEL_BG,
            foreground=theme.TEXT, padx=DOC_PAD, pady=16,
            font=self.f["doc_body"], spacing1=1, spacing3=3, cursor="arrow")
        self.text.grid(row=0, column=0, sticky="nsew")

        sb = ttk.Scrollbar(wrap, orient="vertical", command=self.text.yview)
        sb.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=sb.set)

        self._configure_tags()

        # Read-only, but keep keyboard scrolling and selection working.
        self.text.bind("<Key>", self._on_key)
        self.text.bind("<Configure>", self._on_configure)

    def _configure_tags(self):
        t = self.text
        t.tag_configure("intro", font=self.f["doc_italic"],
                        foreground=theme.TEXT_MUTED, spacing3=14,
                        lmargin1=0, lmargin2=0)
        t.tag_configure("h1", font=self.f["doc_h1"], foreground=theme.NAVY,
                        spacing1=22, spacing3=10)
        t.tag_configure("h2", font=self.f["doc_h2"], foreground="#2a5f8f",
                        spacing1=14, spacing3=6)
        t.tag_configure("p", font=self.f["doc_body"], spacing3=9,
                        lmargin1=0, lmargin2=0)
        t.tag_configure("bullet", font=self.f["doc_body"], spacing3=6,
                        lmargin1=16, lmargin2=32)
        t.tag_configure("formula", font=self.f["doc_mono"],
                        background="#f4f6f8", foreground="#20303d",
                        lmargin1=24, lmargin2=24, rmargin=24,
                        spacing1=0, spacing3=2)
        t.tag_configure("note", font=self.f["doc_italic"],
                        background=NOTE_BG, foreground="#1a4a7a",
                        lmargin1=18, lmargin2=18, rmargin=18,
                        spacing1=0, spacing3=2)
        t.tag_configure("gap", font=self.f["small"], spacing1=0, spacing3=0)
        t.tag_configure("kvkey", font=self.f["doc_bold"], foreground=theme.NAVY)
        t.tag_configure("kvval", font=self.f["doc_body"],
                        lmargin1=16, lmargin2=16, spacing3=5)
        t.tag_configure("rule", font=self.f["small"], foreground="#dfe4e9",
                        spacing1=4, spacing3=10)

    # =================================================================
    # Rendering
    # =================================================================
    def _render(self):
        t = self.text
        t.configure(state="normal")
        t.delete("1.0", "end")
        self._tables.clear()
        self._rules.clear()

        t.insert("end", INTRO + "\n", "intro")

        for sec in SECTIONS:
            t.mark_set("sec_" + sec.id, "end-1c")
            t.mark_gravity("sec_" + sec.id, "left")
            t.insert("end", sec.title + "\n", "h1")
            for block in sec.blocks:
                self._render_block(block)
            self._insert_rule()

        t.insert("end", "\n")
        t.configure(state="disabled")

    def _insert_rule(self):
        """A hairline section divider that spans the document width.

        An embedded frame rather than a run of dashes: with wrap="word" a
        long dash run would fold onto several lines as the window narrows.
        """
        t = self.text
        rule = tk.Frame(t, height=1, background="#e1e6eb")
        t.insert("end", "\n", "gap")
        t.window_create("end", window=rule, pady=6)
        t.insert("end", "\n", "gap")
        self._rules.append(rule)

    def _render_block(self, block):
        kind = block[0]
        t = self.text

        if kind == "h2":
            t.insert("end", block[1] + "\n", "h2")

        elif kind == "p":
            t.insert("end", block[1] + "\n", "p")

        elif kind == "bullets":
            for item in block[1]:
                t.insert("end", "•   " + item + "\n", "bullet")

        elif kind == "numbers":
            for i, item in enumerate(block[1], 1):
                t.insert("end", "%d.   %s\n" % (i, item), "bullet")

        elif kind == "formula":
            # spacing1/spacing3 apply to every logical line, so the block is
            # padded from outside instead - otherwise a five-line formula
            # collects five lots of leading and trailing space.
            t.insert("end", "\n", "gap")
            t.insert("end", block[1] + "\n", "formula")
            t.insert("end", "\n", "gap")

        elif kind == "note":
            t.insert("end", "\n", "gap")
            t.insert("end", block[1] + "\n", "note")
            t.insert("end", "\n", "gap")

        elif kind == "kv":
            for key, value in block[1]:
                t.insert("end", key + "\n", "kvkey")
                t.insert("end", value + "\n", "kvval")

        elif kind == "table":
            self._render_table(block[1], block[2])

    def _render_table(self, headers, rows):
        """Embed a real widget grid so cells wrap instead of overflowing."""
        t = self.text
        frame = tk.Frame(t, background=theme.BORDER, padx=1, pady=1)

        ncols = len(headers)
        cells: List[List[tk.Label]] = []

        for c, head in enumerate(headers):
            lbl = tk.Label(frame, text=head, font=self.f["doc_h2"],
                           background=TABLE_HEADER_BG, foreground=theme.NAVY,
                           anchor="w", justify="left", padx=8, pady=5)
            lbl.grid(row=0, column=c, sticky="nsew", padx=(0, 1), pady=(0, 1))
        cells.append([])

        for r, row in enumerate(rows, start=1):
            bg = TABLE_ROW_BG if r % 2 else TABLE_ALT_BG
            row_cells = []
            for c in range(ncols):
                value = row[c] if c < len(row) else ""
                lbl = tk.Label(frame, text=value, font=self.f["doc_body"],
                               background=bg,
                               foreground=theme.TEXT if c else theme.NAVY,
                               anchor="nw", justify="left", padx=8, pady=4)
                lbl.grid(row=r, column=c, sticky="nsew", padx=(0, 1), pady=(0, 1))
                row_cells.append(lbl)
            cells.append(row_cells)

        # Column share is proportional to the longest cell in each column, so
        # a "Comment" column gets the room and a "yes/no" column does not.
        widths = []
        for c in range(ncols):
            longest = len(str(headers[c]))
            for row in rows:
                if c < len(row):
                    longest = max(longest, len(str(row[c])))
            widths.append(float(min(longest, 60)))
        total = sum(widths) or 1.0
        shares = [w / total for w in widths]

        for c, share in enumerate(shares):
            frame.columnconfigure(c, weight=max(1, int(share * 100)))

        t.insert("end", "\n")
        t.window_create("end", window=frame, padx=0, pady=4)
        t.insert("end", "\n")

        self._tables.append((frame, cells, shares))
        self._apply_table_widths(frame, cells, shares)
        return frame

    # =================================================================
    # Resize handling
    # =================================================================
    def _on_configure(self, _event=None):
        width = self.text.winfo_width()
        if abs(width - self._last_width) < 24:
            return
        self._last_width = width
        if self._resize_job is not None:
            self.after_cancel(self._resize_job)
        self._resize_job = self.after(120, self._reflow_tables)

    def _reflow_tables(self):
        self._resize_job = None
        for frame, cells, shares in self._tables:
            self._apply_table_widths(frame, cells, shares)
        for rule in self._rules:
            rule.configure(width=self._doc_width())

    def _doc_width(self) -> int:
        return max(360, self.text.winfo_width() - 2 * DOC_PAD - 22)

    def _apply_table_widths(self, frame, cells, shares):
        avail = self._doc_width()
        for c, share in enumerate(shares):
            col_px = max(58, int(avail * share))
            # minsize is what actually widens the table: the frame itself
            # shrink-wraps its children, so setting frame width alone does
            # nothing once the labels are narrower than the document.
            frame.columnconfigure(c, minsize=col_px)
            wrap = max(50, col_px - 20)
            for row_cells in cells:
                if c < len(row_cells):
                    row_cells[c].configure(wraplength=wrap)
        for child in frame.grid_slaves(row=0):
            c = int(child.grid_info()["column"])
            if c < len(shares):
                child.configure(wraplength=max(50, int(avail * shares[c]) - 20))

    # =================================================================
    # Interaction
    # =================================================================
    def _on_nav(self, _event=None):
        sel = self.nav.curselection()
        if not sel:
            return
        sid = NAV_LABELS[sel[0]][0]
        # yview(index) puts that line at the top of the view, which is what a
        # table-of-contents jump should do (see() only scrolls minimally).
        self.text.yview("sec_" + sid)

    @staticmethod
    def _on_key(event):
        """Read-only, but leave navigation and copy keys working."""
        allowed = {"Up", "Down", "Left", "Right", "Prior", "Next", "Home",
                   "End", "Tab", "Shift_L", "Shift_R", "Control_L", "Control_R"}
        if event.keysym in allowed:
            return None
        if event.state & 0x4 and event.keysym.lower() in ("c", "a"):
            return None                      # Ctrl+C / Ctrl+A
        return "break"
