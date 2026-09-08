"""Run a Monte-Carlo sweep off the Tk main loop.

A sweep of a few hundred points can take tens of seconds.  Doing that inline
would freeze the window, so the work goes to a worker thread and reports back
through a queue that the main thread drains on a timer.  Only the main thread
ever touches a widget or a Figure.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, Optional, Sequence

from ..core import Scenario, SweepResult, doa_sweep


class SweepRunner:
    """One cancellable background sweep, bound to a Tk widget for timing."""

    POLL_MS = 80

    def __init__(self, widget, on_progress: Callable[[float, str], None],
                 on_done: Callable[[Optional[SweepResult], Optional[str]], None]):
        self._widget = widget
        self._on_progress = on_progress
        self._on_done = on_done
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def busy(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, cfg: Scenario, sweep_type: str, values, trials: int,
              keys: Sequence[str]):
        if self.busy:
            return
        self._stop.clear()
        self._queue = queue.Queue()

        def work():
            try:
                S = doa_sweep(
                    cfg, sweep_type, values, trials, keys,
                    progress=lambda frac, text: self._queue.put(("p", frac, text)),
                    should_stop=self._stop.is_set,
                )
                self._queue.put(("d", S, None))
            except Exception as exc:                       # noqa: BLE001
                self._queue.put(("d", None, str(exc)))

        self._thread = threading.Thread(target=work, daemon=True)
        self._thread.start()
        self._widget.after(self.POLL_MS, self._poll)

    def cancel(self):
        self._stop.set()

    # -----------------------------------------------------------------
    def _poll(self):
        finished = False
        try:
            while True:
                msg = self._queue.get_nowait()
                if msg[0] == "p":
                    self._on_progress(msg[1], msg[2])
                else:
                    finished = True
                    self._on_done(msg[1], msg[2])
        except queue.Empty:
            pass
        if not finished:
            self._widget.after(self.POLL_MS, self._poll)
