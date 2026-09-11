"""Live presentation of authoritative pentanomial SPRT snapshots."""

import time as clock

from utils.plotting import BLUE, INK, RED, TEAL, enable_hover, style_dashboard
from utils.pentanomial import LABELS


class LivePlotter:
    def __init__(self, elo0, elo1, alpha=0.05, beta=0.05,
                 engine_a_data=None, engine_b_data=None, time=None, depth=None, tc=None):
        import matplotlib.pyplot as plt
        import math

        self.plt = plt
        self.elo0, self.elo1 = elo0, elo1
        self.lbound = math.log(beta / (1 - alpha))
        self.ubound = math.log((1 - beta) / alpha)
        self.history = []
        self.pending_snapshot = None
        self.last_render = 0.0
        self.tc = f"{time:g} s / move" if time else f"depth {depth}" if depth else str(tc or "")
        a = (engine_a_data or {}).get("version", "Candidate")
        b = (engine_b_data or {}).get("version", "Baseline")
        self.subtitle = f"{a}  vs  {b}  |  {self.tc}"
        plt.ion()
        self.fig = plt.figure(figsize=(15, 9), layout="constrained")
        grid = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.28])
        self.ax_llr = self.fig.add_subplot(grid[0, 0])
        self.ax_elo = self.fig.add_subplot(grid[0, 1])
        self.ax_score = self.fig.add_subplot(grid[1, 0])
        self.ax_pairs = self.fig.add_subplot(grid[1, 1])
        self.ax_info = self.fig.add_subplot(grid[2, :])
        self.axes = [self.ax_llr, self.ax_elo, self.ax_score, self.ax_pairs, self.ax_info]
        self._render()
        if self.fig.canvas.required_interactive_framework is not None:
            plt.show(block=False)

    def update(self, snapshot, force=False):
        if snapshot["pairs"] and (not self.history or snapshot["pairs"] != self.history[-1]["pairs"]):
            self.history.append(snapshot)
        self.pending_snapshot = snapshot
        if force or clock.monotonic() - self.last_render >= 0.5:
            self._flush_render()

    def _flush_render(self):
        if self.pending_snapshot is not None:
            self._render(self.pending_snapshot)
            self.pending_snapshot = None
            self.last_render = clock.monotonic()
            self.fig.canvas.draw_idle()

    def _render(self, snapshot=None):
        for ax in self.axes:
            ax.clear()
        latest = snapshot or (self.history[-1] if self.history else None)
        result = latest["result"] if latest else "inconclusive"
        status = {"pass": "H1 accepted", "fail": "H0 accepted"}.get(result, "Collecting evidence")
        self.fig.suptitle(f"SPRT  /  {status}\n{self.subtitle}", fontsize=17)
        xs = [s["games_played"] for s in self.history]
        formatters = {}

        def tooltip(index):
            s = self.history[index]
            return (f"{s['pairs']:,} opening pairs / {s['games_played']:,} tested games\n"
                    f"LLR: {s['llr']:+.5f}\nLogistic Elo: {s['elo_diff']:+.3f}\n"
                    f"95% CI: [{s['elo_low']:+.3f}, {s['elo_high']:+.3f}]\n"
                    f"Score: {s['score']:.5%}\nLL / LD / DD+WL / WD / WW: {s['pentanomial']}")

        for ax, key, title, ylabel, color in (
            (self.ax_llr, "llr", "Sequential evidence", "Log-likelihood ratio", TEAL),
            (self.ax_elo, "elo_diff", "Estimated strength", "Logistic Elo", BLUE),
            (self.ax_score, "score", "Candidate score", "Score per game", TEAL),
        ):
            line, = ax.plot(xs, [s[key] for s in self.history], color=color,
                            linewidth=2, label="Completed opening pairs")
            formatters[line] = tooltip
            ax.set(title=title, ylabel=ylabel, xlabel="Tested games")
            ax.set_xlim(0, max(10, xs[-1] * 1.03) if xs else 10)
        self.ax_llr.axhline(self.ubound, color=TEAL, linestyle="--", linewidth=1,
                            label=f"H1 boundary  {self.ubound:+.2f}")
        self.ax_llr.axhline(self.lbound, color=RED, linestyle="--", linewidth=1,
                            label=f"H0 boundary  {self.lbound:+.2f}")
        self.ax_llr.axhspan(0, self.ubound, color=TEAL, alpha=0.035)
        self.ax_llr.axhspan(self.lbound, 0, color=RED, alpha=0.035)
        self.ax_llr.legend(loc="upper left", fontsize=8)
        self.ax_elo.axhline(self.elo0, color=RED, linestyle=":", label=f"H0  {self.elo0:+g}")
        self.ax_elo.axhline(self.elo1, color=TEAL, linestyle=":", label=f"H1  {self.elo1:+g}")
        if self.history:
            self.ax_elo.fill_between(xs, [s["elo_low"] for s in self.history],
                                     [s["elo_high"] for s in self.history], color=BLUE,
                                     alpha=0.13, label="Approx. 95% CI")
            # Keep mature runs readable despite enormous initial uncertainty.
            tail = self.history[len(self.history) // 5:]
            lo = min(self.elo0, min(s["elo_low"] for s in tail))
            hi = max(self.elo1, max(s["elo_high"] for s in tail))
            pad = max(5, (hi - lo) * 0.08)
            self.ax_elo.set_ylim(lo - pad, hi + pad)
        self.ax_elo.legend(loc="upper right", fontsize=8)
        self.ax_score.axhline(0.5, color=INK, alpha=0.35, linewidth=1)
        self.ax_score.yaxis.set_major_formatter(self.plt.matplotlib.ticker.PercentFormatter(1))
        if self.history:
            scores = [s["score"] for s in self.history[len(self.history) // 5:]]
            lo, hi = min(.495, min(scores)), max(.505, max(scores))
            pad = max(.005, (hi - lo) * .1)
            self.ax_score.set_ylim(max(0, lo - pad), min(1, hi + pad))
        else:
            self.ax_score.set_ylim(.4, .6)

        counts = latest["pentanomial"] if latest else [0] * 5
        bars = self.ax_pairs.bar(range(5), counts, color=[RED, "#d9a39a", "#9caeb7", "#79b7b1", TEAL], width=0.62)
        self.ax_pairs.bar_label(bars, padding=4, color=INK, fontsize=10)
        self.ax_pairs.set_xticks(range(5), LABELS)
        self.ax_pairs.set(title="Opening-pair outcomes", ylabel="Completed pairs", xlabel="Candidate result across both colors")
        self.ax_pairs.set_ylim(0, max(1, max(counts) * 1.22))
        # An invisible-at-rest marker gives each bar an inspectable sample.
        markers, = self.ax_pairs.plot(range(5), counts, linestyle="", marker=".",
                                      alpha=0, label="Pair count")
        formatters[markers] = lambda i: f"{LABELS[i]}\nPairs: {counts[i]:,}\nShare: {counts[i] / max(1, sum(counts)):.2%}"
        self.ax_info.axis("off")
        if latest:
            w, d, l = (latest[f"candidate_{key}"] for key in ("wins", "draws", "losses"))
            left = f"{latest['pairs']:,} PAIRS   /   {latest['games_played']:,} TESTED GAMES\nCandidate W / D / L   {w:,} / {d:,} / {l:,}"
            right = (f"Untested completed games: {latest['untested_games']:,}\n"
                     "Pentanomial model • logistic Elo • CI is approximate, not sequential")
        else:
            left = "Waiting for the first complete opening pair"
            right = "Hover near curves or bar tops to inspect values"
        self.ax_info.text(0.01, 0.8, left, transform=self.ax_info.transAxes, va="top", fontsize=11, color=INK, linespacing=1.8)
        self.ax_info.text(0.5, 0.8, right, transform=self.ax_info.transAxes, va="top", fontsize=9, color=INK, linespacing=1.8)
        style_dashboard(self.fig, self.axes)
        enable_hover(self.fig, self.axes[:-1], formatters)

    def finalize(self, snapshot, out_png=None):
        self.update(snapshot, force=True)
        if snapshot["result"] == "inconclusive":
            self.fig.suptitle(f"SPRT  /  Inconclusive\n{self.subtitle}")
        if out_png:
            self.fig.savefig(out_png, dpi=140)

    def _pump_events(self):
        if self.plt.fignum_exists(self.fig.number):
            if clock.monotonic() - self.last_render >= 0.5:
                self._flush_render()
            self.fig.canvas.flush_events()

    def show_final(self):
        self.plt.ioff()
        self.plt.show()
