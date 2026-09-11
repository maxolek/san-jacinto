"""Shared dashboard styling and hover inspection using Matplotlib alone."""

import numpy as np

BACKGROUND = "#f5f4ef"
INK = "#243746"
TEAL = "#148579"
BLUE = "#4479b5"
RED = "#c46b61"
PALETTE = [TEAL, BLUE, "#b78c45", "#8974ae", RED, "#6598a0"]


def style_dashboard(fig, axes):
    fig.set_facecolor(BACKGROUND)
    for ax in axes:
        ax.set_facecolor("#fffefa")
        ax.set_axisbelow(True)
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color("#d9dfdf")
        ax.tick_params(colors="#647580", labelsize=9, length=0, pad=7)
        ax.xaxis.label.set(color=INK, fontsize=10)
        ax.yaxis.label.set(color=INK, fontsize=10)
        ax.title.set(color=INK, fontsize=12, fontweight="bold")
        if ax.axison:
            ax.grid(True, color="#dce3e3", alpha=0.6, linewidth=0.7)
        ax.set_prop_cycle(color=PALETTE)
        legend = ax.get_legend()
        if legend:
            legend.get_frame().set(facecolor="#fffefa", edgecolor="none", alpha=0.9)
            for label in legend.get_texts():
                label.set_color(INK)
    if fig._suptitle:
        fig._suptitle.set(color=INK, fontsize=17, fontweight="bold")


class HoverInspector:
    """Inspect nearest visible sample in screen coordinates, including log axes.

    Refresh bindings after axes.clear(); a figure keeps exactly one callback.
    Series may supply a formatter(index), e.g. to include confidence bounds.
    """

    def __init__(self, fig):
        self.fig = fig
        self.series = []
        self.annotations = {}
        self.cid = fig.canvas.mpl_connect("motion_notify_event", self.on_move)
        self.leave_cid = fig.canvas.mpl_connect("figure_leave_event", self.hide)

    def bind(self, axes, formatters=None):
        self.hide()
        for annotation in self.annotations.values():
            if annotation.axes is not None and annotation in annotation.axes.texts:
                annotation.remove()
        self.annotations = {}
        self.series = []
        formatters = formatters or {}
        for ax in axes:
            annotation = ax.annotate("", xy=(0, 0), xytext=(12, 14),
                                     textcoords="offset points", fontsize=9, color=INK,
                                     bbox=dict(boxstyle="round,pad=0.6", fc="white", ec="#cbd8d8"),
                                     zorder=100, annotation_clip=False)
            annotation.set_visible(False)
            annotation.set_in_layout(False)
            self.annotations[ax] = annotation
            for line in ax.lines:
                # Reference lines use axes coordinates on one dimension.
                if line.get_transform() != ax.transData or not line.get_visible():
                    continue
                points = np.column_stack(line.get_data()).astype(float)
                if not len(points):
                    continue
                label = line.get_label()
                if label.startswith("_"):
                    continue
                self.series.append((ax, points, label, formatters.get(line)))
            for collection in ax.collections:
                label = collection.get_label()
                if label.startswith("_") or not hasattr(collection, "get_sizes"):
                    continue
                points = np.asarray(collection.get_offsets(), dtype=float)
                if len(points):
                    self.series.append((ax, points, label, formatters.get(collection)))

    def hide(self, event=None):
        changed = False
        for annotation in self.annotations.values():
            if annotation.get_visible():
                annotation.set_visible(False)
                changed = True
        if changed:
            self.fig.canvas.draw_idle()

    def on_move(self, event):
        if event.inaxes not in self.annotations or event.x is None or event.y is None:
            self.hide()
            return
        best, distance = None, 18.0 ** 2
        for ax, points, label, formatter in self.series:
            if ax != event.inaxes:
                continue
            pixels = ax.transData.transform(points)
            ds = np.sum((pixels - [event.x, event.y]) ** 2, axis=1)
            visible = (np.isfinite(ds) & (pixels[:, 0] >= ax.bbox.x0)
                       & (pixels[:, 0] <= ax.bbox.x1) & (pixels[:, 1] >= ax.bbox.y0)
                       & (pixels[:, 1] <= ax.bbox.y1))
            if not visible.any():
                continue
            ds = np.where(visible, ds, np.inf)
            index = int(np.argmin(ds))
            if ds[index] < distance:
                distance = ds[index]
                best = (ax, points[index], label, formatter, index)
        self.hide()
        if best is None:
            return
        ax, (x, y), label, formatter, index = best
        annotation = self.annotations[ax]
        annotation.xy = (x, y)
        right = event.x > (ax.bbox.x0 + ax.bbox.x1) / 2
        top = event.y > (ax.bbox.y0 + ax.bbox.y1) / 2
        annotation.set_position((-12 if right else 12, -14 if top else 14))
        annotation.set_ha("right" if right else "left")
        annotation.set_va("top" if top else "bottom")
        annotation.set_text(formatter(index) if formatter else
                            f"{label}\n{ax.get_xlabel()}: {x:,.6g}\nValue: {y:.8g}")
        annotation.set_visible(True)
        self.fig.canvas.draw_idle()


def enable_hover(fig, axes, formatters=None):
    if not hasattr(fig, "_dashboard_hover"):
        fig._dashboard_hover = HoverInspector(fig)
    fig._dashboard_hover.bind(axes, formatters)
