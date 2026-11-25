"""Redraw README figures from the executed notebooks.

Reads the per-epoch Keras log stored in 4_model.ipynb; no raw telemetry is needed.

    python scripts/plot_results.py
"""

import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
OUT.mkdir(exist_ok=True)

INK, MUTED, ACCENT, COPPER, LINE = "#202c3a", "#536171", "#305b8a", "#b47b40", "#dce1e6"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LINE, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlecolor": INK, "savefig.dpi": 200,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})


def notebook_text(name):
    nb = json.loads((ROOT / name).read_text())
    chunks = []
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            chunks.append("".join(o.get("text", "")))
            chunks.append("".join(o.get("data", {}).get("text/plain", "")))
    return "\n".join(chunks)


def training_history():
    log = notebook_text("4_model.ipynb")
    rows = re.findall(r"accuracy: ([\d.]+) - loss: ([\d.]+) - val_accuracy: ([\d.]+) - val_loss: ([\d.]+)", log)
    acc, loss, vacc, vloss = (list(map(float, c)) for c in zip(*rows))
    epochs = range(1, len(acc) + 1)
    best = min(range(len(vloss)), key=vloss.__getitem__)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9))
    for ax, tr, va, title, fmt in ((a1, acc, vacc, "Accuracy", "{:.1%}"), (a2, loss, vloss, "Binary cross-entropy", "{:.3f}")):
        ax.plot(epochs, tr, color=ACCENT, lw=2, label="Training")
        ax.plot(epochs, va, color=COPPER, lw=2, label="Validation")
        ax.axvline(best + 1, color=MUTED, ls=":", lw=1)
        ax.annotate(f"best epoch {best + 1}\nval {fmt.format(va[best])}", (best + 1, va[best]),
                    xytext=(-12, 18 if ax is a1 else 30), textcoords="offset points", ha="right", fontsize=8, color=MUTED,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
        ax.set_title(title, loc="left")
        ax.set_xlabel("Epoch")
        ax.grid(axis="y", color=LINE, lw=0.6)
    a1.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    a1.legend(frameon=False, loc="lower right")
    fig.suptitle(f"MLP training with early stopping (patience 5, {len(acc)} epochs)", x=0.01, ha="left",
                 fontweight="bold", color=INK, y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "training_history.png")
    plt.close(fig)
    return {"epochs": len(acc), "best_epoch": best + 1, "best_val_accuracy": vacc[best], "best_val_loss": vloss[best]}


if __name__ == "__main__":
    print(json.dumps(training_history(), indent=2))
