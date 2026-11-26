"""Render the repository's vector-native research banner."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(14, 4.5), dpi=160)
fig.patch.set_facecolor("#0b1423")
ax.set_facecolor("#0b1423")
ax.set_xlim(0, 1400)
ax.set_ylim(450, 0)
ax.axis("off")
for x in range(0, 1401, 35):
    ax.plot([x, x], [0, 450], color="#1c2b3d", alpha=.35, linewidth=.4)
for y in range(0, 451, 35):
    ax.plot([0, 1400], [y, y], color="#1c2b3d", alpha=.35, linewidth=.4)
ax.text(56, 56, "ONE WRONG EDGE  /  RESEARCH NOTEBOOK", color="#57d9c3", fontsize=10, fontweight="bold", family="DejaVu Sans Mono")
ax.text(50, 161, "One Wrong Edge", color="#f3f7fc", fontsize=48, fontweight="bold", family="DejaVu Sans")
ax.text(56, 215, "Which relation helps which question?", color="#d5e2f1", fontsize=19)
ax.text(56, 251, "Mathematical evidence retrieval, studied through interventions.", color="#9aafc8", fontsize=12)
chip = FancyBboxPatch((56, 297), 435, 38, boxstyle="round,pad=0,rounding_size=8", linewidth=.7, edgecolor="#32546d", facecolor="#142b3d")
ax.add_patch(chip)
ax.text(73, 322, "SOURCE-BACKED  /  QUERY-CONDITIONED  /  AUDITABLE", color="#bce8df", fontsize=9, family="DejaVu Sans Mono")
ax.text(56, 410, "METAMATH  ·  RETRIEVAL  ·  SIGNED EFFECTS  ·  GENERALIZATION", color="#748ea9", fontsize=9, family="DejaVu Sans Mono")
nodes = {"q": (880, 218), "a": (1030, 103), "b": (1030, 333), "c": (1215, 170), "d": (1260, 330)}
for source, destination in (("q", "a"), ("q", "b"), ("a", "c"), ("b", "c"), ("c", "d")):
    x1, y1 = nodes[source]
    x2, y2 = nodes[destination]
    ax.plot([x1, x2], [y1, y2], color="#4b8c9b", linewidth=2, alpha=.8, zorder=1)
ax.plot([1030, 1260], [103, 330], color="#ffb65c", linewidth=3, linestyle=(0, (4, 4)), zorder=2)
for name, (x, y) in nodes.items():
    color = "#57d9c3" if name == "q" else "#adc8e8"
    ax.add_patch(Circle((x, y), 33 if name == "q" else 27, facecolor="#142b3d", edgecolor=color, linewidth=2, zorder=3))
    ax.text(x, y + 5, {"q": "q", "a": "A", "b": "B", "c": "C", "d": "D"}[name], color="#edf6ff", fontsize=16, ha="center", zorder=4)
ax.text(879, 275, "QUESTION", color="#57d9c3", fontsize=8, ha="center", family="DejaVu Sans Mono")
ax.text(1080, 70, "PROOF-USE GRAPH", color="#8da7c2", fontsize=9, family="DejaVu Sans Mono")
ax.text(1114, 243, "ONE EDGE", color="#ffbf75", fontsize=9, rotation=-43, fontweight="bold", zorder=5)
ax.text(1082, 410, "Conceptual illustration", color="#748ea9", fontsize=8, ha="center")
fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
(ROOT / "assets").mkdir(exist_ok=True)
fig.savefig(ROOT / "assets/banner.png", facecolor=fig.get_facecolor())
fig.savefig(ROOT / "assets/banner.svg", facecolor=fig.get_facecolor())
plt.close(fig)
print("Rendered assets/banner.png and assets/banner.svg")
