"""Generate docs/img/scaling.png — peak RSS by corpus size, with HF OOM ceiling."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#222",
    "xtick.color": "#444",
    "ytick.color": "#444",
})

fig, ax = plt.subplots(figsize=(9, 5.2), dpi=140)

corpora = ["v1\n32.8 GB", "v2\n142 GB"]
peak_rss = [7.35, 28.24]
x = [0, 1]

bars = ax.bar(x, peak_rss, width=0.45, color="#3ddc84", edgecolor="#1f8c50", linewidth=1.5, zorder=3)

for bar, val in zip(bars, peak_rss):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.2f} GB",
            ha="center", va="bottom", fontsize=13, fontweight="bold", color="#1a1a1a")

hf_ceiling = 68
ax.axhline(hf_ceiling, color="#d63b3b", linestyle="--", linewidth=2, zorder=2)
ax.text(1.48, hf_ceiling + 1.5, "HF tokenizers OOM ceiling (#1546: 68 GB+ on 35 GB corpus)",
        ha="right", va="bottom", color="#d63b3b", fontsize=10.5, fontweight="bold")

machine = 96
ax.axhline(machine, color="#888", linestyle=":", linewidth=1.5, zorder=2)
ax.text(1.48, machine + 1.5, "Single consumer machine: 96 GB RAM",
        ha="right", va="bottom", color="#666", fontsize=10)

ax.set_xticks(x)
ax.set_xticklabels(corpora, fontsize=12)
ax.set_ylabel("Peak RSS (GB)", fontsize=12)
ax.set_ylim(0, 110)
ax.set_title("WiseTok peak memory vs corpus size  (24,320 merges, same machine)",
             fontsize=13.5, fontweight="bold", pad=14, color="#1a1a1a")

ax.yaxis.grid(True, linestyle="-", linewidth=0.6, color="#e6e6e6", zorder=1)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig("/home/ezel/Development/WiseTok/docs/img/scaling.png",
            dpi=140, bbox_inches="tight", facecolor="white")
print("wrote scaling.png")
