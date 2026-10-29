"""Gera as figuras do README a partir dos CSVs consolidados por src.replicas.

Todas as curvas são a média dos treinos repetidos (checkpoint da última época):

  1. plots/tradeoff_acc_vs_macs.png — acurácia × MACs médios por amostra,
                                    um painel por esquema de pesos.
  2. plots/ptar_effect.png        — acurácia e fração na borda × p_tar.
  3. plots/calib_effect.png       — efeito da calibração com limiar fixo e
                                    no mesmo custo.
  4. plots/edge_accuracy.png      — acerto das amostras resolvidas na borda
                                    × p_tar, contra a diagonal acerto = p_tar.
  5. plots/exit_distribution.png  — onde as amostras saem no ponto de operação.
  6. plots/variants_acc_vs_macs.png — v3b contra a arquitetura padrão.
  7. plots/selection_history.png  — loss e acurácia de validação da saída final
                                    ao longo do treino.

Uso:
    RESULTS_DIR=results_v5 SAVE_DIR=saved_model_v5 uv run python -m src.plots
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

RESULTS_DIR = os.environ.get("RESULTS_DIR", "results")
SAVE_DIR = os.environ.get("SAVE_DIR", "saved_model_v5")
OUT_DIR = os.environ.get("PLOTS_DIR", "plots")

CALIB_ORDER = ["none", "overall", "branches"]
CALIB_LABEL = {"none": "sem calibração", "overall": "overall", "branches": "por saída"}

# --- Paleta (referência validada da skill de dataviz, modo claro) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categórica em ordem fixa (slots 1-4) → identifica o nº de ramos (b2..b5).
CAT = {2: "#2a78d6", 3: "#eb6834", 4: "#1baf7a", 5: "#eda100"}
V3B_COLOR = "#4a3aa7"
# Comparação de calibração: slots 1-3 da categórica, um por modo.
CALIB_COLOR = {"none": "#2a78d6", "overall": "#eb6834", "branches": "#1baf7a"}
CALIB_STYLE = {"none": "--", "overall": "-", "branches": "-."}
# Rampa ordinal azul (profundidade dos ramos na borda) + laranja para a nuvem.
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
CLOUD_COLOR = "#eb6834"
EDGE_COLOR = "#2a78d6"


def ink_on(hex_color):
    r, g, b = (int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5))
    return INK if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.55 else SURFACE

P_TAR_TICKS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.99]
LABEL_DY = {2: 0, 3: 0, 4: 6, 5: -6}


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.xaxis.label.set_color(INK_2)
    ax.yaxis.label.set_color(INK_2)
    ax.title.set_color(INK)


def new_fig(rows, cols, w=4.2, h=3.4, **kw):
    fig, axes = plt.subplots(rows, cols, figsize=(w * cols, h * rows),
                             squeeze=False, facecolor=SURFACE, **kw)
    for row in axes:
        for ax in row:
            style_ax(ax)
    return fig, axes


def legend(ax, **kw):
    leg = ax.legend(fontsize=8, frameon=False, **kw)
    for t in leg.get_texts():
        t.set_color(INK_2)
    return leg


def save(fig, name, title, rect=(0, 0, 1, 0.93)):
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout(rect=rect)
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def line(ax, x, y, color, label, ls="-"):
    ax.plot(x, y, marker="o", ms=5, lw=2, ls=ls, color=color, label=label,
            markeredgecolor=SURFACE, markeredgewidth=1)


def end_label(ax, x, y, text, dx=-6, dy=0, ha="right"):
    ax.annotate(text, (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=ha, va="center", fontsize=8, color=INK_2)


def curve(curves, config, mode="overall"):
    return curves[(curves["config"] == config) & (curves["mode"] == mode)].sort_values("p_tar")


def b0_point(ax, ops):
    b0 = ops[ops["config"] == "b0"].iloc[0]
    ax.scatter([b0["macs_mean"]], [b0["acc_mean"]], s=60, color=INK, zorder=4,
               edgecolor=SURFACE, linewidth=1.5)
    end_label(ax, b0["macs_mean"], b0["acc_mean"], "b0 (sem saídas)", dx=-8, dy=8)


def plot_tradeoff(curves, ops):
    fig, axes = new_fig(1, 2, w=4.6, h=3.8, sharey=True)
    for ax, w in zip(axes[0], ("equal", "linspace")):
        for n in (2, 3, 4, 5):
            df = curve(curves, f"b{n} {w}")
            line(ax, df["macs_mean"], df["acc_mean"], CAT[n], f"b{n}")
            end_label(ax, df["macs_mean"].iloc[0], df["acc_mean"].iloc[0], f"b{n}",
                      dy=LABEL_DY[n])
            op = ops[(ops["config"] == f"b{n} {w}") & (ops["mode"] == "overall")].iloc[0]
            ax.scatter([op["macs_mean"]], [op["acc_mean"]], s=120, facecolor="none",
                       edgecolor=CAT[n], linewidth=2, zorder=5)
        b0_point(ax, ops)
        ax.set_title(f"pesos {w}", fontsize=10)
        ax.set_xlabel("MACs médios por amostra (M)", fontsize=9)
    axes[0][0].set_ylabel("acurácia no teste (%)", fontsize=9)
    legend(axes[0][1], loc="lower right")
    return save(fig, "tradeoff_acc_vs_macs.png",
                "Acurácia × custo da cascata (média de 5 treinos, calibração overall; "
                "círculo vazado = ponto de operação)")


def plot_ptar(curves):
    fig, axes = new_fig(1, 2, w=4.4, h=3.6)
    for n in (2, 3, 4, 5):
        df = curve(curves, f"b{n} equal")
        line(axes[0][0], df["p_tar"], df["acc_mean"], CAT[n], f"b{n}")
        line(axes[0][1], df["p_tar"], df["frac_edge_mean"], CAT[n], f"b{n}")
        end_label(axes[0][0], df["p_tar"].iloc[0], df["acc_mean"].iloc[0], f"b{n}")
    axes[0][0].set_ylabel("acurácia no teste (%)", fontsize=9)
    axes[0][1].set_ylabel("% resolvido nas saídas antecipadas", fontsize=9)
    for ax in axes[0]:
        ax.set_xlabel("p_tar (limiar de confiança)", fontsize=9)
        ax.set_xticks(P_TAR_TICKS)
        ax.set_xlim(0.40, 1.0)
    legend(axes[0][0], loc="lower right")
    return save(fig, "ptar_effect.png",
                "Efeito do limiar (pesos equal, calibração overall, média de 5 treinos)")


def plot_calibration(curves, config="b5 equal"):
    fig, axes = new_fig(1, 2, w=4.4, h=3.6)
    for mode in CALIB_ORDER:
        df = curve(curves, config, mode)
        line(axes[0][0], df["p_tar"], df["acc_mean"], CALIB_COLOR[mode],
             CALIB_LABEL[mode], CALIB_STYLE[mode])
        line(axes[0][1], df["macs_mean"], df["acc_mean"], CALIB_COLOR[mode],
             CALIB_LABEL[mode], CALIB_STYLE[mode])
    axes[0][0].set_xlabel("p_tar (limiar de confiança)", fontsize=9)
    axes[0][0].set_xticks(P_TAR_TICKS)
    axes[0][0].set_title("mesmo limiar", fontsize=10)
    axes[0][1].set_xlabel("MACs médios por amostra (M)", fontsize=9)
    axes[0][1].set_title("mesmo custo", fontsize=10)
    for ax in axes[0]:
        ax.set_ylabel("acurácia no teste (%)", fontsize=9)
    legend(axes[0][1], loc="lower right")
    return save(fig, "calib_effect.png",
                f"Calibração no {config}: ganha com limiar fixo, empata no mesmo custo "
                "(média de 5 treinos)")


def plot_edge_accuracy(curves, configs=("b3 equal", "b5 equal")):
    fig, axes = new_fig(1, len(configs), w=4.4, h=3.6, sharey=True)
    for ax, config in zip(axes[0], configs):
        ax.plot([0.5, 0.99], [50, 99], ls=":", lw=1.5, color=MUTED)
        end_label(ax, 0.62, 62, "acerto = p_tar", dx=6, dy=-8, ha="left")
        for mode in CALIB_ORDER:
            df = curve(curves, config, mode)
            line(ax, df["p_tar"], df["acc_edge_mean"], CALIB_COLOR[mode],
                 CALIB_LABEL[mode], CALIB_STYLE[mode])
        ax.set_title(config, fontsize=10)
        ax.set_xlabel("p_tar (limiar de confiança)", fontsize=9)
        ax.set_xticks(P_TAR_TICKS)
    axes[0][0].set_ylabel("acerto das amostras resolvidas na borda (%)", fontsize=9)
    legend(axes[0][-1], loc="lower right")
    return save(fig, "edge_accuracy.png",
                "O limiar como garantia: acima da diagonal, a borda acerta pelo menos p_tar")


def plot_exit_distribution(ops):
    profile = pd.read_csv(os.path.join(RESULTS_DIR, "model_profile.csv")).set_index("arquitetura")
    fig, axes = new_fig(4, 2, w=4.6, h=1.9, sharex=True, sharey=True)
    for col, w in enumerate(("equal", "linspace")):
        for row, n in enumerate((2, 3, 4, 5)):
            ax = axes[row][col]
            ax.grid(axis="x", visible=False)
            r = ops[(ops["config"] == f"b{n} {w}") & (ops["mode"] == "overall")].iloc[0]
            fracs = [float(x) for x in r["saidas"].split(" / ")]
            macs = [float(x) for x in profile.loc[f"b{n}", "macs_acum_M"].split(" / ")]
            ax.bar(macs, fracs, width=1.1, color=[EDGE_COLOR] * n + [CLOUD_COLOR])
            for x, f in zip(macs, fracs):
                ax.annotate(f"{f:.0f}%" if f >= 1 else f"{f:.1f}%", (x, f),
                            textcoords="offset points", xytext=(0, 3), ha="center",
                            va="bottom", fontsize=8, color=INK_2)
            ax.set_title(f"b{n} {w}", fontsize=9, loc="left", color=INK)
            ax.set_xlim(0, 28.5)
            ax.set_ylim(0, 88)
            ax.set_yticks([0, 20, 40, 60, 80])
        axes[-1][col].set_xlabel("MACs acumulados até a saída (M)", fontsize=9)
    for row in axes:
        row[0].set_ylabel("% das amostras", fontsize=9)
    handles = [Patch(color=EDGE_COLOR, label="saída antecipada (borda)"),
               Patch(color=CLOUD_COLOR, label="saída final (nuvem), rede inteira com 26.6 M MACs")]
    leg = fig.legend(handles=handles, fontsize=8, frameon=False, ncol=2,
                     loc="lower center", bbox_to_anchor=(0.5, 0.0))
    for t in leg.get_texts():
        t.set_color(INK_2)
    return save(fig, "exit_distribution.png",
                "Onde as amostras saem no ponto de operação (calibração overall, média de 5 treinos)",
                rect=(0, 0.04, 1, 0.965))


def plot_variants(curves, ops):
    fig, axes = new_fig(1, 1, w=6.4, h=3.8)
    ax = axes[0][0]
    for config, color, label in (("b3 equal", CAT[3], "padrão (mapa final 2×2)"),
                                 ("v3b", V3B_COLOR, "v3b (mapa final 8×8)")):
        df = curve(curves, config)
        line(ax, df["macs_mean"], df["acc_mean"], color, label)
        end_label(ax, df["macs_mean"].iloc[0], df["acc_mean"].iloc[0], label.split(" (")[0])
    b0_point(ax, ops)
    ax.set_xlabel("MACs médios por amostra (M)", fontsize=9)
    ax.set_ylabel("acurácia no teste (%)", fontsize=9)
    legend(ax, loc="lower right")
    return save(fig, "variants_acc_vs_macs.png",
                "v3b contra a arquitetura padrão (b3 equal, calibração overall, média de 5 treinos)")


def plot_history(config="mobilenet_r1_b3_linspace"):
    h = pd.read_csv(os.path.join(SAVE_DIR, "history", f"{config}.csv"))
    best, last = h["val_loss_Final"].idxmin(), len(h) - 1
    growth = 100 * (h.loc[last, "val_loss_Final"] / h.loc[best, "val_loss_Final"] - 1)
    gain = h.loc[last, "val_acc_Final"] - h.loc[best, "val_acc_Final"]
    panels = (
        ("val_loss_Final", "loss de validação", (0.25, 0.45),
         [(best, "mínimo da loss, época {e}: {v:.3f}", 8, -9, "left"),
          (last, "época {e}: {v:.3f} (" + f"+{growth:.0f}%)", -8, 10, "right")]),
        ("val_acc_Final", "acurácia de validação (%)", (86, 94.5),
         [(best, "época {e}: {v:.1f}%", 8, -10, "left"),
          (last, "época {e}: {v:.1f}% (" + f"+{gain:.1f} pp)", -8, -12, "right")]),
    )
    fig, axes = new_fig(2, 1, w=6.4, h=2.3, sharex=True)
    for ax, (col, ylabel, ylim, marks) in zip(axes[:, 0], panels):
        ax.axvline(h.loc[best, "epoch"], color=AXIS, lw=1, zorder=1)
        ax.plot(h["epoch"], h[col], lw=2, color=ORDINAL_BLUE[-1], zorder=2)
        for i, text, dx, dy, ha in marks:
            e, v = h.loc[i, "epoch"], h.loc[i, col]
            ax.scatter([e], [v], s=50, color=ORDINAL_BLUE[-1], zorder=3,
                       edgecolor=SURFACE, linewidth=1.5)
            end_label(ax, e, v, text.format(e=e, v=v), dx=dx, dy=dy, ha=ha)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(*ylim)
    axes[-1][0].set_xlabel("época", fontsize=9)
    axes[-1][0].set_xlim(0, h["epoch"].max() + 5)
    return save(fig, "selection_history.png",
                "Loss e acurácia de validação da saída final (b3 linspace, treino 1)")


def main():
    curves = pd.read_csv(os.path.join(RESULTS_DIR, "replicas_curves.csv"))
    ops = pd.read_csv(os.path.join(RESULTS_DIR, "replicas_operating_points.csv"))
    for out in (plot_tradeoff(curves, ops), plot_ptar(curves), plot_calibration(curves),
                plot_edge_accuracy(curves), plot_exit_distribution(ops),
                plot_variants(curves, ops), plot_history()):
        print(out)


if __name__ == "__main__":
    main()
