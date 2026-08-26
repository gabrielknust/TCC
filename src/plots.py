"""Gera as curvas dos experimentos de cascata edge→cloud a partir dos CSVs.

Lê todos os `results/cascade_mobilenet_b<N>_<estrategia>_best_<calib>_<ts>.csv`
(usa o CSV mais recente quando há mais de um por configuração) e produz:

  1. plots/acc_vs_ptar.png       — acurácia global (±IC95%) × p_tar, grade
                                    (estratégia × calib), uma linha por nº de ramos.
  2. plots/edge_fraction.png     — fração das amostras resolvida na borda × p_tar,
                                    mesma grade (nuvem = 100% − borda).
  3. plots/tradeoff_acc_vs_edge.png — curva de trade-off: acurácia global ×
                                    fração na borda, parametrizada por p_tar.
  3b. plots/tradeoff_acc_vs_macs.png — acurácia global × MACs médios por
                                    amostra (a métrica honesta de custo).
  4. plots/calib_effect.png      — efeito da calibração na acurácia global,
                                    um subplot por nº de ramos (estratégia equal).
  5. plots/exit_distribution_<estrategia>_<calib>.png — distribuição de exits
                                    (barras empilhadas, borda em azul / nuvem em
                                    laranja) × p_tar, um subplot por nº de ramos.

Uso:
    uv run python -m src.plots
"""

import os
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"
OUT_DIR = "plots"

# Apenas o padrão novo (modelo paramétrico). Ignora CSVs legados cascade_b2_*.csv.
PATTERN = re.compile(
    r"cascade_mobilenet_(?:([A-Za-z0-9.\-]+)_)?b(\d+)_(.+)_best_(none|overall|branches)_"
    r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})\.csv$"
)

CALIB_ORDER = ["none", "overall", "branches"]
CALIB_LABEL = {"none": "sem calib.", "overall": "overall", "branches": "per-branch"}

# --- Paleta (referência validada da skill de dataviz, modo claro) ---
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
# Categórica em ordem fixa (slots 1-4) → identifica o nº de ramos (b2..b5).
CAT = {2: "#2a78d6", 3: "#1baf7a", 4: "#eda100", 5: "#008300"}
# Comparação de calibração: baseline neutro + 2 hues estáveis por entidade.
CALIB_COLOR = {"none": "#898781", "overall": "#2a78d6", "branches": "#e34948"}
CALIB_STYLE = {"none": "--", "overall": "-", "branches": "-."}
# Rampa ordinal azul (profundidade dos ramos na borda) + laranja para a nuvem.
ORDINAL_BLUE = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
CLOUD_COLOR = "#eb6834"

P_TAR_TICKS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


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


def plot_line_ci(ax, df, xcol, ycol, cicol, color, label, ls="-"):
    """Linha com marcadores e banda de IC95% (quando a coluna existe)."""
    x, y = df[xcol], df[ycol]
    if cicol in df.columns:
        ax.fill_between(x, y - df[cicol], y + df[cicol], color=color,
                        alpha=0.15, linewidth=0)
    ax.plot(x, y, marker="o", ms=4.5, lw=2, ls=ls, color=color, label=label,
            markeredgecolor=SURFACE, markeredgewidth=0.7)


def discover():
    """Mapeia (variante, n_branches, estrategia, calib) -> DataFrame do CSV mais recente."""
    latest = {}  # chave -> (timestamp, path)
    for fname in os.listdir(RESULTS_DIR):
        m = PATTERN.match(fname)
        if not m:
            continue
        var, n, strat, calib, ts = (m.group(1) or "", int(m.group(2)),
                                    m.group(3), m.group(4), m.group(5))
        key = (var, n, strat, calib)
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, os.path.join(RESULTS_DIR, fname))
    return {k: pd.read_csv(v[1]) for k, v in latest.items()}


def _grid_by_strategy_calib(data, ycol, cicol, ylabel, title, fname):
    """Grade estratégia × calib; cada subplot tem 1 linha por nº de ramos."""
    strategies = sorted({k[1] for k in data})
    n_branches = sorted({k[0] for k in data})

    fig, axes = new_fig(len(strategies), len(CALIB_ORDER), sharex=True, sharey=True)
    for r, strat in enumerate(strategies):
        for c, calib in enumerate(CALIB_ORDER):
            ax = axes[r][c]
            for n in n_branches:
                df = data.get((n, strat, calib))
                if df is None:
                    continue
                plot_line_ci(ax, df, "p_tar", ycol, cicol, CAT[n], f"{n} ramos")
            if r == 0:
                ax.set_title(f"calib: {CALIB_LABEL[calib]}", fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{strat}\n{ylabel}", fontsize=9)
            if r == len(strategies) - 1:
                ax.set_xlabel("p_tar (limiar de confiança)", fontsize=9)
            ax.set_xticks(P_TAR_TICKS)
            ax.tick_params(axis="x", labelrotation=45)
    leg = axes[0][0].legend(fontsize=8, loc="lower right", frameon=False)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle(title, fontsize=13, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(OUT_DIR, fname)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_acc_vs_ptar(data):
    return _grid_by_strategy_calib(
        data, "acc_global_%", "acc_global_ci95_%", "acurácia global (%)",
        "Acurácia global (±IC95%) × p_tar por nº de ramos", "acc_vs_ptar.png",
    )


def plot_edge_fraction(data):
    return _grid_by_strategy_calib(
        data, "frac_edge_%", "_none_", "% resolvido na borda",
        "Fração das amostras resolvida na borda × p_tar (nuvem = restante)",
        "edge_fraction.png",
    )


def plot_tradeoff(data, strategy="equal"):
    """Curva de trade-off: acurácia global × fração na borda, parametrizada
    por p_tar (cada ponto é um limiar; extremos anotados)."""
    n_branches = sorted({k[0] for k in data if k[1] == strategy})
    if not n_branches:
        return None
    fig, axes = new_fig(1, len(CALIB_ORDER), w=4.4, h=3.8, sharex=True, sharey=True)
    for c, calib in enumerate(CALIB_ORDER):
        ax = axes[0][c]
        for n in n_branches:
            df = data.get((n, strategy, calib))
            if df is None or "frac_edge_%" not in df.columns:
                continue
            df = df.sort_values("p_tar")
            ax.plot(df["frac_edge_%"], df["acc_global_%"], marker="o", ms=4.5,
                    lw=2, color=CAT[n], label=f"{n} ramos",
                    markeredgecolor=SURFACE, markeredgewidth=0.7)
            # Rótulo direto seletivo: só o extremo p_tar=0.95 (o canto direito,
            # onde todas as curvas convergem para p_tar=0.50, ficaria poluído).
            row = df.iloc[-1]
            ax.annotate(f"{row['p_tar']:.2f}",
                        (row["frac_edge_%"], row["acc_global_%"]),
                        textcoords="offset points", xytext=(-2, -11),
                        fontsize=6.5, color=MUTED)
        ax.set_title(f"calib: {CALIB_LABEL[calib]}", fontsize=10)
        ax.set_xlabel("% resolvido na borda", fontsize=9)
        if c == 0:
            ax.set_ylabel("acurácia global (%)", fontsize=9)
    leg = axes[0][0].legend(fontsize=8, loc="lower left", frameon=False)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle(
        f"Trade-off acurácia × permanência na borda (estratégia: {strategy};"
        " pontos = p_tar 0.50–0.95)", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "tradeoff_acc_vs_edge.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_tradeoff_macs(data, strategy="equal"):
    """Acurácia global × custo médio (MACs) por amostra, parametrizada por p_tar.

    A métrica honesta do trade-off: pondera cada saída pelo seu custo
    acumulado real, em vez de tratar todo ramo como "borda barata".
    """
    n_branches = sorted({k[0] for k in data if k[1] == strategy})
    if not n_branches:
        return None
    fig, axes = new_fig(1, len(CALIB_ORDER), w=4.4, h=3.8, sharex=True, sharey=True)
    plotted = False
    for c, calib in enumerate(CALIB_ORDER):
        ax = axes[0][c]
        for n in n_branches:
            df = data.get((n, strategy, calib))
            if df is None or "macs_mean_M" not in df.columns:
                continue
            plotted = True
            df = df.sort_values("p_tar")
            ax.plot(df["macs_mean_M"], df["acc_global_%"], marker="o", ms=4.5,
                    lw=2, color=CAT[n], label=f"{n} ramos",
                    markeredgecolor=SURFACE, markeredgewidth=0.7)
            row = df.iloc[-1]
            ax.annotate(f"{row['p_tar']:.2f}",
                        (row["macs_mean_M"], row["acc_global_%"]),
                        textcoords="offset points", xytext=(4, -9),
                        fontsize=6.5, color=MUTED)
        ax.set_title(f"calib: {CALIB_LABEL[calib]}", fontsize=10)
        ax.set_xlabel("MACs médios por amostra (M)", fontsize=9)
        if c == 0:
            ax.set_ylabel("acurácia global (%)", fontsize=9)
    if not plotted:
        plt.close(fig)
        return None
    leg = axes[0][0].legend(fontsize=8, loc="lower right", frameon=False)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle(
        f"Trade-off acurácia × custo computacional (estratégia: {strategy};"
        " pontos = p_tar 0.50–0.95)", fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "tradeoff_acc_vs_macs.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


# Cor = arquitetura (a entidade); traço = critério de seleção do checkpoint.
VARIANT_ARCH = [
    ("", "padrão (mapa 2×2)", "#2a78d6"),
    ("s14", "v3a (4×4)", "#eda100"),
    ("s7-14", "v3b (8×8)", "#008300"),
]
SELECT_STYLE = [("", "val loss", "-"), ("-acc", "val acc", "--")]


def plot_variants_macs(data, n=3, strategy="equal"):
    fig, axes = new_fig(1, len(CALIB_ORDER), w=4.4, h=3.8, sharex=True, sharey=True)
    plotted = False
    for c, calib in enumerate(CALIB_ORDER):
        ax = axes[0][c]
        for arch, arch_label, color in VARIANT_ARCH:
            for suffix, sel_label, ls in SELECT_STYLE:
                var = f"{arch}{suffix}" if arch else suffix.lstrip("-")
                df = data.get((var, n, strategy, calib))
                if df is None or "macs_mean_M" not in df.columns:
                    continue
                plotted = True
                df = df.sort_values("p_tar")
                ax.plot(df["macs_mean_M"], df["acc_global_%"], marker="o", ms=4,
                        lw=2, ls=ls, color=color,
                        label=f"{arch_label} — {sel_label}",
                        markeredgecolor=SURFACE, markeredgewidth=0.7)
                row = df.iloc[-1]
                ax.annotate(f"{row['p_tar']:.2f}",
                            (row["macs_mean_M"], row["acc_global_%"]),
                            textcoords="offset points", xytext=(4, -9),
                            fontsize=6.5, color=MUTED)
        ax.set_xscale("log", base=2)
        ax.set_xticks([8, 16, 32, 64])
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_title(f"calib: {CALIB_LABEL[calib]}", fontsize=10)
        ax.set_xlabel("MACs médios por amostra (M, escala log)", fontsize=9)
        if c == 0:
            ax.set_ylabel("acurácia global (%)", fontsize=9)
    if not plotted:
        plt.close(fig)
        return None
    leg = axes[0][0].legend(fontsize=7, loc="lower right", frameon=False)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle(
        f"Fronteira acurácia × custo: arquitetura (cor) × seleção de checkpoint"
        f" (traço) — b{n}, {strategy}; pontos = p_tar 0.50–0.95",
        fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "variants_acc_vs_macs.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_calib_effect(data, strategy="equal"):
    """1 subplot por nº de ramos; 1 linha por modo de calib. Estratégia fixa."""
    n_branches = sorted({k[0] for k in data if k[1] == strategy})
    if not n_branches:
        return None
    fig, axes = new_fig(1, len(n_branches), w=4.0, h=3.6, sharex=True, sharey=True)
    for c, n in enumerate(n_branches):
        ax = axes[0][c]
        for calib in CALIB_ORDER:
            df = data.get((n, strategy, calib))
            if df is None:
                continue
            plot_line_ci(ax, df, "p_tar", "acc_global_%", "acc_global_ci95_%",
                         CALIB_COLOR[calib], CALIB_LABEL[calib], CALIB_STYLE[calib])
        ax.set_title(f"{n} ramos", fontsize=10)
        ax.set_xlabel("p_tar", fontsize=9)
        ax.set_xticks(P_TAR_TICKS)
        ax.tick_params(axis="x", labelrotation=45)
        if c == 0:
            ax.set_ylabel("acurácia global (%)", fontsize=9)
    leg = axes[0][0].legend(fontsize=8, loc="lower right", frameon=False)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle(
        f"Efeito da calibração na acurácia global (±IC95%; estratégia: {strategy})",
        fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = os.path.join(OUT_DIR, "calib_effect.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_exit_distribution(data, strategy="equal", calib="none"):
    """Barras empilhadas: % das amostras que saem em cada exit × p_tar.

    Ramos da borda em rampa azul ordinal (raso→profundo); nuvem em laranja.
    """
    n_branches = sorted({k[0] for k in data if k[1] == strategy and k[2] == calib})
    if not n_branches:
        return None
    fig, axes = new_fig(1, len(n_branches), w=4.0, h=3.8, sharey=True)
    handles = {}  # label -> handle; cor fixa por identidade do ramo entre painéis
    for c, n in enumerate(n_branches):
        ax = axes[0][c]
        df = data[(n, strategy, calib)]
        frac_cols = sorted(
            [col for col in df.columns
             if col.startswith("frac_exit") and col.endswith("_%")],
            key=lambda s: int(re.search(r"\d+", s).group()),
        )
        n_exits = len(frac_cols)
        x = [f"{p:.2f}" for p in df["p_tar"]]
        bottom = [0.0] * len(df)
        for e, col in enumerate(frac_cols):
            is_final = e == n_exits - 1
            label = "final (nuvem)" if is_final else f"ramo {e + 1}"
            color = CLOUD_COLOR if is_final else ORDINAL_BLUE[e]
            bars = ax.bar(x, df[col], bottom=bottom, color=color,
                          width=0.72, edgecolor=SURFACE, linewidth=1.0)
            handles.setdefault(label, bars[0])
            bottom = [b + v for b, v in zip(bottom, df[col])]
        ax.set_title(f"{n} ramos", fontsize=10)
        ax.set_xlabel("p_tar", fontsize=9)
        ax.grid(False, axis="x")
        if c == 0:
            ax.set_ylabel("% das amostras", fontsize=9)
        ax.tick_params(axis="x", labelrotation=45)
    order = [f"ramo {i + 1}" for i in range(max(n_branches))] + ["final (nuvem)"]
    order = [lb for lb in order if lb in handles]
    leg = fig.legend([handles[lb] for lb in order], order, fontsize=8,
                     ncol=len(order), frameon=False, loc="lower center")
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle(
        f"Onde cada amostra é decidida × p_tar (azul = borda, laranja = nuvem;"
        f" estratégia: {strategy}, calib: {CALIB_LABEL[calib]})",
        fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0.06, 1, 0.93])
    out = os.path.join(OUT_DIR, f"exit_distribution_{strategy}_{calib}.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


GROUP_TITLE = {
    "D": "Arquitetura — vs. padrão (mapa 2×2)",
    "F": "Nº de ramos — vs. b2",
    "A": "Peso da loss — linspace vs. equal",
    "E": "Seleção de checkpoint — val acc vs. val loss",
    "B": "Calibração — overall vs. per-branch",
    "C": "Calibração — overall vs. sem calibrar",
}
GROUP_ORDER = ["D", "F", "A", "E", "B", "C"]


def _forest_row_label(group, label):
    if group == "D":
        return label.replace(" vs padrao", "").replace("4x4", "4×4").replace("8x8", "8×8")
    if group == "F":
        return label.split(" vs ")[0]
    return label.split(":")[0]


def _fmt_p(p):
    if p < 1e-4:
        return "p < 0.0001"
    if p < 1e-3:
        return f"p = {p:.4f}"
    return f"p = {p:.3f}"


def plot_forest(path="results/stats_comparisons.csv"):
    """Forest plot das comparações pareadas: Δ acurácia ±IC95% e veredito."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)

    rows = []
    for g in GROUP_ORDER:
        sub = df[df["group"] == g]
        if sub.empty:
            continue
        rows.append(("header", GROUP_TITLE[g], None))
        for _, r in sub.sort_values("delta", ascending=False).iterrows():
            rows.append(("row", _forest_row_label(g, r["label"]), r))

    fig, ax = plt.subplots(figsize=(9.0, 0.26 * len(rows) + 1.6))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)

    y = len(rows)
    ticks, labels = [], []
    for kind, text, r in rows:
        y -= 1
        ticks.append(y)
        if kind == "header":
            labels.append(text)
            ax.axhline(y + 0.55, color=AXIS, linewidth=0.8)
            continue
        labels.append(text)
        sig = bool(r["significant"])
        color = INK if sig else MUTED
        ax.errorbar(r["delta"], y, xerr=r["ci95"], fmt="none",
                    ecolor=color, elinewidth=2.0, capsize=0, zorder=2)
        ax.plot(r["delta"], y, "o", markersize=8, zorder=3,
                color=color if sig else SURFACE,
                markeredgecolor=color, markeredgewidth=2.0)
        ax.annotate(_fmt_p(r["p_value"]), xy=(1.02, y),
                    xycoords=("axes fraction", "data"), va="center",
                    fontsize=7.5, color=INK_2 if sig else MUTED)

    ax.axvline(0, color=INK_2, linewidth=1.4, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=8.5)
    for tick, (kind, _, _) in zip(ax.get_yticklabels(), rows):
        tick.set_color(INK if kind == "header" else INK_2)
        if kind == "header":
            tick.set_fontweight("bold")
            tick.set_horizontalalignment("left")
            tick.set_x(-0.34)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel("Δ acurácia global (pontos percentuais), ±IC95% pareado",
                  fontsize=9, color=INK_2)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="-", color=INK, markersize=8,
                   linewidth=2.0, label="significativo (p < 0.05)"),
        plt.Line2D([], [], marker="o", linestyle="-", color=MUTED, markersize=8,
                   linewidth=2.0, markerfacecolor=SURFACE, markeredgewidth=2.0,
                   label="indistinguível de ruído"),
    ]
    leg = ax.legend(handles=handles, fontsize=8, frameon=False, ncol=2,
                    loc="upper left", bbox_to_anchor=(-0.34, -0.055 * 30 / len(rows)))
    for t in leg.get_texts():
        t.set_color(INK_2)

    fig.text(0.012, 0.985, "O que sobrevive ao teste pareado de McNemar",
             fontsize=12, color=INK, ha="left", va="top")
    fig.text(0.012, 0.962, "à direita do zero = a primeira configuração de cada par vence",
             fontsize=9, color=MUTED, ha="left", va="top")
    fig.tight_layout(rect=[0, 0.035, 0.88, 0.945])
    out = os.path.join(OUT_DIR, "forest_mcnemar.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = discover()
    if not data:
        raise SystemExit(f"Nenhum CSV no padrão novo encontrado em {RESULTS_DIR}/")
    base = {k[1:]: df for k, df in data.items() if k[0] == ""}

    configs = defaultdict(list)
    for (var, n, strat, calib) in sorted(data):
        configs[(var, strat)].append((n, calib))
    print("Configurações encontradas:")
    for (var, strat), items in configs.items():
        ramos = sorted({n for n, _ in items})
        calibs = sorted({c for _, c in items})
        tag = f"{var}/" if var else ""
        print(f"  {tag}{strat}: ramos={ramos} calibs={calibs}")
    print()

    generated = [
        plot_acc_vs_ptar(base),
        plot_edge_fraction(base),
        plot_tradeoff(base, "equal"),
        plot_tradeoff_macs(base, "equal"),
        plot_variants_macs(data),
        plot_calib_effect(base, "equal"),
    ]
    for strat in sorted({k[1] for k in base}):
        generated.append(plot_exit_distribution(base, strat, "none"))
    generated.append(plot_latency())
    generated.append(plot_forest())

    print("Plots gerados:")
    for g in generated:
        if g:
            print(f"  {g}")




def plot_latency(pattern="results/latency/*.csv"):
    """Latencia real x acuracia, por ponto de particao; baseline so'-nuvem."""
    import glob
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    fig, axes = new_fig(1, len(files), w=5.0, h=4.0, sharey=True)
    for c, path in enumerate(files):
        ax = axes[0][c]
        df = pd.read_csv(path)
        ee = df[df["mode"] == "early_exit"]
        g = (ee.groupby(["nr_branch_edge", "p_tar"])
               .agg(lat=("total_ms", "mean"), acc=("correct", "mean"))
               .reset_index())
        g["acc"] *= 100
        for i, nr in enumerate(sorted(g["nr_branch_edge"].unique())):
            sub = g[g["nr_branch_edge"] == nr].sort_values("p_tar")
            color = ORDINAL_BLUE[min(i, len(ORDINAL_BLUE) - 1)]
            ax.plot(sub["lat"], sub["acc"], marker="o", ms=5, lw=2, color=color,
                    label=f"{nr} ramo(s) na borda",
                    markeredgecolor=SURFACE, markeredgewidth=0.7)
        oc = df[df["mode"] == "only_cloud"]
        if len(oc):
            ax.axvline(oc["total_ms"].mean(), ls="--", lw=1.6, color=CLOUD_COLOR)
            ax.annotate("só-nuvem", (oc["total_ms"].mean(), ax.get_ylim()[0]),
                        textcoords="offset points", xytext=(-42, 12),
                        fontsize=8, color=CLOUD_COLOR)
        link = "enlace simulado" if "link" in os.path.basename(path) else "localhost"
        ax.set_title(link, fontsize=10)
        ax.set_xlabel("latência média fim-a-fim (ms)", fontsize=9)
        if c == 0:
            ax.set_ylabel("acurácia (%)", fontsize=9)
    leg = axes[0][0].legend(fontsize=8, loc="lower right", frameon=False)
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.suptitle("Latência real × acurácia por ponto de partição borda/nuvem",
                 fontsize=12, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUT_DIR, "latency_vs_acc.png")
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out


if __name__ == "__main__":
    main()
