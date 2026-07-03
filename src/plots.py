"""Gera as curvas dos experimentos de cascata edge→cloud a partir dos CSVs.

Lê todos os `results/cascade_mobilenet_b<N>_<estrategia>_best_<calib>_<ts>.csv`
(usa o CSV mais recente quando há mais de um por configuração) e produz:

  1. plots/acc_vs_ptar.png       — acurácia global (±IC95%) × p_tar, grade
                                    (estratégia × calib), uma linha por nº de ramos.
  2. plots/edge_fraction.png     — fração das amostras resolvida na borda × p_tar,
                                    mesma grade (nuvem = 100% − borda).
  3. plots/tradeoff_acc_vs_edge.png — curva de trade-off: acurácia global ×
                                    fração na borda, parametrizada por p_tar.
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
    r"cascade_mobilenet_b(\d+)_(.+)_best_(none|overall|branches)_"
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
    """Mapeia (n_branches, estrategia, calib) -> DataFrame do CSV mais recente."""
    latest = {}  # chave -> (timestamp, path)
    for fname in os.listdir(RESULTS_DIR):
        m = PATTERN.match(fname)
        if not m:
            continue
        n, strat, calib, ts = int(m.group(1)), m.group(2), m.group(3), m.group(4)
        key = (n, strat, calib)
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


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = discover()
    if not data:
        raise SystemExit(f"Nenhum CSV no padrão novo encontrado em {RESULTS_DIR}/")

    configs = defaultdict(list)
    for (n, strat, calib) in sorted(data):
        configs[strat].append((n, calib))
    print("Configurações encontradas:")
    for strat, items in configs.items():
        ramos = sorted({n for n, _ in items})
        calibs = sorted({c for _, c in items})
        print(f"  {strat}: ramos={ramos} calibs={calibs}")
    print()

    generated = [
        plot_acc_vs_ptar(data),
        plot_edge_fraction(data),
        plot_tradeoff(data, "equal"),
        plot_calib_effect(data, "equal"),
    ]
    for strat in sorted({k[1] for k in data}):
        generated.append(plot_exit_distribution(data, strat, "none"))

    print("Plots gerados:")
    for g in generated:
        if g:
            print(f"  {g}")


if __name__ == "__main__":
    main()
