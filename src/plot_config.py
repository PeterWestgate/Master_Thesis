#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
Shared plot configuration for all thesis figures.
Import this in every script that generates figures.

Usage:
    from src.plot_config import MODEL_COLORS, CORR_COLORS, FIGURE_DEFAULTS, apply_defaults
'''

import matplotlib.pyplot as plt
import matplotlib as mpl
from src.figure_style import apply_apa_style, APA_LABEL, APA_TICK, APA_TICK_ROI, APA_STAR, APA_ANNOT  # noqa: F401

# ── Colorblind-friendly palette (Wong 2011) ───────────────────────────────────

# Theoretical model colors
MODEL_COLORS = {
    'Category': '#0072B2',   # blue
    'Animacy':  '#D55E00',   # vermillion
    'Gist':     '#009E73',   # green
}

# Same keys used in scripts where model names have '80'/'40' suffix stripped
# Access with e.g. MODEL_COLORS.get(name.replace('80','').capitalize())

# Correlation type colors (used in behavioural partial correlation plot)
CORR_COLORS = {
    'full':        '#333333',   # near-black
    'semipartial': '#999999',   # mid-grey
    'partial':     '#CCCCCC',   # light grey
}

# Brain-behavior color
BRAIN_BEH_COLOR = '#CC79A7'   # reddish purple (Wong)

# Bar plot style defaults
BAR_DEFAULTS = {
    'alpha':     0.85,
    'edgecolor': 'none',
    'error_kw':  {'elinewidth': 1.5, 'capsize': 0, 'ecolor': '#555555'},
}

# Noise ceiling
NOISE_CEIL_COLOR = '#E69F00'   # orange (Wong)

# Figure-level defaults
FIGURE_DEFAULTS = {
    'dpi':        300,
    'fontsize':   10,
    'title_size': 11,
    'tick_size':  9,
    'legend_fontsize': 9,
}


def apply_defaults():
    """Call once at the top of a plotting script to apply global rcParams.

    Applies APA 7 typography (Arial/DejaVu Sans, standardised sizes) plus
    the project colour and spine defaults.
    """
    apply_apa_style()   # APA 7 font family and sizes
