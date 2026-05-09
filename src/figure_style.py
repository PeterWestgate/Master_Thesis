#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
APA 7 typography for all thesis figures.

Import and call apply_apa_style() at the top of every figure-generating script,
BEFORE creating any figure.  apply_defaults() in plot_config already calls this
automatically, so scripts that use apply_defaults() are covered.

Constants (import these to replace hard-coded fontsize= values):
    APA_LABEL   = 12   axis labels and panel titles
    APA_TICK    = 10   tick labels, legend text, colourbar labels, annotations
    APA_TICK_ROI = 9   ROI tick labels on 22-ROI axes (never below 8 pt)
    APA_STAR    = 14   significance asterisks (*, **, ***)
    APA_ANNOT   = 10   descriptive / secondary annotations

Font family: Arial → DejaVu Sans → sans-serif (fallback chain).
Math text uses the DejaVu Sans math set for APA-compatible rendering without
forcing Arial into LaTeX mode.
'''

import matplotlib as mpl

# ── Font-size constants ────────────────────────────────────────────────────────

APA_LABEL    = 12   # axis labels, panel titles
APA_TICK     = 10   # tick labels, legend, colourbar
APA_TICK_ROI =  9   # 22-ROI axes (min allowed: 8 pt)
APA_STAR     = 14   # significance asterisks
APA_ANNOT    = 10   # descriptive annotations

_FONT_FAMILY = ['Arial', 'DejaVu Sans', 'sans-serif']


# ── Main entry point ──────────────────────────────────────────────────────────

def apply_apa_style():
    """Set matplotlib rcParams for APA 7 figure typography.

    Call once per script before any figure is created.
    """
    mpl.rcParams.update({
        # Font family
        'font.family':           'sans-serif',
        'font.sans-serif':       _FONT_FAMILY,
        'font.weight':           'normal',

        # Axes labels  (12 pt, regular)
        'axes.labelsize':        APA_LABEL,
        'axes.labelweight':      'normal',

        # Panel titles (12 pt, regular)
        'axes.titlesize':        APA_LABEL,
        'axes.titleweight':      'normal',
        'figure.titlesize':      APA_LABEL,
        'figure.titleweight':    'normal',

        # Tick labels  (10 pt)
        'xtick.labelsize':       APA_TICK,
        'ytick.labelsize':       APA_TICK,

        # Legend       (10 pt)
        'legend.fontsize':       APA_TICK,
        'legend.title_fontsize': APA_TICK,

        # Math: use DejaVu Sans math (closest APA-compatible sans-serif)
        'mathtext.fontset':      'dejavusans',

        # Unchanged from plot_config defaults
        'axes.spines.top':       False,
        'axes.spines.right':     False,
        'figure.dpi':            100,
    })


# ── Nilearn / post-hoc font fix ───────────────────────────────────────────────

def fix_figure_fonts(fig, tick_size=None):
    """Retroactively apply APA font family and sizes to all text in *fig*.

    Use this after nilearn plotting calls, which bypass rcParams for some
    internal text objects (colourbar tick labels, glass-brain annotations).

    Parameters
    ----------
    fig : matplotlib Figure
    tick_size : int, optional
        Override tick label size (defaults to APA_TICK).
    """
    _tick = tick_size if tick_size is not None else APA_TICK

    for ax in fig.get_axes():
        # Axis labels
        for txt in (ax.title, ax.xaxis.label, ax.yaxis.label):
            _set_font(txt, APA_LABEL)

        # Tick labels
        for tick in ax.get_xticklabels() + ax.get_yticklabels():
            _set_font(tick, _tick)

        # Inline text objects (annotations, asterisks, etc.)
        for txt in ax.texts:
            _set_font(txt, txt.get_fontsize())  # preserve existing size

        # Legend
        leg = ax.get_legend()
        if leg:
            for txt in leg.get_texts():
                _set_font(txt, APA_TICK)
            if leg.get_title():
                _set_font(leg.get_title(), APA_TICK)

        # Colourbar axes detected by absence of data
        try:
            if hasattr(ax, 'yaxis') and ax.get_ylabel():
                _set_font(ax.yaxis.label, APA_TICK)
        except Exception:
            pass

    # Figure-level suptitle
    for txt in fig.texts:
        _set_font(txt, txt.get_fontsize())


def _set_font(text_obj, size):
    """Set font family (and optionally size) on a single Text object."""
    if text_obj is None:
        return
    try:
        text_obj.set_fontfamily(_FONT_FAMILY)
        if size is not None:
            text_obj.set_fontsize(size)
    except Exception:
        pass
