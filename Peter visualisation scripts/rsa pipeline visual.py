import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from src.plot_config import apply_defaults, NOISE_CEIL_COLOR
from src.figure_style import APA_LABEL, APA_TICK, APA_ANNOT

apply_defaults()

def draw_styled_box(ax, x, y, width, height, text, facecolor='#f0f0f0', edgecolor='#333333'):
    """Draws a professional rounded box with centered text."""
    box = patches.FancyBboxPatch((x, y), width, height,
                                 boxstyle="round,pad=0.1",
                                 linewidth=1.5,
                                 edgecolor=edgecolor,
                                 facecolor=facecolor)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=APA_TICK, fontweight='bold', color='black')

def draw_rdm_matrix(ax, x, y, size, title, cmap='viridis'):
    """Draws a visual representation of an RDM."""
    rect = patches.Rectangle((x, y), size, size, linewidth=1, edgecolor='black', facecolor='none')
    ax.add_patch(rect)
    np.random.seed(42 if cmap == 'viridis' else 101)
    raw  = np.random.rand(10, 10)
    data = (raw + raw.T) / 2
    np.fill_diagonal(data, 0)
    ax.imshow(data, extent=(x, x+size, y, y+size), cmap=cmap, origin='upper', vmin=0, vmax=1, interpolation='nearest')
    ax.text(x + size/2, y - 0.3, title, ha='center', va='top', fontsize=APA_TICK)

def draw_triangle_icon(ax, x, y):
    """Draws a visual indicating upper-triangle extraction"""
    # Triangle polygon
    poly = patches.Polygon([[x, y], [x+0.8, y+0.8], [x, y+0.8]], closed=True, 
                           facecolor='#dddddd', edgecolor='black', linewidth=1)
    ax.add_patch(poly)
    # Dots to represent vector
    for i in range(3):
        ax.plot([x+1.0 + (i*0.2)], [y+0.4], 'o', color='black', markersize=3)

# --- Main Plot Setup ---
fig, ax = plt.subplots(figsize=(13, 6))
ax.set_xlim(0, 15)
ax.set_ylim(0, 7)
ax.axis('off')

# ==========================================
# 1. INPUTS
# ==========================================
draw_styled_box(ax, x=0.5, y=4.5, width=2.8, height=1.0, text="fMRI ROI Patterns")
draw_styled_box(ax, x=0.5, y=1.2, width=2.8, height=1.0, text="Model Predictors")

# ==========================================
# 2. RDMs
# ==========================================
# Arrow Brain -> RDM
ax.annotate("", xy=(4.5, 5.0), xytext=(3.3, 5.0), arrowprops=dict(arrowstyle="->", lw=2, color='#444444'))
ax.text(3.9, 5.2, "1 - Pearson r", ha='center', fontsize=APA_ANNOT, style='italic', color='#555555')
draw_rdm_matrix(ax, x=4.5, y=4.2, size=1.6, title="Neural RDM", cmap='magma')

# Arrow Model -> RDM
ax.annotate("", xy=(4.5, 1.7), xytext=(3.3, 1.7), arrowprops=dict(arrowstyle="->", lw=2, color='#444444'))
draw_rdm_matrix(ax, x=4.5, y=0.9, size=1.6, title="Model RDM", cmap='gray')

# ==========================================
# 3. VECTORIZE (The "Upper Triangle" step)
# ==========================================
# Arrows to center
ax.annotate("", xy=(7.5, 3.8), xytext=(6.2, 5.0), arrowprops=dict(arrowstyle="->", lw=2, color='#444444'))
ax.annotate("", xy=(7.5, 2.9), xytext=(6.2, 1.7), arrowprops=dict(arrowstyle="->", lw=2, color='#444444'))

# Visual: Vectorize
ax.text(7.5, 3.35, "Vectorize\nUpper Triangle", ha='center', fontsize=APA_ANNOT, fontweight='bold')
# Draw abstract vectors (dots)
ax.text(7.5, 3.9, "[ . . . . . ]", ha='center', fontsize=12, fontweight='bold', color='purple') # Neural
ax.text(7.5, 2.7, "[ . . . . . ]", ha='center', fontsize=12, fontweight='bold', color='gray')   # Model

# ==========================================
# 4. CORRELATION
# ==========================================
ax.annotate("", xy=(9.2, 3.35), xytext=(8.2, 3.35), arrowprops=dict(arrowstyle="->", lw=2, color='#444444'))
draw_styled_box(ax, x=9.2, y=2.75, width=2.2, height=1.2, text="Pearson\nCorrelation")

# ==========================================
# 5. OUTPUT with NOISE CEILING
# ==========================================
ax.annotate("", xy=(12.2, 3.35), xytext=(11.4, 3.35), arrowprops=dict(arrowstyle="->", lw=2, color='#444444'))

# Result Plot Frame
rect_res = patches.Rectangle((12.2, 2.2), 2.2, 2.0, linewidth=1.5, edgecolor='#333333', facecolor='white')
ax.add_patch(rect_res)

# NOISE CEILING (Shaded Area)
rect_ceiling = patches.Rectangle((12.2, 3.5), 2.2, 0.4, facecolor=NOISE_CEIL_COLOR, alpha=0.4)
ax.add_patch(rect_ceiling)
ax.plot([12.2, 14.4], [3.7, 3.7], color=NOISE_CEIL_COLOR, lw=1.5)
ax.text(14.5, 3.7, "Noise\nCeiling", ha='left', va='center', fontsize=APA_ANNOT, color=NOISE_CEIL_COLOR)

# Bars
bar1 = patches.Rectangle((12.4, 2.3), 0.5, 1.4, facecolor='#555555') # Bar hitting ceiling
bar2 = patches.Rectangle((13.2, 2.3), 0.5, 0.6, facecolor='#999999') # Lower bar
ax.add_patch(bar1)
ax.add_patch(bar2)

# Axis lines
ax.plot([12.2, 14.4], [2.3, 2.3], color='black', lw=1) # X axis

# Label
ax.text(13.3, 1.9, "Model Performance", ha='center', fontsize=APA_TICK, fontweight='bold')

plt.tight_layout()
out_fig_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)
plt.savefig(os.path.join(out_fig_dir, 'rsa_pipeline_visual.png'), dpi=300, bbox_inches='tight')
plt.show()