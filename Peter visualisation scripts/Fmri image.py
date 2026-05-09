import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from src.plot_config import apply_defaults
from src.figure_style import APA_LABEL, APA_TICK, APA_ANNOT

apply_defaults()

def draw_timeline(ax):
    # Setup axis
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    
    # 1. Stimulus Box (Very short)
    # Representing 50ms vs 1950ms is hard to scale perfectly visually 
    # so we exaggerate the stimulus width slightly for visibility but label correctly.
    rect_stim = patches.Rectangle((0.5, 1.5), 1.5, 2, linewidth=2, edgecolor='#333333', facecolor='#D3D3D3')
    ax.add_patch(rect_stim)
    ax.text(1.25, 2.5, 'Image', ha='center', fontweight='bold', fontsize=APA_TICK)
    
    # 2. ISI Box (Long)
    rect_isi = patches.Rectangle((2.2, 1.5), 6, 2, linewidth=2, edgecolor='#333333', facecolor='#F5F5F5')
    ax.add_patch(rect_isi)
    ax.text(5.2, 2.5, '+', ha='center', fontsize=30, fontweight='bold')
    
    # Labels
    ax.text(1.25, 1.1, '50 ms', ha='center', fontsize=APA_LABEL)
    ax.text(5.2, 1.1, 'ISI (Fixation)\n1950 ms', ha='center', fontsize=APA_LABEL)
    
    # TR Bracket
    ax.annotate('', xy=(0.5, 0.5), xytext=(8.2, 0.5), 
                arrowprops=dict(arrowstyle='|-|', linewidth=1.5))
    ax.text(4.35, 0.2, 'One Trial (TR = 2000 ms)', ha='center', fontweight='bold', fontsize=APA_LABEL)
    
    ax.set_title("A. Trial Timeline", loc='left', fontsize=APA_LABEL, fontweight='bold')

def draw_hands(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')
    ax.set_title("B. Response Mapping (Example)", loc='left', fontsize=APA_LABEL, fontweight='bold')
    
    # Simple polygons for hands (schematic)
    # Left Hand
    left_hand_x = [1, 1.2, 1.5, 1.8, 2.1, 2.5, 2.5, 1]
    left_hand_y = [1, 2.5, 2.8, 2.8, 2.5, 2.0, 0.5, 0.5] 
    # (This is a simplified shape, usually best to use an icon or just text boxes)
    
    # Instead of drawing complex hands in code, we will use Text Boxes representing fingers
    # Left Hand Fingers
    fingers_L = ['Little (L)', 'Ring (L)', 'Middle (L)', 'Index (L)']
    fingers_R = ['Index (R)', 'Middle (R)', 'Ring (R)', 'Little (R)']
    
    # Draw Left Hand Mappings
    for i, finger in enumerate(fingers_L):
        x_pos = 1 + i*1
        rect = patches.Rectangle((x_pos, 2), 0.8, 0.8, facecolor='#e0e0e0', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x_pos+0.4, 2.4, finger, ha='center', fontsize=APA_TICK, rotation=45)
        # Arrow down to category
        ax.annotate('', xy=(x_pos+0.4, 1.2), xytext=(x_pos+0.4, 2), arrowprops=dict(arrowstyle='->'))
        ax.text(x_pos+0.4, 0.8, f"Cat {i+1}", ha='center', fontsize=APA_TICK, fontweight='bold')

    # Draw Right Hand Mappings
    for i, finger in enumerate(fingers_R):
        x_pos = 5.5 + i*1
        rect = patches.Rectangle((x_pos, 2), 0.8, 0.8, facecolor='#e0e0e0', edgecolor='black')
        ax.add_patch(rect)
        ax.text(x_pos+0.4, 2.4, finger, ha='center', fontsize=APA_TICK, rotation=45)
        # Arrow down to category
        ax.annotate('', xy=(x_pos+0.4, 1.2), xytext=(x_pos+0.4, 2), arrowprops=dict(arrowstyle='->'))
        ax.text(x_pos+0.4, 0.8, f"Cat {i+5}", ha='center', fontsize=APA_TICK, fontweight='bold')

    # Annotation about randomization
    ax.text(5, 3.5, "Categories assigned to specific fingers\n(Randomized per participant)",
            ha='center', fontsize=APA_LABEL, style='italic', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

# --- Main Plot ---
fig = plt.figure(figsize=(12, 5))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

ax1 = fig.add_subplot(gs[0])
draw_timeline(ax1)

ax2 = fig.add_subplot(gs[1])
draw_hands(ax2)

plt.tight_layout()
out_fig_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'figures')
os.makedirs(out_fig_dir, exist_ok=True)
plt.savefig(os.path.join(out_fig_dir, 'fmri_trial_design.png'), dpi=300, bbox_inches='tight')
plt.show()