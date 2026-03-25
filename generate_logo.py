"""
generate_logo.py
----------------
Generate a professional logo for SalesSense project.
Requires: matplotlib, numpy, pillow
Run: python generate_logo.py
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.patches import FancyBboxPatch, Polygon
import os

# ── Configuration ──────────────────────────────────────────────────────────
OUTPUT_DIR = "images"
LOGO_FILENAME = "logo.png"
LOGO_SIZE = (1024, 1024)  # High resolution
DPI = 150

# ── Colors ─────────────────────────────────────────────────────────────────
PRIMARY_BLUE = "#4169E1"  # Royal Blue
ACCENT_RED = "#FF4B4B"    # Modern Red
LIGHT_BLUE = "#6495ED"
WHITE = "#FFFFFF"
DARK_GRAY = "#333333"

# ══════════════════════════════════════════════════════════════════════════════
def create_logo():
    """Create a modern, minimalist SalesSense logo."""
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=DPI)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # ── Minimalist geometric design ─────────────────────────────────────────
    # Three geometric blocks: red, blue, black (modern and clean)
    
    # Red block (left)
    red_block = patches.Rectangle((2, 4.5), 1.2, 2.5, 
                                   facecolor=ACCENT_RED, 
                                   edgecolor='none', zorder=2)
    ax.add_patch(red_block)
    
    # Blue block (middle)
    blue_block = patches.Rectangle((3.5, 3.8), 1.2, 3.2, 
                                    facecolor=PRIMARY_BLUE, 
                                    edgecolor='none', zorder=2)
    ax.add_patch(blue_block)
    
    # Black block (right)
    black_block = patches.Rectangle((5, 3), 1.2, 4, 
                                     facecolor=DARK_GRAY, 
                                     edgecolor='none', zorder=2)
    ax.add_patch(black_block)
    
    # ── Upward trending line (sleek) ───────────────────────────────────────
    line_x = [2, 3.7, 6.2]
    line_y = [4, 6, 8]
    ax.plot(line_x, line_y, color=ACCENT_RED, linewidth=4, zorder=3)
    
    # Large arrow at the end
    ax.annotate('', xy=(6.5, 8.3), xytext=(6.1, 7.8),
                arrowprops=dict(arrowstyle='->', color=ACCENT_RED, lw=3.5, mutation_scale=35),
                zorder=3)
    
    # ── Brand Text: "SalesSense" ───────────────────────────────────────────
    ax.text(5, 1.5,
            "SalesSense",
            fontsize=36,
            fontweight="bold",
            ha="center",
            va="center",
            color=DARK_GRAY,
            fontfamily="Arial",
            zorder=3
    )
    
    # ── Tagline ────────────────────────────────────────────────────────────
    ax.text(5, 0.7,
            "AI-Powered Sales Forecasting",
            fontsize=10,
            ha="center",
            va="center",
            color=PRIMARY_BLUE,
            fontfamily="Arial",
            style="italic",
            zorder=3
    )
    
    plt.tight_layout(pad=0)
    
    output_path = os.path.join(OUTPUT_DIR, LOGO_FILENAME)
    plt.savefig(output_path,
                bbox_inches='tight',
                pad_inches=0.2,
                facecolor='white',
                edgecolor='none',
                dpi=DPI,
                format='png'
    )
    print(f"✅ Logo saved → {output_path}")
    
    # ── Icon version (no text) ─────────────────────────────────────────────
    fig_icon, ax_icon = plt.subplots(1, 1, figsize=(6, 6), dpi=DPI)
    ax_icon.set_xlim(0, 10)
    ax_icon.set_ylim(0, 10)
    ax_icon.axis('off')
    
    # Red block
    red_block = patches.Rectangle((2, 4), 1.2, 2.5, 
                                   facecolor=ACCENT_RED, 
                                   edgecolor='none', zorder=2)
    ax_icon.add_patch(red_block)
    
    # Blue block
    blue_block = patches.Rectangle((3.5, 3.3), 1.2, 3.2, 
                                    facecolor=PRIMARY_BLUE, 
                                    edgecolor='none', zorder=2)
    ax_icon.add_patch(blue_block)
    
    # Black block
    black_block = patches.Rectangle((5, 2.5), 1.2, 4, 
                                     facecolor=DARK_GRAY, 
                                     edgecolor='none', zorder=2)
    ax_icon.add_patch(black_block)
    
    # Trending line
    line_x = [2, 3.7, 6.2]
    line_y = [3.5, 5.5, 7.5]
    ax_icon.plot(line_x, line_y, color=ACCENT_RED, linewidth=4, zorder=3)
    
    # Arrow
    ax_icon.annotate('', xy=(6.5, 7.8), xytext=(6.1, 7.3),
                     arrowprops=dict(arrowstyle='->', color=ACCENT_RED, lw=3.5, mutation_scale=35),
                     zorder=3)
    
    plt.tight_layout(pad=0)
    
    icon_path = os.path.join(OUTPUT_DIR, "logo_icon.png")
    plt.savefig(icon_path,
                bbox_inches='tight',
                pad_inches=0.1,
                facecolor='white',
                edgecolor='none',
                dpi=DPI,
                format='png'
    )
    print(f"✅ Icon saved → {icon_path}")
    
    plt.close('all')


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("🎨 Generating SalesSense logo...")
    create_logo()
    print("\n✨ Done! Logos created successfully.")
    print("   - logo.png (full logo with text)")
    print("   - logo_icon.png (icon only)")
