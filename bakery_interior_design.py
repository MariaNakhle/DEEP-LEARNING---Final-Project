"""
Artisan Bakery Interior Design
================================
Design specifications:
- Space: 11 meters long x 5 meters wide
- North-facing glass storefront
- Style: Rustic modern bakery with Middle Eastern heritage
- Features: Open stone oven, bread display, central walking path, cashier counter, preparation area
- Aesthetic: Warm lighting, natural wood, stone, brick, cozy, authentic, non-corporate

This script generates a floor plan and design visualization for the artisan bakery.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle, Polygon
import numpy as np
import os

# Create output directory for design visualizations
OUTPUT_DIR = os.path.join("images", "bakery_design")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Design constants (in meters)
BAKERY_LENGTH = 11  # meters (North-South dimension)
BAKERY_WIDTH = 5    # meters (East-West dimension)

# Color palette - Rustic Modern with Middle Eastern Heritage
COLORS = {
    'wall': '#D4C4A8',           # Warm beige/stone
    'floor': '#E8DCC4',          # Light wood
    'wood': '#8B6F47',           # Natural wood
    'stone_oven': '#7A6F5D',     # Stone gray
    'brick': '#A67C52',          # Terracotta brick
    'glass': '#B8D8E8',          # Glass storefront
    'counter': '#9B8763',        # Counter surface
    'display': '#C19A6B',        # Display shelving
    'text': '#2C1810',           # Dark text
    'path': '#F5EBD7',           # Walking path
    'prep_area': '#D4B896',      # Preparation zone
    'accent': '#C87533',         # Middle Eastern copper accent
    'fire': '#FF6B35'            # Oven fire/flames
}

def create_floor_plan():
    """
    Creates the main floor plan visualization for the artisan bakery.
    Layout from North (top) to South (bottom):
    - North: Glass storefront (entrance)
    - Left: Bread display area
    - Center: Walking path
    - Right: Open stone oven
    - Middle: Cashier counter (after display)
    - South: Preparation area at back
    """
    fig, ax = plt.subplots(figsize=(12, 16))
    ax.set_xlim(0, BAKERY_WIDTH + 1)
    ax.set_ylim(0, BAKERY_LENGTH + 1)
    ax.set_aspect('equal')
    
    # Set up the plot
    ax.set_title('Artisan Bakery Interior Design\nRustic Modern with Middle Eastern Heritage\n11m x 5m', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xlabel('Width (meters)', fontsize=12)
    ax.set_ylabel('Length (meters)', fontsize=12)
    
    # Add grid for reference
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_xticks(np.arange(0, BAKERY_WIDTH + 1, 0.5))
    ax.set_yticks(np.arange(0, BAKERY_LENGTH + 1, 0.5))
    
    # Draw floor
    floor = Rectangle((0, 0), BAKERY_WIDTH, BAKERY_LENGTH, 
                      facecolor=COLORS['floor'], edgecolor='none', alpha=0.3)
    ax.add_patch(floor)
    
    # === NORTH SIDE: Glass Storefront (entrance) ===
    # Position at y = 10-11 (top of bakery)
    glass_storefront = Rectangle((0.2, 10.2), BAKERY_WIDTH - 0.4, 0.6,
                                 facecolor=COLORS['glass'], edgecolor=COLORS['text'],
                                 linewidth=3, alpha=0.6, label='Glass Storefront')
    ax.add_patch(glass_storefront)
    
    # Add entrance door
    door_width = 1.2
    door_x = (BAKERY_WIDTH - door_width) / 2
    door = Rectangle((door_x, 10.2), door_width, 0.6,
                     facecolor='white', edgecolor=COLORS['text'],
                     linewidth=2, alpha=0.8)
    ax.add_patch(door)
    ax.text(BAKERY_WIDTH / 2, 10.5, 'ENTRANCE', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['text'])
    
    # North label
    ax.text(BAKERY_WIDTH / 2, 11.5, '▲ NORTH', ha='center', va='center',
            fontsize=14, fontweight='bold', color='darkblue')
    
    # === LEFT SIDE: Bread Display Area ===
    # Position: x = 0.3-1.8, y = 6.5-9.5 (left side)
    display_depth = 1.5
    display_length = 3.0
    display_y_start = 6.5
    
    # Display shelving unit
    bread_display = FancyBboxPatch((0.3, display_y_start), display_depth, display_length,
                                   boxstyle="round,pad=0.05", 
                                   facecolor=COLORS['display'], 
                                   edgecolor=COLORS['wood'],
                                   linewidth=3, alpha=0.8)
    ax.add_patch(bread_display)
    
    # Add shelves detail
    for i in range(4):
        shelf_y = display_y_start + 0.3 + i * 0.7
        ax.plot([0.4, 0.4 + display_depth - 0.2], [shelf_y, shelf_y],
                color=COLORS['wood'], linewidth=2, alpha=0.8)
    
    ax.text(0.3 + display_depth/2, display_y_start + display_length/2, 
            'BREAD\nDISPLAY', ha='center', va='center',
            fontsize=11, fontweight='bold', color=COLORS['text'])
    
    # === RIGHT SIDE: Open Stone Oven ===
    # Position: x = 3.2-4.8, y = 7.0-9.5 (right side, visible to customers)
    oven_width = 1.6
    oven_depth = 2.5
    oven_x = 3.2
    oven_y = 7.0
    
    # Oven body
    stone_oven = FancyBboxPatch((oven_x, oven_y), oven_width, oven_depth,
                                boxstyle="round,pad=0.1",
                                facecolor=COLORS['stone_oven'],
                                edgecolor=COLORS['brick'],
                                linewidth=4, alpha=0.9)
    ax.add_patch(stone_oven)
    
    # Oven opening (facing the walking path)
    opening_width = 0.8
    opening_height = 0.6
    opening_x = oven_x + (oven_width - opening_width) / 2
    opening_y = oven_y + 0.3
    oven_opening = Rectangle((opening_x, opening_y), opening_width, opening_height,
                             facecolor=COLORS['fire'], edgecolor=COLORS['text'],
                             linewidth=2, alpha=0.7)
    ax.add_patch(oven_opening)
    
    # Add brick texture pattern
    for i in range(5):
        for j in range(3):
            brick_x = oven_x + 0.2 + j * 0.4
            brick_y = oven_y + 0.2 + i * 0.4
            if brick_x < oven_x + oven_width - 0.2 and brick_y < oven_y + oven_depth - 0.2:
                ax.plot([brick_x, brick_x + 0.3], [brick_y, brick_y],
                       color=COLORS['brick'], linewidth=1, alpha=0.5)
    
    ax.text(oven_x + oven_width/2, oven_y + oven_depth - 0.5,
            'STONE\nOVEN', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    
    # === CENTER: Clear Walking Path ===
    # Position: x = 1.9-3.1, y = 1.0-10.0 (central corridor)
    path_width = 1.2
    path_x = 1.9
    
    walking_path = Rectangle((path_x, 1.0), path_width, 9.0,
                             facecolor=COLORS['path'], edgecolor=COLORS['wood'],
                             linewidth=2, linestyle='--', alpha=0.5)
    ax.add_patch(walking_path)
    
    # Add directional arrow
    ax.annotate('', xy=(path_x + path_width/2, 9.5), 
                xytext=(path_x + path_width/2, 2.0),
                arrowprops=dict(arrowstyle='<->', color=COLORS['text'], 
                              lw=2, alpha=0.5))
    ax.text(path_x + path_width/2, 5.5, 'WALKING\nPATH', 
            ha='center', va='center', fontsize=10, 
            fontweight='bold', color=COLORS['text'], alpha=0.7,
            rotation=90)
    
    # === MIDDLE: Cashier Counter ===
    # Position: x = 0.3-1.8, y = 5.0-6.0 (after the display, left side)
    counter_width = 1.5
    counter_depth = 1.0
    counter_x = 0.3
    counter_y = 5.0
    
    cashier_counter = FancyBboxPatch((counter_x, counter_y), counter_width, counter_depth,
                                     boxstyle="round,pad=0.05",
                                     facecolor=COLORS['counter'],
                                     edgecolor=COLORS['accent'],
                                     linewidth=3, alpha=0.9)
    ax.add_patch(cashier_counter)
    
    # Add register/POS indicator
    register = Circle((counter_x + counter_width - 0.3, counter_y + counter_depth/2), 
                     0.15, facecolor=COLORS['accent'], 
                     edgecolor=COLORS['text'], linewidth=2)
    ax.add_patch(register)
    
    ax.text(counter_x + counter_width/2, counter_y + counter_depth/2,
            'CASHIER', ha='center', va='center',
            fontsize=10, fontweight='bold', color='white')
    
    # === BACK: Preparation Area ===
    # Position: y = 0.5-3.5 (southern end, back of bakery)
    prep_height = 3.0
    prep_y = 0.5
    
    preparation_area = Rectangle((0.3, prep_y), BAKERY_WIDTH - 0.6, prep_height,
                                 facecolor=COLORS['prep_area'],
                                 edgecolor=COLORS['text'],
                                 linewidth=3, alpha=0.7, linestyle='--')
    ax.add_patch(preparation_area)
    
    # Add prep station details
    # Work counter on left
    work_counter = Rectangle((0.5, prep_y + 0.3), 1.5, 0.8,
                            facecolor=COLORS['counter'],
                            edgecolor=COLORS['wood'], linewidth=2)
    ax.add_patch(work_counter)
    
    # Mixing station on right
    mixing_station = Rectangle((3.0, prep_y + 0.3), 1.5, 0.8,
                               facecolor=COLORS['counter'],
                               edgecolor=COLORS['wood'], linewidth=2)
    ax.add_patch(mixing_station)
    
    # Storage area at back wall
    storage = Rectangle((1.5, prep_y + 0.2), 1.5, 0.4,
                       facecolor=COLORS['wood'],
                       edgecolor=COLORS['text'], linewidth=1.5)
    ax.add_patch(storage)
    
    ax.text(BAKERY_WIDTH / 2, prep_y + prep_height/2,
            'PREPARATION AREA\n(Staff Only)', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['text'])
    
    # === Additional Design Elements ===
    
    # Add seating/waiting area (small bench near entrance)
    bench_width = 1.2
    bench_x = BAKERY_WIDTH - 1.5
    bench_y = 8.5
    bench = Rectangle((bench_x, bench_y), bench_width, 0.4,
                     facecolor=COLORS['wood'], edgecolor=COLORS['text'],
                     linewidth=2, alpha=0.8)
    ax.add_patch(bench)
    ax.text(bench_x + bench_width/2, bench_y + 0.2, 'Seating',
            ha='center', va='center', fontsize=8, color='white')
    
    # Add decorative elements indicators
    # Hanging lights (warm lighting)
    light_positions = [(1.5, 8.0), (3.5, 8.0), (2.5, 5.0), (2.5, 2.0)]
    for lx, ly in light_positions:
        light = Circle((lx, ly), 0.1, facecolor='#FFD700',
                      edgecolor=COLORS['accent'], linewidth=1, alpha=0.8)
        ax.add_patch(light)
        # Light rays
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            rad = np.radians(angle)
            dx = 0.15 * np.cos(rad)
            dy = 0.15 * np.sin(rad)
            ax.plot([lx, lx + dx], [ly, ly + dy], color='#FFD700', 
                   linewidth=1, alpha=0.5)
    
    # Add dimension annotations
    # Length dimension
    ax.annotate('', xy=(BAKERY_WIDTH + 0.5, 0), xytext=(BAKERY_WIDTH + 0.5, BAKERY_LENGTH),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(BAKERY_WIDTH + 0.7, BAKERY_LENGTH/2, f'{BAKERY_LENGTH}m',
            rotation=90, va='center', fontsize=11, fontweight='bold')
    
    # Width dimension
    ax.annotate('', xy=(0, -0.5), xytext=(BAKERY_WIDTH, -0.5),
                arrowprops=dict(arrowstyle='<->', color='black', lw=2))
    ax.text(BAKERY_WIDTH/2, -0.7, f'{BAKERY_WIDTH}m',
            ha='center', fontsize=11, fontweight='bold')
    
    # Add legend
    legend_elements = [
        patches.Patch(facecolor=COLORS['glass'], edgecolor=COLORS['text'], 
                     label='Glass Storefront (North)'),
        patches.Patch(facecolor=COLORS['display'], edgecolor=COLORS['wood'], 
                     label='Bread Display (Left)'),
        patches.Patch(facecolor=COLORS['stone_oven'], edgecolor=COLORS['brick'], 
                     label='Open Stone Oven'),
        patches.Patch(facecolor=COLORS['counter'], edgecolor=COLORS['accent'], 
                     label='Cashier Counter'),
        patches.Patch(facecolor=COLORS['prep_area'], edgecolor=COLORS['text'], 
                     label='Preparation Area (Back)'),
        patches.Patch(facecolor=COLORS['path'], edgecolor=COLORS['wood'], 
                     label='Central Walking Path'),
        patches.Patch(facecolor=COLORS['wood'], edgecolor=COLORS['text'], 
                     label='Natural Wood Elements'),
    ]
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, -0.05),
             ncol=2, fontsize=9, framealpha=0.95)
    
    plt.tight_layout()
    
    # Save the floor plan
    output_path = os.path.join(OUTPUT_DIR, 'bakery_floor_plan.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Floor plan saved to: {output_path}")
    
    return fig, ax


def create_design_specifications():
    """
    Creates a detailed design specifications document with text and visual elements.
    """
    fig = plt.figure(figsize=(14, 18))
    
    # Title
    fig.suptitle('Artisan Bakery Interior Design\nDetailed Specifications', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Create text layout
    spec_text = f"""
SPACE DIMENSIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Total Area: {BAKERY_LENGTH}m (length) × {BAKERY_WIDTH}m (width) = {BAKERY_LENGTH * BAKERY_WIDTH}m²
• Ceiling Height: 3.5m (recommended for spacious feel)
• Orientation: North-facing entrance with glass storefront


DESIGN STYLE & ATMOSPHERE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Primary Style: Rustic Modern
• Cultural Influence: Middle Eastern Heritage
• Atmosphere: Cozy, authentic, warm, non-corporate
• Target: Artisan quality, handcrafted authenticity


SPATIAL LAYOUT (North to South)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NORTH SECTION (Entrance Zone)
   ─────────────────────────────────────
   • Glass Storefront: Full-width glazing for natural light
   • Entrance Door: Central position, 1.2m wide
   • Visibility: Clear view of interior and oven from outside
   • Materials: Aluminum-framed glass, bronze/copper handles

2. LEFT SIDE (Display Area)
   ─────────────────────────────────────
   • Bread Display: 1.5m deep × 3.0m long shelving unit
   • Position: Immediate left upon entry (y: 6.5-9.5m)
   • Height: 1.8m tall, 4 shelves
   • Material: Reclaimed wood with natural finish
   • Features: Open shelving, woven baskets, tilted display

3. RIGHT SIDE (Stone Oven)
   ─────────────────────────────────────
   • Open Stone Oven: 1.6m × 2.5m visible cooking area
   • Position: Right side (y: 7.0-9.5m), facing central path
   • Material: Natural stone, exposed brick accents
   • Features: Arched opening, visible flames, terracotta tiles
   • Visibility: Fully visible to customers in walking path

4. CENTER (Walking Path)
   ─────────────────────────────────────
   • Width: 1.2m clear corridor
   • Length: 9.0m (entrance to prep area boundary)
   • Flooring: Wide-plank natural oak, matte finish
   • Purpose: Customer flow, oven viewing, counter access

5. CASHIER COUNTER
   ─────────────────────────────────────
   • Position: Left side after display (y: 5.0-6.0m)
   • Dimensions: 1.5m wide × 1.0m deep
   • Height: 1.1m (standing counter)
   • Material: Butcher block wood top, copper accents
   • Features: POS system, cash drawer, small display

6. BACK SECTION (Preparation Area)
   ─────────────────────────────────────
   • Position: Southern end (y: 0.5-3.5m)
   • Access: Staff only, partial wall/screen separation
   • Work Surfaces: Stainless steel with wood trim
   • Storage: Built-in shelving, ingredient storage
   • Equipment: Mixing stations, proofing areas


MATERIALS & FINISHES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Flooring:
• Main Area: Wide-plank natural oak (matte finish)
• Prep Area: Sealed concrete with wood accents

Walls:
• Primary: Exposed brick in warm earth tones
• Accent: White lime wash plaster
• Feature: Reclaimed wood paneling (selected areas)

Oven Area:
• Stone: Natural fieldstone or limestone
• Brick: Terracotta red, visible mortar
• Hearth: Clay tiles, heat-resistant

Counters & Display:
• Display Shelves: Reclaimed wood, natural grain
• Cashier Counter: Butcher block maple or oak
• Prep Surfaces: Stainless steel with wood edges

Accents:
• Metal: Brushed copper and bronze fixtures
• Textiles: Woven baskets, linen bread bags
• Details: Hand-forged iron brackets, ceramic tiles


LIGHTING DESIGN
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Ambient Lighting:
• Pendant Lights: 4 copper/brass pendants (see floor plan)
• Temperature: Warm white (2700-3000K)
• Style: Industrial-artisan with Edison bulbs

Task Lighting:
• Display Area: LED strips under shelves (warm white)
• Oven Area: Firelight provides natural warm glow
• Counter: Focused pendants over cashier station

Natural Light:
• North Storefront: Maximum daylight entry
• Light Quality: Soft, even, non-harsh northern exposure
• Control: Linen curtains/shades for evening ambiance


COLOR PALETTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Primary: Warm beiges, cream, natural wood tones
• Secondary: Terracotta, stone gray, soft white
• Accents: Copper, bronze, burnt orange
• Atmosphere: Warm, inviting, earth-connected


CULTURAL & DESIGN ELEMENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Middle Eastern Heritage Integration:
• Geometric Patterns: Subtle tile work near oven
• Arched Details: Oven opening, mirror frames
• Textiles: Woven baskets, hand-crafted displays
• Materials: Clay, copper, natural fibers
• Craftsmanship: Visible hand-made quality

Modern Rustic Balance:
• Clean Lines: Simplified forms, uncluttered
• Natural Materials: Authentic, aged finishes
• Functional Beauty: Everything serves purpose
• Honest Construction: Visible joinery, exposed materials


CUSTOMER EXPERIENCE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Entry Experience:
1. Glass storefront creates anticipation
2. View of stone oven draws customers in
3. Aroma of fresh bread greets at door
4. Clear sight lines throughout space

Journey Flow:
1. Enter → immediately see display and oven
2. Browse bread display on left
3. Watch baking process (oven on right)
4. Proceed to cashier counter
5. Exit with clear path

Sensory Design:
• Visual: Warm colors, natural materials, active oven
• Olfactory: Fresh bread, wood fire, natural scents
• Auditory: Soft background, oven crackling
• Tactile: Natural wood, woven baskets, authentic materials


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design Philosophy: Create an authentic artisan bakery
that celebrates traditional craft while providing modern
comfort and functionality. Every element tells a story
of quality, heritage, and genuine craftsmanship.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    # Add text to figure
    ax = fig.add_subplot(111)
    ax.axis('off')
    ax.text(0.05, 0.95, spec_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # Save specifications
    output_path = os.path.join(OUTPUT_DIR, 'design_specifications.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Design specifications saved to: {output_path}")
    
    return fig


def create_3d_perspective_sketch():
    """
    Creates a simple 3D perspective view sketch of the bakery interior.
    """
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Title
    ax.text(7, 9.5, 'Interior Perspective View - Looking from Entrance',
            ha='center', fontsize=16, fontweight='bold')
    
    # Background - back wall
    back_wall = Polygon([(2, 2), (12, 2), (12, 8), (2, 8)],
                       facecolor=COLORS['wall'], edgecolor=COLORS['text'],
                       linewidth=2, alpha=0.8)
    ax.add_patch(back_wall)
    
    # Floor (perspective)
    floor = Polygon([(1, 1), (13, 1), (12, 2), (2, 2)],
                   facecolor=COLORS['floor'], edgecolor=COLORS['wood'],
                   linewidth=2, alpha=0.7)
    ax.add_patch(floor)
    
    # Left wall with display
    left_wall = Polygon([(2, 2), (2, 8), (1, 9), (1, 1)],
                       facecolor=COLORS['wall'], edgecolor=COLORS['text'],
                       linewidth=2, alpha=0.6)
    ax.add_patch(left_wall)
    
    # Right wall with oven
    right_wall = Polygon([(12, 2), (12, 8), (13, 9), (13, 1)],
                        facecolor=COLORS['wall'], edgecolor=COLORS['text'],
                        linewidth=2, alpha=0.6)
    ax.add_patch(right_wall)
    
    # Stone oven on right (perspective)
    oven_body = Polygon([(9.5, 3.5), (11.5, 3.5), (11.5, 6.5), (9.5, 6.5)],
                       facecolor=COLORS['stone_oven'], edgecolor=COLORS['brick'],
                       linewidth=3, alpha=0.9)
    ax.add_patch(oven_body)
    
    # Oven opening
    oven_opening = Polygon([(10, 4), (11, 4), (11, 5.5), (10, 5.5)],
                          facecolor=COLORS['fire'], edgecolor=COLORS['text'],
                          linewidth=2, alpha=0.8)
    ax.add_patch(oven_opening)
    
    # Fire glow
    fire = Circle((10.5, 4.75), 0.3, facecolor='#FFD700', alpha=0.7)
    ax.add_patch(fire)
    
    ax.text(10.5, 7, 'STONE OVEN', ha='center', fontsize=10, fontweight='bold')
    
    # Bread display on left (perspective)
    display_shelves = [
        Polygon([(2.5, 4), (4.5, 4), (4.5, 4.2), (2.5, 4.2)], 
                facecolor=COLORS['display'], edgecolor=COLORS['wood'], linewidth=2),
        Polygon([(2.5, 5), (4.5, 5), (4.5, 5.2), (2.5, 5.2)], 
                facecolor=COLORS['display'], edgecolor=COLORS['wood'], linewidth=2),
        Polygon([(2.5, 6), (4.5, 6), (4.5, 6.2), (2.5, 6.2)], 
                facecolor=COLORS['display'], edgecolor=COLORS['wood'], linewidth=2),
    ]
    for shelf in display_shelves:
        ax.add_patch(shelf)
    
    # Support posts
    ax.plot([2.5, 2.5], [3.5, 6.5], color=COLORS['wood'], linewidth=4)
    ax.plot([4.5, 4.5], [3.5, 6.5], color=COLORS['wood'], linewidth=4)
    
    ax.text(3.5, 7, 'BREAD DISPLAY', ha='center', fontsize=10, fontweight='bold')
    
    # Add bread illustrations on shelves
    bread_positions = [(3, 4.1), (3.8, 4.1), (2.8, 5.1), (3.5, 5.1), (4.2, 5.1),
                      (3, 6.1), (3.8, 6.1)]
    for bx, by in bread_positions:
        bread = Circle((bx, by), 0.15, facecolor='#D4A574', 
                      edgecolor='#8B6F47', linewidth=1)
        ax.add_patch(bread)
    
    # Cashier counter (center-left)
    counter = Rectangle((4, 2.5), 2, 0.8, facecolor=COLORS['counter'],
                       edgecolor=COLORS['accent'], linewidth=2)
    ax.add_patch(counter)
    ax.text(5, 2.9, 'CASHIER', ha='center', fontsize=9, fontweight='bold', color='white')
    
    # Hanging lights
    light_positions = [(5, 8.5), (7, 8.5), (9, 8.5)]
    for lx, ly in light_positions:
        # Cord
        ax.plot([lx, lx], [ly, ly + 0.5], color=COLORS['text'], linewidth=1)
        # Shade
        shade = Polygon([(lx-0.3, ly), (lx+0.3, ly), (lx+0.2, ly-0.4), (lx-0.2, ly-0.4)],
                       facecolor=COLORS['accent'], edgecolor=COLORS['text'], 
                       linewidth=1, alpha=0.8)
        ax.add_patch(shade)
        # Glow
        glow = Circle((lx, ly-0.5), 0.8, facecolor='#FFD700', alpha=0.2)
        ax.add_patch(glow)
    
    # Walking path indication
    ax.annotate('', xy=(7, 1.5), xytext=(7, 0.5),
                arrowprops=dict(arrowstyle='->', color=COLORS['text'], lw=3))
    ax.text(7, 0.3, 'WALKING PATH', ha='center', fontsize=10, fontweight='bold')
    
    # Add atmosphere notes
    atmosphere_text = """
    ATMOSPHERE ELEMENTS:
    • Warm pendant lighting (copper/brass)
    • Natural wood display shelving
    • Active stone oven with visible fire
    • Earth-toned color palette
    • Handcrafted, artisan quality
    • Middle Eastern architectural details
    """
    
    ax.text(0.5, 5, atmosphere_text, fontsize=8, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor=COLORS['floor'], alpha=0.7))
    
    plt.tight_layout()
    
    # Save perspective view
    output_path = os.path.join(OUTPUT_DIR, 'interior_perspective.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Interior perspective saved to: {output_path}")
    
    return fig


def generate_design_report():
    """
    Generates a summary report of the bakery design.
    """
    report = f"""
╔══════════════════════════════════════════════════════════════════════╗
║                  ARTISAN BAKERY INTERIOR DESIGN                      ║
║                         Design Summary Report                         ║
╚══════════════════════════════════════════════════════════════════════╝

PROJECT SPECIFICATIONS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Space Dimensions: {BAKERY_LENGTH}m (length) × {BAKERY_WIDTH}m (width)
Total Area: {BAKERY_LENGTH * BAKERY_WIDTH}m²
Orientation: North-facing glass storefront
Style: Rustic modern with Middle Eastern heritage
Atmosphere: Cozy, authentic, artisan, non-corporate

LAYOUT SUMMARY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ North-facing glass storefront with central entrance door
✓ Open stone oven positioned on right side, visible to customers
✓ Bread display area on left side (1.5m × 3.0m)
✓ Clear central walking path (1.2m wide × 9.0m long)
✓ Cashier counter positioned after display area
✓ Preparation area at the back (staff only)

KEY FEATURES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Glass Storefront (North)
   - Full-width glazing for natural light
   - Clear visibility into bakery interior
   - Inviting entrance with view of oven

2. Stone Oven (Right Side)
   - Open design, visible to customers
   - Natural stone and brick construction
   - Active fire creates warm ambiance
   - Central focal point of design

3. Bread Display (Left Side)
   - 4-tier shelving unit
   - Natural reclaimed wood
   - Open design with woven baskets
   - Immediate visibility upon entry

4. Walking Path (Center)
   - 1.2m wide corridor
   - Clear customer flow
   - Viewing access to oven
   - Wide-plank oak flooring

5. Cashier Counter
   - After display area
   - Butcher block wood with copper accents
   - Efficient transaction point
   - Integrated POS system

6. Preparation Area (Back)
   - Staff-only workspace
   - Stainless steel work surfaces
   - Storage and equipment
   - Separated from customer area

MATERIALS & FINISHES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Flooring: Wide-plank natural oak (matte finish)
Walls: Exposed brick, lime wash plaster, reclaimed wood paneling
Oven: Natural stone, terracotta brick, clay tiles
Counters: Butcher block wood, stainless steel
Display: Reclaimed wood shelving, woven baskets
Accents: Copper/bronze fixtures, hand-forged iron
Lighting: Warm pendant lights (2700-3000K)

COLOR PALETTE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Primary: Warm beige (#D4C4A8), Natural wood (#8B6F47), Cream (#E8DCC4)
Secondary: Stone gray (#7A6F5D), Terracotta (#A67C52)
Accents: Copper (#C87533), Bronze, Burnt orange
Atmosphere: Warm, earthy, inviting, authentic

DESIGN PHILOSOPHY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
This artisan bakery design celebrates traditional baking heritage while
providing modern comfort and functionality. The space creates an authentic
experience through:

• Visible craftsmanship (open oven, handmade displays)
• Natural, honest materials (wood, stone, brick)
• Warm, inviting atmosphere (lighting, colors, textures)
• Cultural authenticity (Middle Eastern heritage elements)
• Customer engagement (visible baking process)
• Clear, intuitive flow (entrance to exit journey)

The design balances rustic charm with modern efficiency, creating a
non-corporate, community-focused bakery that emphasizes quality,
tradition, and genuine artisan craft.

DELIVERABLES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✓ Detailed floor plan with dimensions
✓ Design specifications document
✓ Interior perspective view
✓ Material and color specifications
✓ Layout and flow analysis

All design documents saved to: {OUTPUT_DIR}/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design completed with attention to authentic artisan quality,
Middle Eastern heritage, and warm, cozy atmosphere.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
    
    # Save report to text file
    report_path = os.path.join(OUTPUT_DIR, 'design_report.txt')
    try:
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
    except IOError as e:
        print(f"Warning: Could not save design report: {e}")
        return report
    
    print(report)
    print(f"\n✓ Design report saved to: {report_path}")
    
    return report


def main():
    """
    Main execution function - generates all design deliverables.
    """
    print("=" * 80)
    print("ARTISAN BAKERY INTERIOR DESIGN GENERATOR")
    print("=" * 80)
    print(f"\nGenerating design for {BAKERY_LENGTH}m × {BAKERY_WIDTH}m bakery space...")
    print(f"Style: Rustic modern with Middle Eastern heritage")
    print(f"Output directory: {OUTPUT_DIR}\n")
    
    # Generate all design components
    print("\n[1/4] Creating floor plan...")
    create_floor_plan()
    plt.close()
    
    print("\n[2/4] Creating design specifications...")
    create_design_specifications()
    plt.close()
    
    print("\n[3/4] Creating interior perspective view...")
    create_3d_perspective_sketch()
    plt.close()
    
    print("\n[4/4] Generating design report...")
    generate_design_report()
    
    print("\n" + "=" * 80)
    print("✓ DESIGN GENERATION COMPLETE")
    print("=" * 80)
    print(f"\nAll design files have been saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("\nGenerated files:")
    print("  • bakery_floor_plan.png - Detailed floor plan with dimensions")
    print("  • design_specifications.png - Complete design specifications")
    print("  • interior_perspective.png - Perspective view of interior")
    print("  • design_report.txt - Summary report")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
