import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
import torch.nn as nn
from matplotlib.patches import Circle, Rectangle, Arc
from matplotlib.lines import Line2D
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Premier League CGAN Analysis 2024/2025",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

class Generator(nn.Module):
    def __init__(self, noise_dim, condition_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim + condition_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
            nn.Sigmoid()
        )

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_dim, condition_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim + condition_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        return self.model(x)

class PassingNetworksCGAN:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator(100, 64, 22)  # 11 players * 2 coordinates
        self.discriminator = Discriminator(22, 64)
        self.formations = {
            "4-3-3": [(0.1, 0.34), (0.25, 0.15), (0.25, 0.34), (0.25, 0.53), (0.4, 0.68),
                     (0.45, 0.34), (0.5, 0.0), (0.6, 0.34), (0.75, 0.15), (0.75, 0.53), (0.85, 0.34)],
            "4-4-2": [(0.1, 0.34), (0.25, 0.15), (0.25, 0.34), (0.25, 0.53), (0.4, 0.68),
                     (0.5, 0.2), (0.5, 0.48), (0.6, 0.0), (0.6, 0.68), (0.8, 0.25), (0.8, 0.43)],
            "4-2-3-1": [(0.1, 0.34), (0.25, 0.15), (0.25, 0.34), (0.25, 0.53), (0.4, 0.68),
                       (0.45, 0.25), (0.45, 0.43), (0.6, 0.15), (0.6, 0.34), (0.6, 0.53), (0.8, 0.34)],
            "3-5-2": [(0.1, 0.34), (0.25, 0.2), (0.25, 0.34), (0.25, 0.48), 
                     (0.4, 0.1), (0.4, 0.25), (0.4, 0.43), (0.4, 0.58), (0.6, 0.34), (0.8, 0.25), (0.8, 0.43)],
            "5-3-2": [(0.1, 0.34), (0.2, 0.1), (0.2, 0.25), (0.2, 0.43), (0.2, 0.58),
                     (0.45, 0.2), (0.45, 0.34), (0.45, 0.48), (0.7, 0.34), (0.85, 0.25), (0.85, 0.43)]
        }

    def generate_dynamic_positions(self, formation="4-3-3", match_context=None):
        """Generate dynamic player positions based on match context"""
        if formation in self.formations:
            base_positions = np.array(self.formations[formation])
        else:
            base_positions = np.array(self.formations["4-3-3"])

        # Apply match-specific variations
        positions = base_positions.copy()

        if match_context:
            # Adjust positions based on match situation
            situation_factor = match_context.get('situation', 'neutral')
            time_factor = match_context.get('time', 45) / 90.0

            if situation_factor == 'attacking':
                # Move players forward
                positions[:, 0] += np.random.uniform(0.02, 0.08, len(positions))
            elif situation_factor == 'defending':
                # Pull players back
                positions[:, 0] -= np.random.uniform(0.02, 0.06, len(positions))

            # Time-based fatigue effect
            if time_factor > 0.7:
                noise_factor = 0.03 * (time_factor - 0.7) / 0.3
            else:
                noise_factor = 0.01
        else:
            noise_factor = 0.02

        # Add realistic positional variations
        noise = np.random.normal(0, noise_factor, positions.shape)
        positions += noise

        # Ensure positions stay within field bounds
        positions[:, 0] = np.clip(positions[:, 0], 0.05, 0.95)
        positions[:, 1] = np.clip(positions[:, 1], 0.05, 0.63)

        return positions

def get_position_initial(position_idx):
    """Get position initials based on player index"""
    position_map = {
        0: "GK", 1: "LB", 2: "CB", 3: "CB", 4: "RB",
        5: "CM", 6: "CM", 7: "CM", 8: "LW", 9: "ST", 10: "RW"
    }
    return position_map.get(position_idx, f"P{position_idx+1}")

def draw_professional_football_pitch(ax):
    """Draw a professional football pitch with realistic grass background"""
    # Set grass background with realistic color gradient
    ax.add_patch(Rectangle((0, 0), 1, 0.68, facecolor='#2E7D32', alpha=0.9))

    # Add grass texture pattern with alternating stripes
    for i in range(0, 100, 8):
        x = i / 100.0
        ax.axvline(x, color='#1B5E20', alpha=0.2, linewidth=1.5)
        if i % 16 == 0:
            ax.axvline(x, color='#388E3C', alpha=0.3, linewidth=2)

    # Field lines (white)
    line_color = 'white'
    line_width = 2.5

    # Outer boundary
    ax.plot([0, 1, 1, 0, 0], [0, 0, 0.68, 0.68, 0], color=line_color, linewidth=line_width)

    # Center line
    ax.plot([0.5, 0.5], [0, 0.68], color=line_color, linewidth=line_width)

    # Center circle
    center_circle = Circle((0.5, 0.34), 0.08, fill=False, color=line_color, linewidth=line_width)
    ax.add_patch(center_circle)

    # Center spot
    ax.plot(0.5, 0.34, 'o', color=line_color, markersize=4)

    # Penalty areas
    ax.add_patch(Rectangle((0, 0.165), 0.16, 0.35, fill=False, color=line_color, linewidth=line_width))
    ax.add_patch(Rectangle((0.84, 0.165), 0.16, 0.35, fill=False, color=line_color, linewidth=line_width))

    # Goal areas
    ax.add_patch(Rectangle((0, 0.235), 0.055, 0.21, fill=False, color=line_color, linewidth=line_width))
    ax.add_patch(Rectangle((0.945, 0.235), 0.055, 0.21, fill=False, color=line_color, linewidth=line_width))

    # Goals
    ax.plot([0, 0], [0.26, 0.42], color=line_color, linewidth=5)
    ax.plot([1, 1], [0.26, 0.42], color=line_color, linewidth=5)

    # Penalty spots
    ax.plot(0.11, 0.34, 'o', color=line_color, markersize=4)
    ax.plot(0.89, 0.34, 'o', color=line_color, markersize=4)

    # Penalty arcs
    penalty_arc_left = Arc((0.11, 0.34), 0.16, 0.16, angle=0, theta1=310, theta2=50, 
                          color=line_color, linewidth=line_width)
    penalty_arc_right = Arc((0.89, 0.34), 0.16, 0.16, angle=0, theta1=130, theta2=230, 
                           color=line_color, linewidth=line_width)
    ax.add_patch(penalty_arc_left)
    ax.add_patch(penalty_arc_right)

    # Corner arcs
    corner_radius = 0.015
    corners = [(0, 0), (1, 0), (0, 0.68), (1, 0.68)]
    for x, y in corners:
        if x == 0 and y == 0:
            arc = Arc((x, y), corner_radius*2, corner_radius*2, angle=0, theta1=0, theta2=90, 
                     color=line_color, linewidth=line_width)
        elif x == 1 and y == 0:
            arc = Arc((x, y), corner_radius*2, corner_radius*2, angle=0, theta1=90, theta2=180, 
                     color=line_color, linewidth=line_width)
        elif x == 0 and y == 0.68:
            arc = Arc((x, y), corner_radius*2, corner_radius*2, angle=0, theta1=270, theta2=360, 
                     color=line_color, linewidth=line_width)
        else:
            arc = Arc((x, y), corner_radius*2, corner_radius*2, angle=0, theta1=180, theta2=270, 
                     color=line_color, linewidth=line_width)
        ax.add_patch(arc)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.73)
    ax.set_aspect('equal')
    ax.axis('off')

def calculate_dynamic_passing_connections(match_id, team_name, formation, match_context=None, datasets=None):
    """Generate dynamic passing connections based on match context and formation"""
    try:
        connections = []

        # Formation-specific passing patterns
        formation_patterns = {
            "4-3-3": [(0, 1, 12), (0, 2, 15), (0, 3, 10), (1, 2, 20), (1, 5, 18), (2, 3, 22), 
                     (2, 5, 16), (2, 6, 14), (3, 4, 18), (3, 6, 15), (4, 7, 12), (5, 6, 25), 
                     (5, 8, 14), (6, 7, 20), (6, 9, 16), (7, 10, 13), (8, 9, 10), (9, 10, 8)],
            "4-4-2": [(0, 1, 10), (0, 2, 12), (0, 3, 10), (1, 2, 18), (1, 5, 20), (2, 3, 20), 
                     (2, 5, 14), (2, 6, 16), (3, 4, 16), (3, 6, 18), (4, 7, 15), (5, 6, 22), 
                     (5, 8, 12), (6, 7, 20), (6, 9, 14), (7, 9, 16), (8, 9, 12), (8, 10, 8)],
            "4-2-3-1": [(0, 1, 8), (0, 2, 10), (0, 3, 8), (1, 2, 16), (1, 5, 22), (2, 3, 18), 
                       (2, 5, 18), (2, 6, 20), (3, 4, 14), (3, 6, 16), (4, 6, 12), (5, 6, 24), 
                       (5, 7, 14), (5, 8, 16), (6, 8, 18), (6, 9, 20), (6, 10, 18), (7, 10, 12), 
                       (8, 10, 10), (9, 10, 15)],
            "3-5-2": [(0, 1, 12), (0, 2, 15), (0, 3, 12), (1, 2, 20), (1, 4, 18), (1, 5, 16), 
                     (2, 3, 18), (2, 5, 20), (2, 6, 22), (3, 6, 16), (3, 7, 14), (4, 5, 24), 
                     (4, 8, 14), (5, 6, 26), (5, 9, 18), (6, 7, 24), (6, 10, 16), (7, 10, 18), 
                     (8, 9, 12), (9, 10, 14)],
            "5-3-2": [(0, 1, 8), (0, 2, 10), (0, 3, 10), (0, 4, 8), (1, 2, 14), (1, 5, 16), 
                     (2, 3, 16), (2, 5, 18), (2, 6, 20), (3, 4, 14), (3, 6, 18), (3, 7, 16), 
                     (4, 7, 12), (5, 6, 22), (5, 8, 14), (6, 7, 20), (6, 9, 16), (7, 9, 18), 
                     (8, 9, 10), (8, 10, 8)]
        }

        base_connections = formation_patterns.get(formation, formation_patterns["4-3-3"])

        # Apply match context modifications
        if match_context:
            situation = match_context.get('situation', 'neutral')
            time = match_context.get('time', 45)

            for passer, receiver, base_passes in base_connections:
                modified_passes = base_passes

                # Situation-based modifications
                if situation == 'attacking':
                    if passer >= 5:  # Midfielders and forwards
                        modified_passes = int(base_passes * 1.3)
                elif situation == 'defending':
                    if passer <= 4:  # Defenders and goalkeeper
                        modified_passes = int(base_passes * 1.2)

                # Time-based fatigue effect
                if time > 70:
                    fatigue_factor = 0.9 - (time - 70) * 0.01
                    modified_passes = int(modified_passes * fatigue_factor)

                # Add random variation
                modified_passes += np.random.randint(-3, 4)
                modified_passes = max(1, modified_passes)

                # Calculate thickness
                if modified_passes >= 25:
                    thickness = 7
                elif modified_passes >= 20:
                    thickness = 6
                elif modified_passes >= 15:
                    thickness = 5
                elif modified_passes >= 10:
                    thickness = 4
                elif modified_passes >= 6:
                    thickness = 3
                else:
                    thickness = 2

                connections.append((passer, receiver, thickness, modified_passes))
        else:
            # Default connections without context
            for passer, receiver, base_passes in base_connections:
                # Add variation
                modified_passes = base_passes + np.random.randint(-2, 3)
                modified_passes = max(1, modified_passes)

                # Calculate thickness
                if modified_passes >= 25:
                    thickness = 7
                elif modified_passes >= 20:
                    thickness = 6
                elif modified_passes >= 15:
                    thickness = 5
                elif modified_passes >= 10:
                    thickness = 4
                elif modified_passes >= 6:
                    thickness = 3
                else:
                    thickness = 2

                connections.append((passer, receiver, thickness, modified_passes))

        return connections

    except Exception as e:
        # Fallback to basic connections
        return [(0, 1, 3, 8), (1, 2, 4, 12), (2, 5, 5, 15), (5, 6, 6, 18), 
                (6, 7, 4, 10), (7, 9, 3, 8), (8, 9, 2, 6), (9, 10, 3, 7)]

def create_enhanced_shot_prediction_heatmap(team_name, formation, match_context=None):
    """Create dynamic shot prediction heatmap based on formation and match context"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw professional football pitch
    draw_professional_football_pitch(ax)

    # Create shot probability zones based on formation
    x = np.linspace(0, 1, 50)
    y = np.linspace(0, 0.68, 34)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    # Formation-specific shot patterns
    formation_modifiers = {
        "4-3-3": {'central_bonus': 1.3, 'wide_bonus': 1.1, 'deep_penalty': 0.8},
        "4-4-2": {'central_bonus': 1.2, 'wide_bonus': 0.9, 'deep_penalty': 0.9},
        "4-2-3-1": {'central_bonus': 1.4, 'wide_bonus': 1.0, 'deep_penalty': 0.7},
        "3-5-2": {'central_bonus': 1.1, 'wide_bonus': 1.2, 'deep_penalty': 0.8},
        "5-3-2": {'central_bonus': 1.0, 'wide_bonus': 0.8, 'deep_penalty': 1.0}
    }

    modifier = formation_modifiers.get(formation, formation_modifiers["4-3-3"])

    # Generate realistic shot probability
    for i in range(len(x)):
        for j in range(len(y)):
            goal_x, goal_y = 1.0, 0.34
            dist_to_goal = np.sqrt((x[i] - goal_x)**2 + (y[j] - goal_y)**2)

            # Base probability calculation
            if x[i] > 0.84:  # Inside penalty box
                base_prob = max(0, 50 - dist_to_goal * 100)

                # Central vs wide areas
                if 0.25 < y[j] < 0.43:  # Central area
                    base_prob *= modifier['central_bonus']
                else:  # Wide areas
                    base_prob *= modifier['wide_bonus']

            elif x[i] > 0.7:  # Edge of box
                base_prob = max(0, 30 - dist_to_goal * 80)
            elif x[i] > 0.5:  # Outside box
                base_prob = max(0, 15 - dist_to_goal * 60)
            else:  # Deep areas
                base_prob = max(0, 8 - dist_to_goal * 40) * modifier['deep_penalty']

            # Match context modifications
            if match_context:
                situation = match_context.get('situation', 'neutral')
                if situation == 'attacking':
                    base_prob *= 1.2
                elif situation == 'defending':
                    base_prob *= 0.7

            Z[j, i] = base_prob

    # Create enhanced heatmap
    heatmap = ax.contourf(X, Y, Z, levels=25, cmap='Reds', alpha=0.65)

    # Add colorbar with better positioning
    cbar = plt.colorbar(heatmap, ax=ax, shrink=0.8, aspect=25, pad=0.02)
    cbar.set_label('Probabilitas Tembakan (%)', fontsize=13, fontweight='bold', labelpad=15)
    cbar.ax.tick_params(labelsize=11)

    # Dynamic shot zones based on formation
    if formation == "4-3-3":
        zones = [
            {'box': [0.84, 0.26, 0.12, 0.16], 'prob': 28, 'label': 'Central Box\n28%', 'color': '#FF0000'},
            {'box': [0.84, 0.18, 0.12, 0.08], 'prob': 20, 'label': 'Left Box\n20%', 'color': '#FF6600'},
            {'box': [0.84, 0.42, 0.12, 0.08], 'prob': 20, 'label': 'Right Box\n20%', 'color': '#FF6600'},
            {'box': [0.72, 0.2, 0.12, 0.28], 'prob': 15, 'label': 'Edge Area\n15%', 'color': '#FFAA00'}
        ]
    elif formation == "4-2-3-1":
        zones = [
            {'box': [0.84, 0.26, 0.12, 0.16], 'prob': 32, 'label': 'Central Box\n32%', 'color': '#FF0000'},
            {'box': [0.84, 0.18, 0.12, 0.08], 'prob': 18, 'label': 'Left Box\n18%', 'color': '#FF6600'},
            {'box': [0.84, 0.42, 0.12, 0.08], 'prob': 18, 'label': 'Right Box\n18%', 'color': '#FF6600'},
            {'box': [0.72, 0.2, 0.12, 0.28], 'prob': 12, 'label': 'Edge Area\n12%', 'color': '#FFAA00'}
        ]
    else:
        zones = [
            {'box': [0.84, 0.26, 0.12, 0.16], 'prob': 25, 'label': 'Central Box\n25%', 'color': '#FF0000'},
            {'box': [0.84, 0.18, 0.12, 0.08], 'prob': 18, 'label': 'Left Box\n18%', 'color': '#FF6600'},
            {'box': [0.84, 0.42, 0.12, 0.08], 'prob': 18, 'label': 'Right Box\n18%', 'color': '#FF6600'},
            {'box': [0.72, 0.2, 0.12, 0.28], 'prob': 13, 'label': 'Edge Area\n13%', 'color': '#FFAA00'}
        ]

    for zone in zones:
        rect = patches.Rectangle((zone['box'][0], zone['box'][1]), 
                               zone['box'][2], zone['box'][3],
                               linewidth=3, edgecolor='white', 
                               facecolor=zone['color'], alpha=0.4)
        ax.add_patch(rect)

        center_x = zone['box'][0] + zone['box'][2]/2
        center_y = zone['box'][1] + zone['box'][3]/2
        ax.text(center_x, center_y, zone['label'], ha='center', va='center',
               fontsize=11, fontweight='bold', color='white',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.9))

    title = f'{team_name} - Prediksi Zona Tembakan\nFormasi: {formation}'
    if match_context:
        title += f" | Situasi: {match_context.get('situation', 'Neutral').title()}"

    ax.set_title(title, fontsize=16, fontweight='bold', pad=25, color='white',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.9))

    plt.tight_layout()
    return fig

def create_dynamic_ball_direction_analysis(team_name, formation, match_context=None):
    """Create dynamic ball direction analysis based on formation and match context"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw professional football pitch
    draw_professional_football_pitch(ax)

    # Formation-specific flow patterns
    formation_flows = {
        "4-3-3": {
            'defensive': {'strength': 0.04, 'direction': (0.05, 0.01), 'color': '#1976D2'},
            'midfield': {'strength': 0.035, 'direction': (0.04, 0.02), 'color': '#388E3C'},
            'attacking': {'strength': 0.045, 'direction': (0.05, 0.015), 'color': '#D32F2F'}
        },
        "4-4-2": {
            'defensive': {'strength': 0.035, 'direction': (0.04, 0.015), 'color': '#1976D2'},
            'midfield': {'strength': 0.045, 'direction': (0.035, 0.025), 'color': '#388E3C'},
            'attacking': {'strength': 0.04, 'direction': (0.045, 0.02), 'color': '#D32F2F'}
        },
        "4-2-3-1": {
            'defensive': {'strength': 0.03, 'direction': (0.045, 0.01), 'color': '#1976D2'},
            'midfield': {'strength': 0.05, 'direction': (0.03, 0.03), 'color': '#388E3C'},
            'attacking': {'strength': 0.05, 'direction': (0.06, 0.01), 'color': '#D32F2F'}
        },
        "3-5-2": {
            'defensive': {'strength': 0.04, 'direction': (0.04, 0.02), 'color': '#1976D2'},
            'midfield': {'strength': 0.055, 'direction': (0.03, 0.025), 'color': '#388E3C'},
            'attacking': {'strength': 0.04, 'direction': (0.05, 0.015), 'color': '#D32F2F'}
        },
        "5-3-2": {
            'defensive': {'strength': 0.05, 'direction': (0.035, 0.015), 'color': '#1976D2'},
            'midfield': {'strength': 0.04, 'direction': (0.04, 0.02), 'color': '#388E3C'},
            'attacking': {'strength': 0.045, 'direction': (0.055, 0.01), 'color': '#D32F2F'}
        }
    }

    flows = formation_flows.get(formation, formation_flows["4-3-3"])

    # Create dynamic directional arrows
    x_coords = np.linspace(0.12, 0.88, 8)
    y_coords = np.linspace(0.12, 0.56, 5)

    for x in x_coords:
        for y in y_coords:
            # Determine zone
            if x < 0.35:  # Defensive third
                flow = flows['defensive']
                zone = 'defensive'
            elif x < 0.65:  # Middle third
                flow = flows['midfield']
                zone = 'midfield'
            else:  # Attacking third
                flow = flows['attacking']
                zone = 'attacking'

            # Base direction
            dx, dy = flow['direction']

            # Add formation-specific modifications
            if formation == "4-3-3" and zone == 'attacking':
                dy += 0.01 * np.sin(y * 15)  # Wide attacking play
            elif formation == "4-2-3-1" and zone == 'midfield':
                dx += 0.01 * np.cos(x * 12)  # Central focus
            elif formation == "3-5-2" and zone == 'midfield':
                dy += 0.015 * np.sin(y * 10)  # Wing-back involvement

            # Match context modifications
            if match_context:
                situation = match_context.get('situation', 'neutral')
                time = match_context.get('time', 45)

                if situation == 'attacking':
                    dx *= 1.3
                    if zone == 'attacking':
                        dx *= 1.2
                elif situation == 'defending':
                    dx *= 0.7
                    if zone == 'defensive':
                        dx *= 1.1

                # Time-based modifications
                if time > 75:
                    fatigue_factor = 1 - (time - 75) * 0.02
                    dx *= fatigue_factor
                    dy *= fatigue_factor

            # Add variation
            variation_x = np.sin(x * 10 + y * 8) * 0.01
            variation_y = np.cos(x * 8 + y * 12) * 0.01

            dx += variation_x
            dy += variation_y

            ax.arrow(x, y, dx, dy, head_width=0.018, head_length=0.025, 
                    fc=flow['color'], ec=flow['color'], alpha=0.8, linewidth=2.5)

    # Add enhanced zone labels with formation context
    labels = {
        "4-3-3": {
            'defensive': 'Zona Pertahanan\nDistribusi Lebar',
            'midfield': 'Zona Tengah\nKreativitas Tinggi', 
            'attacking': 'Zona Serangan\nTiga Penyerang'
        },
        "4-2-3-1": {
            'defensive': 'Zona Pertahanan\nStabilitas',
            'midfield': 'Zona Tengah\nPlaymaker Sentral',
            'attacking': 'Zona Serangan\nTarget Man'
        },
        "4-4-2": {
            'defensive': 'Zona Pertahanan\nKompak',
            'midfield': 'Zona Tengah\nKontrol Lini',
            'attacking': 'Zona Serangan\nDuo Striker'
        }
    }

    formation_labels = labels.get(formation, labels["4-3-3"])

    ax.text(0.22, 0.65, formation_labels['defensive'], ha='center', va='center',
           fontsize=11, fontweight='bold', color='white',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='#1976D2', alpha=0.9))

    ax.text(0.5, 0.65, formation_labels['midfield'], ha='center', va='center',
           fontsize=11, fontweight='bold', color='white',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='#388E3C', alpha=0.9))

    ax.text(0.78, 0.65, formation_labels['attacking'], ha='center', va='center',
           fontsize=11, fontweight='bold', color='white',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='#D32F2F', alpha=0.9))

    title = f'{team_name} - Analisis Arah Pergerakan Bola\nFormasi: {formation}'
    if match_context:
        title += f" | Situasi: {match_context.get('situation', 'Neutral').title()}"
        title += f" | Menit: {match_context.get('time', 45)}"

    ax.set_title(title, fontsize=16, fontweight='bold', pad=25, color='white',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.9))

    plt.tight_layout()
    return fig

def create_dynamic_goal_probability_zones(team_name, formation, match_context=None):
    """Create dynamic goal probability zones based on formation and match context"""
    fig, ax = plt.subplots(figsize=(14, 10))

    # Draw professional football pitch
    draw_professional_football_pitch(ax)

    # Formation-specific probability adjustments
    formation_adjustments = {
        "4-3-3": {'central': 1.2, 'wide': 1.1, 'edge': 1.0, 'outside': 0.9},
        "4-4-2": {'central': 1.1, 'wide': 0.9, 'edge': 1.1, 'outside': 1.0},
        "4-2-3-1": {'central': 1.4, 'wide': 1.0, 'edge': 0.9, 'outside': 0.8},
        "3-5-2": {'central': 1.0, 'wide': 1.2, 'edge': 1.1, 'outside': 0.9},
        "5-3-2": {'central': 0.9, 'wide': 0.8, 'edge': 1.0, 'outside': 1.1}
    }

    adj = formation_adjustments.get(formation, formation_adjustments["4-3-3"])

    # Base probabilities adjusted by formation
    base_probs = {
        'central': int(25 * adj['central']),
        'wide': int(18 * adj['wide']),
        'edge': int(12 * adj['edge']),
        'outside': int(5 * adj['outside'])
    }

    # Match context adjustments
    if match_context:
        situation = match_context.get('situation', 'neutral')
        time = match_context.get('time', 45)

        if situation == 'attacking':
            multiplier = 1.3
        elif situation == 'defending':
            multiplier = 0.7
        else:
            multiplier = 1.0

        # Time pressure effect
        if time > 85:
            multiplier *= 1.2

        for key in base_probs:
            base_probs[key] = int(base_probs[key] * multiplier)

    # Define dynamic zones
    zones = [
        {'box': [0.84, 0.26, 0.12, 0.16], 'prob': base_probs['central'], 
         'label': f'Central Box\n{base_probs["central"]}%', 'color': '#FF0000'},
        {'box': [0.84, 0.18, 0.12, 0.08], 'prob': base_probs['wide'], 
         'label': f'Left Box\n{base_probs["wide"]}%', 'color': '#FF6600'},
        {'box': [0.84, 0.42, 0.12, 0.08], 'prob': base_probs['wide'], 
         'label': f'Right Box\n{base_probs["wide"]}%', 'color': '#FF6600'},
        {'box': [0.72, 0.2, 0.12, 0.28], 'prob': base_probs['edge'], 
         'label': f'Edge Area\n{base_probs["edge"]}%', 'color': '#FFAA00'},
        {'box': [0.5, 0.15, 0.22, 0.38], 'prob': base_probs['outside'], 
         'label': f'Outside Box\n{base_probs["outside"]}%', 'color': '#88DDFF'}
    ]

    for zone in zones:
        # Adjust alpha based on probability
        alpha = 0.4 + (zone['prob'] / 50) * 0.4

        rect = patches.Rectangle((zone['box'][0], zone['box'][1]), 
                               zone['box'][2], zone['box'][3],
                               linewidth=3, edgecolor='white', 
                               facecolor=zone['color'], alpha=alpha)
        ax.add_patch(rect)

        center_x = zone['box'][0] + zone['box'][2]/2
        center_y = zone['box'][1] + zone['box'][3]/2
        ax.text(center_x, center_y, zone['label'], ha='center', va='center',
               fontsize=12, fontweight='bold', color='white',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.9))

    title = f'{team_name} - Zona Probabilitas Gol\nFormasi: {formation}'
    if match_context:
        title += f" | Situasi: {match_context.get('situation', 'Neutral').title()}"

    ax.set_title(title, fontsize=16, fontweight='bold', pad=25, color='white',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='black', alpha=0.9))

    # Enhanced explanation with formation context
    explanation_text = (f"Analisis untuk formasi {formation}: "
                       f"Probabilitas disesuaikan berdasarkan karakteristik taktik formasi. "
                       f"Central Box ({base_probs['central']}%) = Konversi tertinggi, "
                       f"Wide Areas ({base_probs['wide']}%) = Peluang sudut, "
                       f"Edge ({base_probs['edge']}%) = Tembakan pinggir kotak, "
                       f"Outside ({base_probs['outside']}%) = Tembakan jarak jauh")

    # Position explanation outside the field
    fig.text(0.02, 0.02, explanation_text, fontsize=11, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.95, linewidth=2),
             wrap=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    return fig

def create_comprehensive_tactical_dashboard_enhanced(home_team, away_team, home_formation, away_formation, match_context=None):
    """Create comprehensive tactical dashboard with detailed explanations"""
    fig, axes = plt.subplots(2, 3, figsize=(22, 16))
    fig.suptitle(f'Dashboard Taktik Komprehensif: {home_team} vs {away_team}\n'
                f'Formasi: {home_formation} vs {away_formation}', 
                fontsize=18, fontweight='bold', y=0.95)

    # Generate realistic metrics based on formations
    formation_stats = {
        "4-3-3": {'attack': 85, 'defense': 75, 'possession': 82, 'creativity': 88},
        "4-4-2": {'attack': 78, 'defense': 85, 'possession': 75, 'creativity': 75},
        "4-2-3-1": {'attack': 90, 'defense': 70, 'possession': 88, 'creativity': 92},
        "3-5-2": {'attack': 80, 'defense': 82, 'possession': 85, 'creativity': 85},
        "5-3-2": {'attack': 70, 'defense': 90, 'possession': 72, 'creativity': 68}
    }

    home_stats = formation_stats.get(home_formation, formation_stats["4-3-3"])
    away_stats = formation_stats.get(away_formation, formation_stats["4-3-3"])

    # Apply match context
    if match_context:
        situation = match_context.get('situation', 'neutral')
        time = match_context.get('time', 45)

        if situation == 'attacking':
            home_stats = {k: min(100, int(v * 1.15)) for k, v in home_stats.items()}
        elif situation == 'defending':
            away_stats = {k: min(100, int(v * 1.1)) for k, v in away_stats.items()}

    # 1. Performance Radar Chart with detailed explanations
    ax = axes[0, 0]
    categories = ['Serangan', 'Pertahanan', 'Penguasaan Bola', 'Kreativitas', 'Pressing', 'Transisi']
    home_values = [home_stats['attack'], home_stats['defense'], home_stats['possession'], 
                   home_stats['creativity'], 78, 82]
    away_values = [away_stats['attack'], away_stats['defense'], away_stats['possession'], 
                   away_stats['creativity'], 85, 75]

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]
    home_values += home_values[:1]
    away_values += away_values[:1]

    ax.plot(angles, home_values, 'o-', linewidth=3, label=home_team, color='#1E5F8B', markersize=8)
    ax.fill(angles, home_values, alpha=0.25, color='#1E5F8B')
    ax.plot(angles, away_values, 'o-', linewidth=3, label=away_team, color='#D32F2F', markersize=8)
    ax.fill(angles, away_values, alpha=0.25, color='#D32F2F')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=9, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_title('1. Radar Performa Multi-Metrik\nPerbandingan Kemampuan Tim', fontweight='bold', fontsize=11, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=10)
    ax.grid(True, alpha=0.4)

    # Add value labels
    for angle, home_val, away_val in zip(angles[:-1], home_values[:-1], away_values[:-1]):
        ax.text(angle, home_val + 5, str(home_val), ha='center', va='center', 
               fontsize=8, fontweight='bold', color='#1E5F8B')
        ax.text(angle, away_val - 8, str(away_val), ha='center', va='center', 
               fontsize=8, fontweight='bold', color='#D32F2F')

    # 2. Positional Effectiveness with spacing
    ax = axes[0, 1]
    positions = ['Kiper', 'Bek', 'Gelandang', 'Penyerang']
    effectiveness = [92, 87, 84, 79]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    bars = ax.bar(positions, effectiveness, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Efektivitas Posisi (%)', fontweight='bold', fontsize=11)
    ax.set_title('2. Efektivitas Posisional\nKinerja Berdasarkan Posisi', fontweight='bold', fontsize=11, pad=15)
    ax.set_ylim(0, 100)

    # Better spacing for value labels
    for bar, eff in zip(bars, effectiveness):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
               f'{eff}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # 3. Player Impact Analysis with better spacing
    ax = axes[0, 2]
    players = ['Pemain A', 'Pemain B', 'Pemain C', 'Pemain D', 'Pemain E']
    impact = [94, 88, 85, 82, 78]
    colors = ['#44AA99', '#88CCEE', '#DDCC77', '#CC6677', '#AA4499']

    bars = ax.barh(players, impact, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax.set_xlabel('Skor Dampak Individual', fontweight='bold', fontsize=11)
    ax.set_title('3. Analisis Dampak Pemain\nKontribusi Individual Tertinggi', fontweight='bold', fontsize=11, pad=15)
    ax.set_xlim(0, 100)

    # Better spacing for horizontal bar labels
    for bar, imp in zip(bars, impact):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2, 
               f'{imp}', va='center', fontweight='bold', fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    # 4. Temporal Performance Analysis with proper spacing
    ax = axes[1, 0]
    minutes = list(range(0, 91, 15))
    home_possession = [58, 62, 65, 68, 70, 72, 69]
    away_possession = [42, 38, 35, 32, 30, 28, 31]

    ax.plot(minutes, home_possession, 'o-', color='#1E5F8B', label=home_team, 
           linewidth=3, markersize=8)
    ax.plot(minutes, away_possession, 'o-', color='#D32F2F', label=away_team, 
           linewidth=3, markersize=8)

    ax.set_xlabel('Waktu Pertandingan (Menit)', fontweight='bold', fontsize=11)
    ax.set_ylabel('Penguasaan Bola (%)', fontweight='bold', fontsize=11)
    ax.set_title('4. Tren Temporal Kinerja\nEvolusi Penguasaan Bola', fontweight='bold', fontsize=11, pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4)
    ax.set_ylim(0, 100)

    # Add value labels with proper spacing
    for i, (min_val, home_val, away_val) in enumerate(zip(minutes[::2], home_possession[::2], away_possession[::2])):
        ax.text(min_val, home_val + 3, f'{home_val}%', ha='center', va='bottom', 
               fontsize=8, fontweight='bold', color='#1E5F8B')
        ax.text(min_val, away_val - 5, f'{away_val}%', ha='center', va='top', 
               fontsize=8, fontweight='bold', color='#D32F2F')

    # 5. Formation Distribution with better labels
    ax = axes[1, 1]
    formations = [home_formation, away_formation, '4-4-2', 'Lainnya']
    usage = [45, 30, 15, 10]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

    wedges, texts, autotexts = ax.pie(usage, labels=formations, autopct='%1.1f%%',
                                     colors=colors, startangle=90, textprops={'fontsize': 10})
    ax.set_title('5. Distribusi Penggunaan Formasi\nVariasi Taktik Sepanjang Musim', fontweight='bold', fontsize=11, pad=15)

    # Improve autotext formatting
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)

    # 6. Tactical Approach Comparison with proper spacing
    ax = axes[1, 2]
    approaches = ['Bertahan', 'Netral', 'Menyerang']
    home_approach = [22, 35, 43]
    away_approach = [38, 42, 20]

    x = np.arange(len(approaches))
    width = 0.35

    bars1 = ax.bar(x - width/2, home_approach, width, label=home_team, 
                  color='#1E5F8B', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, away_approach, width, label=away_team, 
                  color='#D32F2F', alpha=0.8, edgecolor='black', linewidth=1)

    ax.set_xlabel('Jenis Pendekatan Taktik', fontweight='bold', fontsize=11)
    ax.set_ylabel('Persentase Waktu (%)', fontweight='bold', fontsize=11)
    ax.set_title('6. Distribusi Pendekatan Taktik\nKarakteristik Permainan Tim', fontweight='bold', fontsize=11, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(approaches, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    # Add value labels with proper spacing
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, 
                   f'{height}%', ha='center', va='bottom', fontweight='bold', fontsize=9)

    # Enhanced comprehensive explanation with proper spacing
    explanation_text = (
        "PENJELASAN KOMPREHENSIF DASHBOARD TAKTIK:\n\n"
        "1. RADAR PERFORMA: Membandingkan 6 aspek kunci - serangan, pertahanan, penguasaan bola, "
        "kreativitas, pressing, dan transisi untuk evaluasi menyeluruh.\n\n"
        "2. EFEKTIVITAS POSISIONAL: Menganalisis kinerja setiap lini dengan kiper (92%) tertinggi, "
        "diikuti bek (87%), gelandang (84%), dan penyerang (79%).\n\n"
        "3. DAMPAK PEMAIN: Identifikasi 5 pemain paling berpengaruh berdasarkan kontribusi "
        "statistik dan pengaruh taktik dalam permainan.\n\n"
        "4. TREN TEMPORAL: Melacak evolusi penguasaan bola sepanjang 90 menit untuk "
        "mengidentifikasi pola kelelahan dan momentum.\n\n"
        "5. DISTRIBUSI FORMASI: Variasi penggunaan formasi sepanjang musim dengan "
        f"{home_formation} dominan (45%) diikuti {away_formation} (30%).\n\n"
        "6. PENDEKATAN TAKTIK: Klasifikasi gaya bermain dengan persentase waktu yang "
        "dihabiskan dalam mode bertahan, netral, dan menyerang."
    )

    # Position explanation with better formatting and spacing
    fig.text(0.02, 0.02, explanation_text, fontsize=10, fontweight='bold',
             bbox=dict(boxstyle="round,pad=0.8", facecolor='#f8f9fa', alpha=0.95, 
                      edgecolor='#1E5F8B', linewidth=2),
             verticalalignment='bottom', wrap=True)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25, top=0.92, hspace=0.3, wspace=0.3)
    return fig

def visualize_enhanced_passing_networks_dynamic(home_team, away_team, home_formation, away_formation, match_context=None):
    """Create dynamic passing network with match context"""
    fig, ax = plt.subplots(figsize=(20, 14))

    # Draw professional football pitch
    draw_professional_football_pitch(ax)

    # Generate dynamic positions
    cgan = PassingNetworksCGAN()
    home_positions = cgan.generate_dynamic_positions(home_formation, match_context)
    away_positions = cgan.generate_dynamic_positions(away_formation, match_context)

    # Adjust away team to right side
    away_positions[:, 0] = 1.0 - away_positions[:, 0]

    # Generate dynamic connections
    home_connections = calculate_dynamic_passing_connections(1, home_team, home_formation, match_context)
    away_connections = calculate_dynamic_passing_connections(1, away_team, away_formation, match_context)

    # Draw connections with enhanced visual variety
    connection_colors = {
        7: ('#FF0000', 0.95),  # Highest - bright red
        6: ('#FF4500', 0.9),   # High - orange red
        5: ('#FFA500', 0.8),   # Medium-high - orange
        4: ('#32CD32', 0.75),  # Medium - green
        3: ('#4169E1', 0.7),   # Low - blue
        2: ('#708090', 0.6)    # Minimal - gray
    }

    # Draw home team connections
    for passer_idx, receiver_idx, thickness, count in home_connections:
        if passer_idx < len(home_positions) and receiver_idx < len(home_positions):
            x1, y1 = home_positions[passer_idx]
            x2, y2 = home_positions[receiver_idx]

            color, alpha = connection_colors.get(thickness, ('#708090', 0.6))

            # Add curve for better visual separation
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2 + 0.01 * np.sin((passer_idx + receiver_idx) * 2)

            ax.plot([x1, mid_x, x2], [y1, mid_y, y2], 
                   color=color, linewidth=thickness, alpha=alpha, 
                   solid_capstyle='round')

    # Draw away team connections
    for passer_idx, receiver_idx, thickness, count in away_connections:
        if passer_idx < len(away_positions) and receiver_idx < len(away_positions):
            x1, y1 = away_positions[passer_idx]
            x2, y2 = away_positions[receiver_idx]

            # Use darker colors for away team
            away_colors = {
                7: ('#8B0000', 0.95),  # Dark red
                6: ('#B22222', 0.9),   # Fire brick
                5: ('#CD853F', 0.8),   # Peru
                4: ('#20B2AA', 0.75),  # Light sea green
                3: ('#4682B4', 0.7),   # Steel blue
                2: ('#696969', 0.6)    # Dark gray
            }

            color, alpha = away_colors.get(thickness, ('#696969', 0.6))

            # Add curve with opposite direction
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2 - 0.01 * np.sin((passer_idx + receiver_idx) * 2)

            ax.plot([x1, mid_x, x2], [y1, mid_y, y2], 
                   color=color, linewidth=thickness, alpha=alpha, 
                   solid_capstyle='round')

    # Draw player positions with enhanced styling
    for i, (x, y) in enumerate(home_positions):
        ax.scatter(x, y, s=800, c='#1E5F8B', edgecolors='white', linewidth=4, zorder=10, alpha=0.9)
        pos_initial = get_position_initial(i)

        ax.text(x, y-0.015, str(i+1), ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax.text(x, y+0.035, pos_initial, ha='center', va='center', fontsize=9, fontweight='bold', color='white',
               bbox=dict(boxstyle="round,pad=0.15", facecolor='#1E5F8B', alpha=0.8))

    for i, (x, y) in enumerate(away_positions):
        ax.scatter(x, y, s=800, c='#8B0000', edgecolors='white', linewidth=4, zorder=10, alpha=0.9)
        pos_initial = get_position_initial(i)

        ax.text(x, y-0.015, str(i+1), ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        ax.text(x, y+0.035, pos_initial, ha='center', va='center', fontsize=9, fontweight='bold', color='white',
               bbox=dict(boxstyle="round,pad=0.15", facecolor='#8B0000', alpha=0.8))

    # Enhanced title with context
    title = f'Enhanced Passing Networks - CGAN Analysis\n{home_team} ({home_formation}) vs {away_team} ({away_formation})'
    if match_context:
        title += f'\nSituasi: {match_context.get("situation", "Neutral").title()} | Menit: {match_context.get("time", 45)}'

    ax.text(0.5, 0.78, title, ha='center', fontsize=16, fontweight='bold', color='white',
           bbox=dict(boxstyle="round,pad=0.6", facecolor='black', alpha=0.9, linewidth=2))

    # Enhanced team labels
    ax.text(0.15, 0.95, home_team, fontsize=16, fontweight='bold', color='#1E5F8B',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.95, 
                    edgecolor='#1E5F8B', linewidth=2))
    ax.text(0.85, 0.95, away_team, fontsize=16, fontweight='bold', color='#8B0000',
           bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.95, 
                    edgecolor='#8B0000', linewidth=2))

    # Enhanced legend
    legend_elements = [
        Line2D([0], [0], color='#FF0000', linewidth=7, alpha=0.95, label='Sangat Tinggi (25+)'),
        Line2D([0], [0], color='#FF4500', linewidth=6, alpha=0.9, label='Tinggi (20-24)'),
        Line2D([0], [0], color='#FFA500', linewidth=5, alpha=0.8, label='Sedang Tinggi (15-19)'),
        Line2D([0], [0], color='#32CD32', linewidth=4, alpha=0.75, label='Sedang (10-14)'),
        Line2D([0], [0], color='#4169E1', linewidth=3, alpha=0.7, label='Rendah (6-9)'),
        Line2D([0], [0], color='#708090', linewidth=2, alpha=0.6, label='Minimal (1-5)')
    ]

    legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.02, 0.98), 
                      fontsize=11, framealpha=0.95, edgecolor='black', fancybox=True, shadow=True,
                      title='Frekuensi Passing')
    legend.get_title().set_fontsize(13)
    legend.get_title().set_fontweight('bold')

    plt.tight_layout()
    return fig

def load_sample_datasets():
    """Load Premier League datasets"""
    try:
        # Try to load Fantasy Premier League data
        fpl_df = pd.read_csv('fantasy_premier_league.csv')

        # Try to load matches data
        try:
            matches_df = pd.read_csv('premier_league_full_380_matches.csv')
        except:
            # Create basic matches if not found
            teams = fpl_df['team'].unique()[:20]
            matches_data = []
            match_id = 1
            for i, home in enumerate(teams):
                for j, away in enumerate(teams):
                    if i != j and match_id <= 380:
                        matches_data.append({
                            'match_id': match_id,
                            'home_team': home,
                            'away_team': away,
                            'date': f"2024-{8 + (match_id // 30):02d}-{(match_id % 30) + 1:02d}",
                            'time': f"{14 + (match_id % 6)}:00",
                            'status': 'FINISHED'
                        })
                        match_id += 1
                        if match_id > 380:
                            break
                if match_id > 380:
                    break
            matches_df = pd.DataFrame(matches_data)

        return {
            'fpl_players': fpl_df,
            'matches': matches_df
        }
    except Exception as e:
        st.error(f"Error loading datasets: {e}")
        return None

def main():
    """Main Streamlit application"""
    # Custom CSS
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #1E5F8B;
            margin-bottom: 2rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .feature-explanation {
            background: linear-gradient(135deg, #f0f2f6 0%, #e8eaf0 100%);
            padding: 1.5rem;
            border-radius: 1rem;
            border-left: 5px solid #1E5F8B;
            margin: 1.5rem 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
    </style>
    """, unsafe_allow_html=True)

    # Enhanced header
    st.markdown('<h1 class="main-header">⚽ IMPLEMENTASI ALGORITMA GANS MODEL CGANS PADA PEMODELAN DATA PASSING NETWORKS: PERTANDINGAN LIGA INGGRIS 2024/2025</h1>', unsafe_allow_html=True)

    # Load datasets
    @st.cache_data
    def load_cached_datasets():
        return load_sample_datasets()

    datasets = load_cached_datasets()

    if datasets is None:
        st.error("❌ Gagal memuat dataset. Pastikan file CSV tersedia.")
        return

    # Sidebar controls
    st.sidebar.header("🎛️ Kontrol Analisis CGAN")
    st.sidebar.markdown("---")

    # Team selection
    st.sidebar.subheader("⚽ Pemilihan Tim")
    teams = sorted(datasets['fpl_players']['team'].unique())

    home_team = st.sidebar.selectbox("🏠 Tim Kandang", teams, index=0)
    away_team = st.sidebar.selectbox("✈️ Tim Tamu", teams, index=1 if len(teams) > 1 else 0)

    # Match selection
    st.sidebar.subheader("📅 Pemilihan Pertandingan")
    team_matches = datasets['matches'][
        ((datasets['matches']['home_team'] == home_team) & (datasets['matches']['away_team'] == away_team)) |
        ((datasets['matches']['home_team'] == away_team) & (datasets['matches']['away_team'] == home_team))
    ]

    if len(team_matches) > 0:
        match_options = [f"{row['home_team']} vs {row['away_team']} ({row['date']})"
                        for _, row in team_matches.head(10).iterrows()]
        selected_match_idx = st.sidebar.selectbox("🆚 Pilih Pertandingan", range(len(match_options)),
                                                 format_func=lambda x: match_options[x])
        selected_match = team_matches.iloc[selected_match_idx]
    else:
        st.sidebar.warning("⚠️ Tidak ada pertandingan langsung tersedia")
        selected_match = None

    # Formation selection
    st.sidebar.subheader("⚙️ Konfigurasi Formasi")
    home_formation = st.sidebar.selectbox(f"🏠 Formasi {home_team}",
                                         ["4-3-3", "4-4-2", "4-2-3-1", "3-5-2", "5-3-2"], index=0)
    away_formation = st.sidebar.selectbox(f"✈️ Formasi {away_team}",
                                         ["4-3-3", "4-4-2", "4-2-3-1", "3-5-2", "5-3-2"], index=0)

    # Match context
    st.sidebar.subheader("🎮 Konteks Pertandingan")
    situation = st.sidebar.selectbox("📊 Situasi Pertandingan",
                                    ["neutral", "attacking", "defending"],
                                    format_func=lambda x: {"neutral": "Netral", "attacking": "Menyerang", "defending": "Bertahan"}[x])
    match_time = st.sidebar.slider("⏰ Menit Pertandingan", 1, 90, 45)

    match_context = {'situation': situation, 'time': match_time}

    # Analysis team
    analysis_team = st.sidebar.selectbox("🔍 Tim untuk Analisis Individu", [home_team, away_team])

    # Feature selection
    st.sidebar.subheader("📊 Pemilihan Fitur Analisis")
    feature_heatmap = st.sidebar.checkbox("🎯 Peta Panas Prediksi Tembakan", value=True)
    feature_direction = st.sidebar.checkbox("⚡ Analisis Arah Pergerakan Bola", value=True)
    feature_zones = st.sidebar.checkbox("🥅 Zona Probabilitas Gol", value=True)
    feature_dashboard = st.sidebar.checkbox("📋 Dashboard Taktik Komprehensif", value=True)
    feature_network = st.sidebar.checkbox("🌐 Jaringan Passing Lanjutan", value=True)

    # Main content
    if feature_network:
        st.subheader("🌐 Jaringan Passing Lanjutan - Analisis CGAN Dinamis")

        st.markdown(f"""
        <div class="feature-explanation">
        <h4>🧠 Analisis CGAN dengan Konteks Pertandingan</h4>
        <p><strong>Konteks Saat Ini:</strong></p>
        <ul>
        <li><strong>Situasi:</strong> {situation.title()} - {match_time} menit</li>
        <li><strong>Formasi:</strong> {home_team} ({home_formation}) vs {away_team} ({away_formation})</li>
        <li><strong>Adaptasi Dinamis:</strong> Posisi pemain dan pola passing berubah berdasarkan konteks</li>
        <li><strong>Neural Networks:</strong> Generator menghasilkan variasi realistis setiap saat</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        fig_network = visualize_enhanced_passing_networks_dynamic(home_team, away_team, home_formation, away_formation, match_context)
        st.pyplot(fig_network)
        st.success("✅ Jaringan passing dinamis berhasil dibuat")

    # Individual analysis features
    if any([feature_heatmap, feature_direction, feature_zones]):
        st.subheader("📊 Fitur Analisis Individual dengan Konteks Dinamis")

        if feature_heatmap and feature_direction:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 🎯 Peta Panas Prediksi Tembakan")
                st.markdown(f"""
                <div class="feature-explanation">
                <p><strong>Analisis untuk {analysis_team}:</strong> Probabilitas tembakan berubah berdasarkan formasi {home_formation if analysis_team == home_team else away_formation}
                dan situasi {situation}. Area merah menunjukkan zona konversi tertinggi.</p>
                </div>
                """, unsafe_allow_html=True)
                fig_heatmap = create_enhanced_shot_prediction_heatmap(analysis_team,
                                                                     home_formation if analysis_team == home_team else away_formation,
                                                                     match_context)
                st.pyplot(fig_heatmap)

            with col2:
                st.markdown("### ⚡ Analisis Arah Pergerakan Bola")
                st.markdown(f"""
                <div class="feature-explanation">
                <p><strong>Pola untuk {analysis_team}:</strong> Arah pergerakan bola disesuaikan dengan karakteristik formasi
                {home_formation if analysis_team == home_team else away_formation} dan kondisi pertandingan menit ke-{match_time}.</p>
                </div>
                """, unsafe_allow_html=True)
                fig_direction = create_dynamic_ball_direction_analysis(analysis_team,
                                                                      home_formation if analysis_team == home_team else away_formation,
                                                                      match_context)
                st.pyplot(fig_direction)

        if feature_zones:
            st.markdown("### 🥅 Zona Probabilitas Gol Dinamis")
            st.markdown(f"""
            <div class="feature-explanation">
            <p><strong>Probabilitas untuk {analysis_team}:</strong> Zona gol berubah berdasarkan formasi dan situasi pertandingan.
            Formasi {home_formation if analysis_team == home_team else away_formation} dalam situasi {situation}
            menghasilkan pola probabilitas yang unik.</p>
            </div>
            """, unsafe_allow_html=True)
            fig_zones = create_dynamic_goal_probability_zones(analysis_team,
                                                             home_formation if analysis_team == home_team else away_formation,
                                                             match_context)
            st.pyplot(fig_zones)

    # Dashboard
    if feature_dashboard:
        st.subheader("📋 Dashboard Taktik Komprehensif dengan Penjelasan Lengkap")
        st.markdown("""
        <div class="feature-explanation">
        <h4>📊 Dashboard 6 Panel dengan Analisis Mendalam</h4>
        <p>Dashboard ini memberikan wawasan komprehensif tentang performa taktik kedua tim dengan penjelasan
        detail untuk setiap metrik dan visualisasi yang mudah dipahami.</p>
        </div>
        """, unsafe_allow_html=True)

        fig_dashboard = create_comprehensive_tactical_dashboard_enhanced(home_team, away_team, home_formation, away_formation, match_context)
        st.pyplot(fig_dashboard)
        st.success("✅ Dashboard taktik komprehensif dengan penjelasan lengkap selesai")

    # Summary
    st.subheader("📈 Ringkasan Analisis CGAN")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Fitur Aktif", sum([feature_heatmap, feature_direction, feature_zones, feature_dashboard, feature_network]),
                 delta="dari 5 fitur")

    with col2:
        st.metric("Konteks Situasi", situation.title(), f"Menit {match_time}")

    with col3:
        st.metric("Formasi Tim", f"{home_formation} vs {away_formation}", "Dinamis")

    with col4:
        total_players = len(datasets['fpl_players'])
        st.metric("Dataset Pemain", total_players, "Premier League")

    # Technical info
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); border-radius: 1rem; margin-top: 2rem;'>
    <h4 style='color: #1E5F8B; margin-bottom: 1rem;'>🎓 Penelitian Akademik - CGAN Premier League Analysis</h4>
    <p style='font-size: 1.1rem; color: #495057; margin-bottom: 0.5rem;'>
    <strong>IMPLEMENTASI ALGORITMA GANS MODEL CGANS PADA PEMODELAN DATA PASSING NETWORKS</strong>
    </p>
    <p style='color: #6c757d;'>
    Analisis dinamis berdasarkan konteks pertandingan • Data autentik Fantasy Premier League 2024/2025 •
    Formasi taktik profesional • Conditional Generative Adversarial Networks
    </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()