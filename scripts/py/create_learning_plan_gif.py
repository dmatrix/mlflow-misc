#!/usr/bin/env env python3
"""
Create a futuristic animated GIF for MLflow 3.x GenAI Learning Plan

This script generates an animated visualization showing the 8 learning modules
with a cyberpunk/futuristic aesthetic featuring glowing effects, data streams,
and smooth transitions.

Requirements:
    pip install pillow numpy matplotlib
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os
from pathlib import Path


def create_gradient_background(width, height, colors):
    """Create a futuristic gradient background."""
    img = Image.new('RGB', (width, height))
    draw = ImageDraw.Draw(img)

    # Create vertical gradient
    for y in range(height):
        ratio = y / height

        # Interpolate between colors
        if ratio < 0.5:
            t = ratio * 2
            r = int(colors[0][0] * (1 - t) + colors[1][0] * t)
            g = int(colors[0][1] * (1 - t) + colors[1][1] * t)
            b = int(colors[0][2] * (1 - t) + colors[1][2] * t)
        else:
            t = (ratio - 0.5) * 2
            r = int(colors[1][0] * (1 - t) + colors[2][0] * t)
            g = int(colors[1][1] * (1 - t) + colors[2][1] * t)
            b = int(colors[1][2] * (1 - t) + colors[2][2] * t)

        draw.line([(0, y), (width, y)], fill=(r, g, b))

    return img


def add_glow_effect(img, intensity=2):
    """Add a glow effect to bright areas."""
    # Create glow mask from bright areas
    glow = img.filter(ImageFilter.GaussianBlur(radius=10))

    # Blend original with glow
    return Image.blend(img, glow, alpha=0.3)


def draw_glowing_circle(draw, center, radius, color, glow_intensity=3):
    """Draw a circle with glowing effect."""
    # Draw multiple circles with decreasing opacity for glow
    for i in range(glow_intensity, 0, -1):
        alpha = int(255 * (1 - i / (glow_intensity + 1)))
        r = radius + i * 3
        glow_color = tuple(list(color) + [alpha])
        bbox = [center[0] - r, center[1] - r, center[0] + r, center[1] + r]
        draw.ellipse(bbox, fill=color, outline=None)


def draw_connection_line(draw, start, end, color, width=2):
    """Draw a glowing connection line between two points."""
    # Draw thicker line for glow effect
    draw.line([start, end], fill=color, width=width + 4)
    # Draw bright center line
    bright_color = tuple(min(c + 100, 255) for c in color)
    draw.line([start, end], fill=bright_color, width=width)


def create_module_frame(frame_num, total_frames=16):
    """Create a single frame of the animation."""
    width, height = 1200, 800

    # Futuristic color palette (RGB)
    bg_colors = [
        (10, 5, 30),      # Deep purple
        (20, 10, 50),     # Mid purple
        (15, 5, 40)       # Dark purple
    ]

    # Create base image with gradient
    img = create_gradient_background(width, height, bg_colors)

    # Create drawing layer with transparency for glow effects
    overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Module information
    modules = [
        "Module 1: Fundamentals",
        "Module 2: Tracing",
        "Module 3: Prompts",
        "Module 4: Agents",
        "Module 5: RAG",
        "Module 6: Evaluation",
        "Module 7: Production",
        "Module 8: Advanced"
    ]

    # Colors for each module (cyan to magenta gradient)
    module_colors = [
        (0, 255, 255),    # Cyan
        (64, 224, 255),   # Sky blue
        (128, 128, 255),  # Blue-purple
        (192, 64, 255),   # Purple
        (255, 0, 255),    # Magenta
        (255, 64, 192),   # Pink
        (255, 128, 128),  # Light red
        (255, 192, 64)    # Orange-gold
    ]

    # Calculate which modules to highlight based on frame
    progress = frame_num / total_frames
    active_module = int(progress * len(modules))

    # Draw circular progress path
    center_x, center_y = width // 2, height // 2
    radius = 280

    # Draw connection lines first (background)
    for i in range(len(modules)):
        angle_current = (i / len(modules)) * 2 * np.pi - np.pi / 2
        angle_next = ((i + 1) / len(modules)) * 2 * np.pi - np.pi / 2

        x1 = center_x + int(radius * np.cos(angle_current))
        y1 = center_y + int(radius * np.sin(angle_current))
        x2 = center_x + int(radius * np.cos(angle_next))
        y2 = center_y + int(radius * np.sin(angle_next))

        # Dim color for inactive connections
        color = module_colors[i] if i < active_module else tuple(c // 3 for c in module_colors[i])
        draw_connection_line(draw, (x1, y1), (x2, y2), color, width=3)

    # Draw module nodes
    for i, module in enumerate(modules):
        angle = (i / len(modules)) * 2 * np.pi - np.pi / 2

        x = center_x + int(radius * np.cos(angle))
        y = center_y + int(radius * np.sin(angle))

        # Determine if this module is active
        is_active = i <= active_module
        is_current = i == active_module

        # Node appearance
        if is_current:
            node_radius = 25 + int(10 * np.sin(frame_num * 0.5))  # Pulsing effect
            node_color = module_colors[i]
        elif is_active:
            node_radius = 20
            node_color = module_colors[i]
        else:
            node_radius = 15
            node_color = tuple(c // 4 for c in module_colors[i])  # Very dim

        # Draw glowing node
        for glow_ring in range(5, 0, -1):
            alpha = int(80 * (1 - glow_ring / 6))
            glow_radius = node_radius + glow_ring * 4
            bbox = [x - glow_radius, y - glow_radius, x + glow_radius, y + glow_radius]
            glow_color = tuple(list(node_color) + [alpha])
            draw.ellipse(bbox, fill=glow_color)

        # Draw solid node
        bbox = [x - node_radius, y - node_radius, x + node_radius, y + node_radius]
        draw.ellipse(bbox, fill=node_color)

        # Draw inner highlight
        highlight_color = tuple(min(c + 100, 255) for c in node_color)
        inner_bbox = [x - node_radius//2, y - node_radius//2,
                     x + node_radius//2, y + node_radius//2]
        draw.ellipse(inner_bbox, fill=highlight_color)

    # Composite overlay onto base image
    base_rgba = img.convert('RGBA')
    combined = Image.alpha_composite(base_rgba, overlay)
    img = combined.convert('RGB')

    # Add text overlay
    draw = ImageDraw.Draw(img)

    # Try to use a nice font, fall back to default if not available
    try:
        title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 60)
        subtitle_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 36)
        module_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 28)
    except:
        try:
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 60)
            subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
            module_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
        except:
            title_font = ImageFont.load_default()
            subtitle_font = ImageFont.load_default()
            module_font = ImageFont.load_default()

    # Draw title
    title = "MLflow 3.x GenAI & Agents"
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text((width // 2 - title_width // 2, 50), title,
             fill=(0, 255, 255), font=title_font)

    # Draw subtitle in a hexagonal badge
    subtitle = "Learning Journey"
    subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
    subtitle_height = subtitle_bbox[3] - subtitle_bbox[1]

    # Hexagon badge dimensions
    badge_center_x = width // 2
    badge_center_y = center_y  # Center of the circular path
    badge_width = subtitle_width + 60
    badge_height = subtitle_height + 30

    # Create hexagon points (6 points)
    hex_points = []
    for i in range(6):
        angle = i * np.pi / 3 - np.pi / 2  # Start from top
        x = badge_center_x + int(badge_width * 0.6 * np.cos(angle))
        y = badge_center_y + int(badge_height * 0.8 * np.sin(angle))
        hex_points.append((x, y))

    # Draw hexagon with glow effect
    for glow_size in range(6, 0, -1):
        glow_alpha = int(60 * (1 - glow_size / 7))
        glow_color = (255, 0, 255, glow_alpha)  # Magenta glow

        # Expand hexagon for glow
        expanded_points = []
        for x, y in hex_points:
            dx = x - badge_center_x
            dy = y - badge_center_y
            factor = 1 + (glow_size * 0.05)
            expanded_points.append((
                badge_center_x + int(dx * factor),
                badge_center_y + int(dy * factor)
            ))

        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.polygon(expanded_points, fill=glow_color, outline=glow_color)
        img_rgba = img.convert('RGBA')
        img = Image.alpha_composite(img_rgba, overlay).convert('RGB')

    # Draw solid hexagon background
    draw = ImageDraw.Draw(img)
    draw.polygon(hex_points, fill=(40, 20, 80), outline=(255, 0, 255), width=3)

    # Draw subtitle text centered in hexagon
    text_x = badge_center_x - subtitle_width // 2
    text_y = badge_center_y - subtitle_height // 2
    draw.text((text_x, text_y), subtitle,
             fill=(255, 128, 255), font=subtitle_font)

    # Draw current module name
    if active_module < len(modules):
        current_module = modules[active_module]
        module_bbox = draw.textbbox((0, 0), current_module, font=module_font)
        module_width = module_bbox[2] - module_bbox[0]
        draw.text((width // 2 - module_width // 2, height - 100), current_module,
                 fill=module_colors[active_module], font=module_font)

    # Add "floating" data particles for futuristic effect
    np.random.seed(frame_num)  # Consistent randomness per frame
    for _ in range(20):
        px = np.random.randint(0, width)
        py = np.random.randint(0, height)
        particle_size = np.random.randint(1, 4)
        particle_alpha = np.random.randint(100, 200)

        # Cyan particles
        overlay = Image.new('RGBA', (width, height), (0, 0, 0, 0))
        particle_draw = ImageDraw.Draw(overlay)
        particle_draw.ellipse([px, py, px + particle_size, py + particle_size],
                            fill=(0, 255, 255, particle_alpha))
        img_rgba = img.convert('RGBA')
        img = Image.alpha_composite(img_rgba, overlay).convert('RGB')

    return img


def create_animated_gif(output_path='images/mlflow_genai_learning_plan.gif',
                       num_frames=16, duration=200, loop=0):
    """
    Create the complete animated GIF.

    Args:
        output_path: Path to save the GIF
        num_frames: Number of frames in the animation
        duration: Duration of each frame in milliseconds
        loop: Number of loops (0 = infinite)
    """
    print(f"🎨 Generating futuristic MLflow 3.x GenAI Learning Plan GIF...")
    print(f"📊 Creating {num_frames} frames...")

    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all frames
    frames = []
    for i in range(num_frames):
        print(f"  Frame {i + 1}/{num_frames}...", end='\r')
        frame = create_module_frame(i, num_frames)
        frames.append(frame)

    print(f"\n✅ All frames generated!")

    # Save as animated GIF
    print(f"💾 Saving animated GIF to {output_path}...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
        optimize=False
    )

    print(f"✨ GIF created successfully!")
    print(f"📏 Size: {os.path.getsize(output_path) / 1024:.1f} KB")
    print(f"🎬 Frames: {num_frames}, Duration: {duration}ms per frame")

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Generate MLflow GenAI Learning Plan GIF')
    parser.add_argument('--output', default='images/mlflow_genai_learning_plan.gif',
                       help='Output path for GIF')
    parser.add_argument('--frames', type=int, default=16,
                       help='Number of frames (default: 16)')
    parser.add_argument('--duration', type=int, default=200,
                       help='Duration per frame in ms (default: 200)')

    args = parser.parse_args()

    create_animated_gif(
        output_path=args.output,
        num_frames=args.frames,
        duration=args.duration
    )
