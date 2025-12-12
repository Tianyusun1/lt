import os
from PIL import Image, ImageDraw, ImageFont
from typing import List, Tuple, Dict

# 类别定义 (必须与 dataset.py 中的 ID 2-10 保持一致)
CLASS_COLORS = {
    2: "red",      # mountain
    3: "blue",     # water
    4: "green",    # people
    5: "brown",    # tree
    6: "yellow",   # building
    7: "cyan",     # bridge
    8: "magenta",  # flower
    9: "orange",   # bird
    10: "lime"     # animal
}

CLASS_NAMES = {
    2: "mountain", 3: "water", 4: "people", 5: "tree",
    6: "building", 7: "bridge", 8: "flower", 9: "bird", 10: "animal"
}

def draw_layout(
    layout_seq: List[Tuple], 
    poem: str,
    output_path: str, 
    img_size: Tuple[int, int] = (512, 512)
):
    """
    Draws the predicted layout onto a blank white canvas and saves it.
    
    Args:
        layout_seq: List of (cls_id, cx, cy, w, h). cls_id is 2-10.
        poem: The input text (used as title).
        output_path: Path to save the PNG image.
        img_size: (width, height) of the canvas.
    """
    try:
        # Create a white canvas
        img = Image.new('RGB', img_size, 'white')
        draw = ImageDraw.Draw(img)
        W, H = img_size
        
        # Try to load a font (using default if not found)
        try:
            # 尝试加载一个通用字体
            font = ImageFont.truetype("arial.ttf", 10) 
        except IOError:
            font = ImageFont.load_default()
            
        # Draw all bounding boxes
        for item in layout_seq:
            if len(item) != 5: continue
            
            # cls_id 必须是整数
            cls_id, cx, cy, w, h = item
            cls_id = int(cls_id)
            
            # Convert normalized YOLO format (cx, cy, w, h) to pixel (xmin, ymin, xmax, ymax)
            xmin = int((cx - w / 2) * W)
            ymin = int((cy - h / 2) * H)
            xmax = int((cx + w / 2) * W)
            ymax = int((cy + h / 2) * H)
            
            color = CLASS_COLORS.get(cls_id, "black")
            cls_name = CLASS_NAMES.get(cls_id, "Unknown")
            
            # Draw the box
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=2)
            
            # Draw class label
            label_text = f"{cls_name} ({cls_id})"
            draw.text((xmin + 2, ymin + 2), label_text, fill=color, font=font)
        
        # Add poem/title
        draw.text((10, 10), f"Input: {poem}", fill="black", font=font)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path)
        
    except ImportError:
        print("\n[Warning]: PIL/Pillow library not found. Skipping visualization. Please run 'pip install Pillow'.")
    except Exception as e:
        print(f"\n[Error]: During layout drawing for poem '{poem}': {e}. Skipping visualization.")