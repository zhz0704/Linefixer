import os
from PIL import Image, ImageDraw

def simulate_paint_pencil(coords, filename="pencil_drawing.png", canvas_size=(64, 64), line_width=1, post_process=True):
    """
    Simulates a pencil tool drawing on a canvas and saves the image.

    Args:
        coords (list of tuple): A list of (x, y) coordinates to draw the line through.
        filename (str): The name of the file to save (e.g., "drawing.png").
        canvas_size (tuple): The (width, height) of the canvas in pixels.
        line_color (str): The color of the line (e.g., "black", "#FF0000").
        line_width (int): The thickness of the line in pixels.
    """
    # 1. Create a blank white canvas
    # 'RGB' mode for standard color image, (255, 255, 255) is white
    try:
        img = Image.new('RGB', canvas_size, color = (255, 255, 255))
        draw = ImageDraw.Draw(img)
    except Exception as e:
        print(f"Error creating image or drawing object: {e}")
        return

    # 2. Draw the line
    # ImageDraw.line requires a sequence of coordinates (x1, y1, x2, y2, x3, y3, ...)
    # which is flat. We convert the list of (x, y) tuples into a flattened list.
    # The 'pencil' effect is achieved by connecting all the points in the list.
    
    # Flatten the list of coordinates: [(x1, y1), (x2, y2), ...] -> [x1, y1, x2, y2, ...]
    flat_coords = [item for sublist in coords for item in sublist]
    
    if len(flat_coords) >= 4:  # Need at least two points (x1, y1, x2, y2) to draw a line
        try:
            draw.line(flat_coords, fill=(0, 0, 0), width=line_width)
        except Exception as e:
            print(f"Error drawing line: {e}")
            return
    else:
        print("Not enough coordinates provided to draw a line.")
        return

    if post_process:
        img = post_process_image(img)

    # 3. Save the image
    output_dir = "./images"
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    full_path = os.path.join(output_dir, filename)
    
    try:
        img.save(full_path)
        print(f"Drawing saved successfully to {full_path}")
    except Exception as e:
        print(f"Error saving image: {e}")

def post_process_image(img: Image.Image) -> Image.Image:
    pixels = img.load()

    pixels_to_change = []
    width, height = img.size
    for x in range(width):
        for y in range(height):
            if pixels[x, y] != (255, 255, 255):
                continue
            # if bottom is not white and left is not white, make the pixel black
            if y < height - 1 and x > 0:
                bottom_pixel = pixels[x, y + 1]
                left_pixel = pixels[x - 1, y]
                if bottom_pixel != (255, 255, 255) and left_pixel != (255, 255, 255):
                    pixels_to_change.append((x, y))
            # if bottom is not white and right is not white, make the pixel black
            if y < height - 1 and x < width - 1:
                bottom_pixel = pixels[x, y + 1]
                right_pixel = pixels[x + 1, y]
                if bottom_pixel != (255, 255, 255) and right_pixel != (255, 255, 255):
                    pixels_to_change.append((x, y))
    
    for (x, y) in pixels_to_change:
        pixels[x, y] = (0, 0, 0)
    
    return img


# --- Example Usage ---

# Simulate mouse movement coordinates (list of x, y tuples)
# These coordinates should be within the 64x64 bounds.
# Example: a simple diagonal line followed by a curve
mouse_coords = [
    (10, 10), 
    (15, 15), 
    (20, 20), 
    (30, 25),
    (40, 30),
    (45, 40),
    (50, 50)
]

# Run the simulation
simulate_paint_pencil(mouse_coords, line_color="#000000", line_width=1, post_process=True) 

# Example with a thicker line and different filename
# A simple square shape
square_coords = [
    (5, 5),
    (55, 5),
    (55, 55),
    (5, 55),
    (5, 5) # Connects back to the start
]

simulate_paint_pencil(square_coords, filename="thick_square.png", line_width=5, post_process=True)