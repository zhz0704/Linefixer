import numpy as np
from PIL import Image
import os, tqdm

# --- 1D Perlin Noise Implementation ---
class PerlinNoise1D:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.p = np.arange(256, dtype=int)
        np.random.shuffle(self.p)
        self.p = np.concatenate((self.p, self.p))

    def fade(self, t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def grad(self, hash_val, x):
        return (1 - ((hash_val % 2) << 1)) * x

    def noise(self, x):
        X = int(np.floor(x))
        x -= X
        u = self.fade(x)
        X = X & 255
        g0 = self.grad(self.p[X], x)
        g1 = self.grad(self.p[X + 1], x - 1)
        return (1.0 - u) * g0 + u * g1

# --- Bézier Curve Function ---
def cubic_bezier(t, P0, P1, P2, P3):
    t_sq = t * t
    t_cub = t_sq * t
    one_minus_t = 1 - t
    one_minus_t_sq = one_minus_t * one_minus_t
    one_minus_t_cub = one_minus_t_sq * one_minus_t

    x = (one_minus_t_cub * P0[0] + 
         3 * one_minus_t_sq * t * P1[0] + 
         3 * one_minus_t * t_sq * P2[0] + 
         t_cub * P3[0])
    
    y = (one_minus_t_cub * P0[1] + 
         3 * one_minus_t_sq * t * P1[1] + 
         3 * one_minus_t * t_sq * P2[1] + 
         t_cub * P3[1])
         
    return np.array([x, y])

# --- Core Algorithm to Generate Coords ---
def generate_pixel_coords(control_points, num_points=100, canvas_size=64, 
                          noise_amplitude=2.0, noise_frequency=3.0, seed=None):
    """
    Generates a list of integer (x, y) coordinates for a single-pixel line
    on a small canvas, including Perlin noise for realistic wobble.
    """
    if seed is not None:
        np.random.seed(seed)
    
    P0, P1, P2, P3 = control_points
    perlin = PerlinNoise1D(seed=seed)
    
    t_values = np.linspace(0, 1, num_points)

    raw_coords = [] # Store raw coordinates before rounding and unique filtering
    
    for t_val in t_values:
        base_point = cubic_bezier(t_val, P0, P1, P2, P3)
        
        epsilon = 0.01
        tangent = cubic_bezier(min(t_val + epsilon, 1), P0, P1, P2, P3) - base_point
        norm = np.linalg.norm(tangent)
        if norm > 1e-6:
            tangent /= norm
        else:
            tangent = np.array([1, 0]) 
        
        normal = np.array([-tangent[1], tangent[0]])

        noise_value = perlin.noise(t_val * noise_frequency)
        
        offset = normal * noise_value * noise_amplitude
        
        final_point_float = base_point + offset
        
        x_int = int(np.round(final_point_float[0]))
        y_int = int(np.round(final_point_float[1]))

        x_clamped = np.clip(x_int, 0, canvas_size - 1)
        y_clamped = np.clip(y_int, 0, canvas_size - 1)
        
        raw_coords.append((x_clamped, y_clamped))

    # Convert to a set to remove duplicates, then back to a list
    unique_coords = list(set(raw_coords))
    
    return unique_coords

def post_process_image(img: Image.Image) -> Image.Image:
    pixels = np.array(img)

    pixels_to_change = []
    width, height = img.size
    for x in range(width):
        for y in range(height):
            if pixels[x, y] != 255:
                continue
            # if bottom is not white and left is not white AND the bottom left is white, make the pixel black
            if y < height - 1 and x > 0:
                bottom_pixel = pixels[x, y + 1]
                left_pixel = pixels[x - 1, y]
                bottom_left_pixel = pixels[x - 1, y + 1]
                if bottom_pixel != 255 and left_pixel != 255 and bottom_left_pixel == 255:
                    pixels_to_change.append((x, y))
            # if bottom is not white and right is not white AND the bottom right is white, make the pixel black
            if y < height - 1 and x < width - 1:
                bottom_pixel = pixels[x, y + 1]
                right_pixel = pixels[x + 1, y]
                bottom_right_pixel = pixels[x + 1, y + 1]
                if bottom_pixel != 255 and right_pixel != 255 and bottom_right_pixel == 255:
                    pixels_to_change.append((x, y))
    # print(pixels_to_change)
    for (x, y) in pixels_to_change:
        pixels[x, y] = 0
    return Image.fromarray(pixels)

# --- Function to draw and save the image ---
def draw_and_save_image(coords_list, canvas_size, filepath, post_process=True):
    """
    Draws the given list of (x, y) coordinates onto a canvas and saves it as an image.
    The line will be white on a black background.
    """
    # Create a black canvas (0 for black)
    canvas_array = np.full((canvas_size, canvas_size), 255, dtype=np.uint8)

    # Set specified coordinates to white (255)
    for x, y in coords_list:
        canvas_array[y, x] = 0 # Image indexing is typically [row (y), col (x)]

    # Create a PIL Image from the NumPy array
    img = Image.fromarray(canvas_array)
    img.save(filepath)

    if post_process:
        img2 = post_process_image(img)
        # print the pixels:
        # print(np.array(img) == np.array(img2))
        img2.save(filepath.replace('broken', 'fixed'))


import time

# --- Example Usage ---
if __name__ == '__main__':
    training_size = 10000
    val_size = 2000
    test_size = 1000
    canvas_size = 64
    output_dir = './train/broken'
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm.tqdm(range(training_size), desc="generating training lines"):
        # Generate random control points for each line
        P0 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        P1 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        P2 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        P3 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        control_points_rand = [P0, P1, P2, P3]

        random_coords_list = generate_pixel_coords(
            control_points_rand, 
            num_points=200,
            canvas_size=canvas_size,
            noise_amplitude=np.random.uniform(1.0, 4.0),
            noise_frequency=np.random.uniform(2.0, 5.0),
            seed=int(time.time()) + i # Unique seed for each random line
        )
        
        rand_image_filename = os.path.join(output_dir, f"{i:03d}.png")
        draw_and_save_image(random_coords_list, canvas_size, rand_image_filename)
        # print(f"Saved {rand_image_filename}")

    output_dir = './val/broken'
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm.tqdm(range(val_size), desc="generating validation lines"):
        # Generate random control points for each line
        P0 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        P1 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        P2 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        P3 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        control_points_rand = [P0, P1, P2, P3]

        random_coords_list = generate_pixel_coords(
            control_points_rand, 
            num_points=200,
            canvas_size=canvas_size,
            noise_amplitude=np.random.uniform(1.0, 4.0),
            noise_frequency=np.random.uniform(2.0, 5.0),
            seed=int(time.time()) + i # Unique seed for each random line
        )
        
        rand_image_filename = os.path.join(output_dir, f"{i:03d}.png")
        draw_and_save_image(random_coords_list, canvas_size, rand_image_filename)
        # print(f"Saved {rand_image_filename}")
    
    output_dir = './test/broken/random_lines'
    os.makedirs(output_dir, exist_ok=True)
    for i in tqdm.tqdm(range(test_size), desc="generating test lines"):
        # Generate random control points for each line
        P0 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        P1 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        P2 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        P3 = np.random.uniform([0, 0], [canvas_size, canvas_size])
        control_points_rand = [P0, P1, P2, P3]

        random_coords_list = generate_pixel_coords(
            control_points_rand, 
            num_points=200,
            canvas_size=canvas_size,
            noise_amplitude=np.random.uniform(1.0, 4.0),
            noise_frequency=np.random.uniform(2.0, 5.0),
            seed=int(time.time()) + i # Unique seed for each random line
        )

        rand_image_filename = os.path.join(output_dir, f"{i:03d}.png")
        draw_and_save_image(random_coords_list, canvas_size, rand_image_filename, post_process=False)
        # print(f"Saved {rand_image_filename}")
