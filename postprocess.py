from PIL import Image

def post_process_image(img: Image.Image) -> Image.Image:
    pixels = img.load()

    pixels_to_change = []
    width, height = img.size
    for x in range(width):
        for y in range(height):
            if pixels[x, y] != (255, 255, 255):
                continue
            # if bottom is not white and left is not white AND the bottom left is white, make the pixel black
            if y < height - 1 and x > 0:
                bottom_pixel = pixels[x, y + 1]
                left_pixel = pixels[x - 1, y]
                bottom_left_pixel = pixels[x - 1, y + 1]
                if bottom_pixel != (255, 255, 255) and left_pixel != (255, 255, 255) and bottom_left_pixel == (255, 255, 255):
                    pixels_to_change.append((x, y))
            # if bottom is not white and right is not white AND the bottom right is white, make the pixel black
            if y < height - 1 and x < width - 1:
                bottom_pixel = pixels[x, y + 1]
                right_pixel = pixels[x + 1, y]
                bottom_right_pixel = pixels[x + 1, y + 1]
                if bottom_pixel != (255, 255, 255) and right_pixel != (255, 255, 255) and bottom_right_pixel == (255, 255, 255):
                    pixels_to_change.append((x, y))
    
    for (x, y) in pixels_to_change:
        pixels[x, y] = (0, 0, 0)
    
    return img

import os, tqdm
input_dir = "./train/broken"
output_dir = "./train/fixed"

for filename in tqdm.tqdm(os.listdir(input_dir), desc="Postprocessing training images"):
    if filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        img = img.convert("RGB")
        processed_img = post_process_image(img)
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        processed_img.save(output_path)

input_dir = "./val/broken"
output_dir = "./val/fixed"

for filename in tqdm.tqdm(os.listdir(input_dir), desc="Postprocessing validation images"):
    if filename.endswith(".png"):
        img_path = os.path.join(input_dir, filename)
        img = Image.open(img_path)
        img = img.convert("RGB")
        processed_img = post_process_image(img)
        output_path = os.path.join(output_dir, filename)
        os.makedirs(output_dir, exist_ok=True)
        processed_img.save(output_path)