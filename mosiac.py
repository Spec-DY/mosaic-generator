import os
import numpy as np
from PIL import Image
import gradio as gr
from pathlib import Path
from sklearn.cluster import KMeans
from skimage.metrics import structural_similarity as ssim

POKEMON_IMAGES = "pokemon_all"


class PokemonMosaic:
    def __init__(self, pokemon_dir, tile_size=32, n_colors=16):
        self.tile_size = tile_size
        self.n_colors = n_colors
        self.pokemon_tiles = []
        self.pokemon_colors = []

        script_dir = os.path.dirname(os.path.abspath(__file__))
        pokemon_path = os.path.join(script_dir, pokemon_dir)

        print(f"Looking for Pokemon images in: {pokemon_path}")

        if not os.path.exists(pokemon_path):
            raise ValueError(f"Pokemon directory not found: {pokemon_path}")

        png_files = list(Path(pokemon_path).glob("*.png"))
        print(f"Found {len(png_files)} PNG files")

        for img_path in png_files:
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img = img.resize((tile_size, tile_size))
                img_array = np.array(img)
                quantized_img = self._color_quantization(img_array)

                avg_color = np.mean(quantized_img, axis=(0, 1))

                self.pokemon_tiles.append(quantized_img)
                self.pokemon_colors.append(avg_color)

            except Exception as e:
                print(f"Error processing {img_path}: {e}")

        self.pokemon_colors = np.array(self.pokemon_colors)

        if not self.pokemon_tiles:
            raise ValueError(
                f"No valid Pokemon images found in: {pokemon_path}")

        print(f"Successfully loaded {len(self.pokemon_tiles)} Pokemon tiles")

    def _color_quantization(self, img_array):
        """Quantize colors using KMeans"""
        pixels = img_array.reshape(-1, 3)
        kmeans = KMeans(n_clusters=self.n_colors, random_state=0)
        labels = kmeans.fit_predict(pixels)
        quantized = kmeans.cluster_centers_[labels].reshape(img_array.shape)
        return quantized.astype(np.uint8)

    def _find_closest_pokemon(self, target_color):
        """Find closest Pokemon tile using color distance"""
        distances = np.sqrt(
            np.sum((self.pokemon_colors - target_color) ** 2, axis=1))
        return np.argmin(distances)

    def _calculate_metrics(self, original_img, mosaic_img):
        """Calculate MSE and SSIM between two images"""
        original_arr = np.array(original_img)
        mosaic_arr = np.array(mosaic_img)

        # MSE
        mse = np.mean((original_arr - mosaic_arr) ** 2)

        # SSIM
        ssim_value = ssim(original_arr, mosaic_arr,
                          channel_axis=2,
                          data_range=255)

        return {
            'MSE': mse,
            'SSIM': ssim_value
        }

    def create_mosaic(self, target_image_path, grid_size=16):
        """Create Pokemon mosaic"""

        target_img = Image.open(target_image_path)
        if target_img.mode != 'RGB':
            target_img = target_img.convert('RGB')

        w, h = target_img.size
        new_w = (w // grid_size) * grid_size
        new_h = (h // grid_size) * grid_size
        target_img = target_img.resize((new_w, new_h))
        target_array = np.array(target_img)

        quantized_target = self._color_quantization(target_array)

        # Create mosaic
        mosaic_w = new_w // grid_size * self.tile_size
        mosaic_h = new_h // grid_size * self.tile_size
        mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)

        for i in range(0, new_h, grid_size):
            for j in range(0, new_w, grid_size):

                grid_block = quantized_target[i:i+grid_size, j:j+grid_size]
                avg_color = np.mean(grid_block, axis=(0, 1))

                pokemon_idx = self._find_closest_pokemon(avg_color)
                pokemon_tile = self.pokemon_tiles[pokemon_idx]

                mosaic_i = i // grid_size * self.tile_size
                mosaic_j = j // grid_size * self.tile_size
                mosaic[mosaic_i:mosaic_i+self.tile_size,
                       mosaic_j:mosaic_j+self.tile_size] = pokemon_tile

        mosaic_img = Image.fromarray(mosaic)
        quantized_img = Image.fromarray(quantized_target)

        metrics = self._calculate_metrics(quantized_target,
                                          np.array(mosaic_img.resize(quantized_img.size)))

        return mosaic_img, quantized_img, metrics


def make_mosaic(input_image, grid_size=16, n_colors=16):
    try:
        if input_image is None:
            raise ValueError("Please upload an image")

        temp_path = "temp_target.png"
        input_image.save(temp_path)

        mosaic_maker = PokemonMosaic(f"{POKEMON_IMAGES}",
                                     tile_size=32,
                                     n_colors=n_colors)

        mosaic_img, quantized_img, metrics = mosaic_maker.create_mosaic(
            temp_path,
            grid_size=int(grid_size)
        )

        os.remove(temp_path)

        metrics_md = (f"Performance Metrics:\n"
                      f"MSE: {metrics['MSE']:.2f}\n"
                      f"SSIM: {metrics['SSIM']:.2f}")

        return mosaic_img, metrics_md

    except Exception as e:
        raise gr.Error(str(e))


# Gradio interface
iface = gr.Interface(
    fn=make_mosaic,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Slider(minimum=8, maximum=32, step=4, value=16, label="Grid Size"),
        gr.Slider(minimum=4, maximum=64, step=4,
                  value=16, label="Number of Colors")
    ],
    outputs=[
        gr.Image(label="Pokemon Mosaic"),
        gr.Markdown(label="Performance Metrics")
    ],
    title="Enhanced Pokemon Mosaic Generator",
    description="Create a mosaic using Pokemon sprites. Compare original, quantized, and final mosaic images."
)

if __name__ == "__main__":
    iface.launch()
