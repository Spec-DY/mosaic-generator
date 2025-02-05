import os 
import numpy as np
from PIL import Image
import gradio as gr
from pathlib import Path
from sklearn.cluster import KMeans

class PokemonMosaic:
    def __init__(self, pokemon_dir, tile_size=32):
        self.tile_size = tile_size
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
                print(f"Processing: {img_path}")
                img = Image.open(img_path)

                if img.mode != 'RGB':
                    img = img.convert('RGB')

                img = img.resize((tile_size, tile_size))

                img_array = np.array(img)

                main_color = self._get_main_color(img_array)
                
                self.pokemon_tiles.append(img_array)
                self.pokemon_colors.append(main_color)
                print(f"Successfully processed: {img_path}")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        self.pokemon_colors = np.array(self.pokemon_colors)
        
        print(f"Total processed Pokemon images: {len(self.pokemon_tiles)}")
        if not self.pokemon_tiles:
            raise ValueError(f"No valid Pokemon images found in: {pokemon_path}")
    
    def _get_main_color(self, img_array):

        pixels = img_array.reshape(-1, 3)

        kmeans = KMeans(n_clusters=1, random_state=0).fit(pixels)
        return kmeans.cluster_centers_[0]
    
    def _find_closest_pokemon(self, target_color):

        distances = np.sqrt(np.sum((self.pokemon_colors - target_color) ** 2, axis=1))

        return np.argmin(distances)
    
    def create_mosaic(self, target_image_path, grid_size=16):

        target_img = Image.open(target_image_path)

        if target_img.mode != 'RGB':
            target_img = target_img.convert('RGB')

        w, h = target_img.size
        new_w = (w // grid_size) * grid_size
        new_h = (h // grid_size) * grid_size
        target_img = target_img.resize((new_w, new_h))
        target_array = np.array(target_img)
        

        mosaic_w = new_w // grid_size * self.tile_size
        mosaic_h = new_h // grid_size * self.tile_size
        mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
        

        for i in range(0, new_h, grid_size):
            for j in range(0, new_w, grid_size):
                
                grid_block = target_array[i:i+grid_size, j:j+grid_size]
                avg_color = np.mean(grid_block, axis=(0,1))
                
                
                pokemon_idx = self._find_closest_pokemon(avg_color)
                pokemon_tile = self.pokemon_tiles[pokemon_idx]
                
                
                mosaic_i = i // grid_size * self.tile_size
                mosaic_j = j // grid_size * self.tile_size
                mosaic[mosaic_i:mosaic_i+self.tile_size, 
                      mosaic_j:mosaic_j+self.tile_size] = pokemon_tile
                
        return Image.fromarray(mosaic)

def make_mosaic(input_image, grid_size=16):
    try:
        if input_image is None:
            raise ValueError("Please upload an image")
            
        
        temp_path = "temp_target.png"
        input_image.save(temp_path)
        
        mosaic_maker = PokemonMosaic("pokemon_images", tile_size=32)
        
        result = mosaic_maker.create_mosaic(temp_path, grid_size=int(grid_size))
        
        os.remove(temp_path)
        
        return result
        
    except Exception as e:
        raise gr.Error(str(e))


iface = gr.Interface(
    fn=make_mosaic,
    inputs=[
        gr.Image(type="pil", label="Input Image"),
        gr.Slider(minimum=8, maximum=32, step=4, value=16, label="Grid Size")
    ],
    outputs=gr.Image(label="Mosaic Result"),
    title="Pokemon Mosaic Generator",
    description="Upload an image to create a mosaic using Pokemon sprites!"
)

if __name__ == "__main__":
    iface.launch()