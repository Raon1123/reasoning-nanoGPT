"""
This code visualize as follows:
- Dataset samples from the dataset used for training
- Model architecture summary
- Number of parameters in the model
- Computational graph of the model

"""
import os

# save image
import numpy as np
import matplotlib.pyplot as plt


def get_color_map(format='rgb'):
    """
    Color map from ARC-AGI challenge

    Returns a dictionary mapping symbol names to colors.
    format: 'hex' (default) returns hex color strings like '#0074D9'
            'rgb' returns RGB tuples like (0, 116, 217)
    """
    colors_hex = {
        'symbol_0': '#000000',
        'symbol_1': '#0074D9',  # blue
        'symbol_2': '#FF4136',  # red
        'symbol_3': '#2ECC40',  # green
        'symbol_4': '#FFDC00',  # yellow
        'symbol_5': '#AAAAAA',  # grey
        'symbol_6': '#F012BE',  # fuchsia
        'symbol_7': '#FF851B',  # orange
        'symbol_8': '#7FDBFF',  # teal
        'symbol_9': '#870C25',  # brown
        'symbol_pad': '#FFFFFF',  # white
        'symbol_eos': '#111111',  # dark grey
    }

    if format == 'hex':
        return colors_hex
    elif format == 'rgb':
        def hex_to_rgb(h):
            h = h.lstrip('#')
            if len(h) == 3:
                h = ''.join(ch * 2 for ch in h)
            return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
        return {k: hex_to_rgb(v) for k, v in colors_hex.items()}
    else:
        raise ValueError("format must be 'hex' or 'rgb'")
    
    
def get_rgb_image_from_symbols(symbol_array, color_map):
    """
    Convert a 2D array of symbol indices to an RGB image using the provided color map.
    
    symbol_array: 2D numpy array of shape (H, W) with integer symbol indices
    color_map: dictionary mapping symbol names to RGB tuples
    """
    # PAD: 0, <eos>: 1, digits: 2 ... 11
    h, w = symbol_array.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for symbol_idx in range(len(color_map)):
        symbol_name = f'symbol_{symbol_idx-2}' if symbol_idx > 1 else ('symbol_pad' if symbol_idx == 0 else 'symbol_eos')
        rgb_color = color_map[symbol_name]
        rgb_image[symbol_array == symbol_idx] = rgb_color
    
    return rgb_image


def main():
    color_map = get_color_map(format='rgb')
    
    data_dir = os.path.join('data', 'arc_agi', 'processed_data')
    
    trn_root = os.path.join(data_dir, 'train')
    tst_root = os.path.join(data_dir, 'test')
    
    trn_X = np.load(os.path.join(trn_root, 'all__inputs.npy'), mmap_mode='r')
    trn_y = np.load(os.path.join(trn_root, 'all__labels.npy'), mmap_mode='r')
    
    tst_X = np.load(os.path.join(tst_root, 'all__inputs.npy'), mmap_mode='r')
    tst_y = np.load(os.path.join(tst_root, 'all__labels.npy'), mmap_mode='r')
    
    # sample index
    sample_idx = 0
    sample_input = trn_X[sample_idx].reshape(30, 30)
    sample_label = trn_y[sample_idx].reshape(30, 30)
    
    input_image = get_rgb_image_from_symbols(sample_input, color_map)
    label_image = get_rgb_image_from_symbols(sample_label, color_map)
    
    # visualize and save
    visualize_path = configs['visualize_path']
    os.makedirs(visualize_path, exist_ok=True)
    
    # plotting with matplotlib with two images side by side
    # I want to show the grid as well
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    H, W = sample_input.shape
    for ax, img, title in zip(axes, (input_image, label_image), ('Sample Input', 'Sample Label')):
        ax.imshow(img, interpolation='nearest')
        ax.set_title(title)
        # draw grid lines between pixels
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        # hide tick labels
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(os.path.join(visualize_path, 'sample_input_label.png'))
    plt.close()
    
    # plotting test image with same way
    sample_idx = 0
    sample_input = tst_X[sample_idx].reshape(30, 30)
    sample_label = tst_y[sample_idx].reshape(30, 30)

    input_image = get_rgb_image_from_symbols(sample_input, color_map)
    label_image = get_rgb_image_from_symbols(sample_label, color_map)
    
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    H, W = sample_input.shape
    for ax, img, title in zip(axes, (input_image, label_image), ('Test Sample Input', 'Test Sample Label')):
        ax.imshow(img, interpolation='nearest')
        ax.set_title(title)
        # draw grid lines between pixels
        ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5)
        # hide tick labels
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.tight_layout()
    plt.savefig(os.path.join(visualize_path, 'test_sample_input_label.png'))
    plt.close()
    
    
    
    


if __name__ == '__main__':
    configs = {
        'visualize_path': './visualizations',
    }
    main()