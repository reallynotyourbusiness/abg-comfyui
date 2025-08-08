# Anime Background Remover for ComfyUI

This repository contains a custom ComfyUI node designed to remove the background from anime-style images. It intelligently segments the characters from the background, providing several useful outputs for creative workflows.

This node is based on the [anime-seg ONNX model](https://huggingface.co/skytnt/anime-seg) and is inspired by the original [ABG extension for AUTOMATIC1111](https://github.com/KutsuyaYuki/ABG_extension).

## Features

- **Batch Processing**: Process multiple images in a single operation.
- **Multiple Outputs**:
    - **White Background**: The character with the background replaced by solid white.
    - **Transparent Background**: The character on a transparent background (RGBA).
    - **Mask**: A black-and-white segmentation mask of the character.
- **Adjustable Resolution**: Control the internal processing resolution to balance speed and quality.

## Installation

1.  **Navigate to the `custom_nodes` directory**:
    Open a terminal or command prompt and navigate to your ComfyUI `custom_nodes` folder.
    ```bash
    cd path/to/ComfyUI/custom_nodes/
    ```

2.  **Clone the repository**:
    Run the following command to clone this repository into the `custom_nodes` directory.
    ```bash
    git clone https://github.com/Jcd1230/rembg-comfyui-node.git
    ```

3.  **Install Dependencies**:
    Activate your ComfyUI's virtual environment (if you are using one) and install the required Python packages from within the cloned repository's directory.
    ```bash
    cd rembg-comfyui-node
    pip install -r requirements.txt
    ```

4.  **Restart ComfyUI**:
    Restart ComfyUI, and the new node will be available.

## Usage

1.  In ComfyUI, add the **Remove Image Background (abg)** node.
2.  Connect an `IMAGE` output from another node (e.g., Load Image) to the `image` input of this node.
3.  The node provides the following inputs and outputs:

### Inputs

-   **`image`**: The input image or batch of images to process.
-   **`processing_resolution`**: The resolution used for internal processing. Higher values may yield better quality at the cost of VRAM and processing time. The default is `1024`.

### Outputs

1.  **`white_background`**: An `IMAGE` where the background has been replaced with solid white.
2.  **`transparent_background`**: An `IMAGE` with the character on a transparent background (RGBA). Use this for layering over other images.
3.  **`mask`**: A black and white `IMAGE` representing the segmentation mask of the character.

The node seamlessly handles batches of images. If you provide a batch as input, all three outputs will also be batches of the same size.
