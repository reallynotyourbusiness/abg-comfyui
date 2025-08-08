import huggingface_hub
import torch
import onnxruntime as rt
import numpy as np
import cv2

def get_mask(img:torch.Tensor, s=1024):
    img = (img / 255).astype(np.float32)
    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask

# Declare Execution Providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Download and host the model
model_path = huggingface_hub.hf_hub_download(
    "skytnt/anime-seg", "isnetis.onnx")
rmbg_model = rt.InferenceSession(model_path, providers=providers)

class RemoveImageBackgroundabg:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "processing_resolution": ("INT", {
                    "default": 1024,
                    "min": 256,
                    "max": 2048,
                    "step": 64
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("white_background", "transparent_background", "mask")
    FUNCTION = "abg_remover"
    CATEGORY = "image"

    def abg_remover(self, image: torch.Tensor, processing_resolution: int):
        imgs_np_bgr = image2nparray(image)

        white_bg_results = []
        transparent_bg_results = []
        mask_results = []

        for img_np_bgr in imgs_np_bgr:
            original_img_bgr = img_np_bgr[..., :3] if img_np_bgr.shape[-1] == 4 else img_np_bgr
            mask = get_mask(original_img_bgr, s=processing_resolution)
            white_bg_img_bgr = (mask * original_img_bgr + 255 * (1 - mask)).astype(np.uint8)
            transparent_bg_img_bgra = np.concatenate([original_img_bgr, (mask * 255).astype(np.uint8)], axis=2)
            mask_img_bgr = (mask * 255).astype(np.uint8).repeat(3, axis=2)

            white_bg_results.append(white_bg_img_bgr)
            transparent_bg_results.append(transparent_bg_img_bgra)
            mask_results.append(mask_img_bgr)

        white_bg_batch_np = np.stack(white_bg_results, axis=0)
        transparent_bg_batch_np = np.stack(transparent_bg_results, axis=0)
        mask_batch_np = np.stack(mask_results, axis=0)

        white_bg_batch_tensor = nparray2image(white_bg_batch_np)
        transparent_bg_batch_tensor = nparray2image(transparent_bg_batch_np)
        mask_batch_tensor = nparray2image(mask_batch_np)

        return (white_bg_batch_tensor, transparent_bg_batch_tensor, mask_batch_tensor)

def image2nparray(image:torch.Tensor):
    narray:np.array = np.clip(255. * image.cpu().numpy(), 0, 255).astype(np.uint8)
    if narray.shape[-1] == 4:
        narray =  narray[..., [2, 1, 0, 3]]
    else:
        narray = narray[..., [2, 1, 0]]
    return narray

def nparray2image(narray:np.array):
    print(f"narray shape: {narray.shape}")
    if narray.shape[-1] == 4:
        narray =  narray[..., [2, 1, 0, 3]]
    else:
        narray =  narray[..., [2, 1, 0]] 
    tensor = torch.from_numpy(narray/255.).float()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor

NODE_CLASS_MAPPINGS = {
    "Remove Image Background (abg)": RemoveImageBackgroundabg
}