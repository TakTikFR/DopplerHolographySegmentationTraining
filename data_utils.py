import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
# import model_utils
import os
# import cv2
from fastai.vision.all import *

class BinaryVesselDataset:
    def __init__(self, hf_split, input=["M0"], size=(512, 512)):
        self.data = []
        for sample in hf_split:
            x = np.array(sample[input[0]])                    # already preprocessed numpy
            y = np.array(np.array(sample["maskArtery"]).astype(bool) | np.array(sample["maskVein"]).astype(bool))
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = sample[0]  # PIL Image
        y = sample[1]  # PIL Image
        return PILImage.create(x.astype(np.uint8)), PILMask.create(y.astype(np.uint8))

class ArteryVeinDataset:
    def __init__(self, hf_split, input=["M0", "correlation", "diasys"], one_hot = False, size=(512, 512)):
        self.data = []
        for sample in hf_split:
            x = np.zeros((size[0], size[1], len(input)), dtype=np.uint8)
            for i, col in enumerate(input):
                x[:,:,i] = np.array(sample[col].convert("L").resize(size, Image.BILINEAR))
            artery = np.array(sample["maskArtery"].convert("L").resize(size, Image.NEAREST))
            vein = np.array(sample["maskVein"].convert("L").resize(size, Image.NEAREST))
            if one_hot:
                y = np.stack([artery, vein], axis=0)
            else:
                y = np.zeros((size[0], size[1]), dtype=np.uint8)
                y[artery > 0] = 1
                y[vein > 0] +=2
            self.data.append((x, y))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        x = sample[0]  # PIL Image
        y = sample[1]  # PIL Image
        return PILImage.create(x.astype(np.uint8)), y.astype(np.uint8)
    

# def multi2onehot(x:np.ndarray, # Non one-hot encoded targs
#         axis:int=2 # The axis to stack for encoding (class dimension)
#     ) -> np.ndarray:
#         "Creates one binary mask per class"
#         return np.stack([np.where((x==1) | (x==3), 1, 0), np.where((x==2) | (x==3), 1, 0)], axis=axis)

def multi2onehot_tensor(x:torch.Tensor, # Non one-hot encoded targs
        dim:int=2 # The axis to stack for encoding (class dimension)
    ) -> torch.Tensor:
        "Creates one binary mask per class"
        return torch.stack([torch.where((x==1) | (x==3), 1, 0), torch.where((x==2) | (x==3), 1, 0)], dim=dim)

# def mask_to_rgb(mask):
#     # Create an RGB image

#     if len(mask.shape) == 2:
#         one_hot_masks = multi2onehot(mask) # Convert to one-hot encoding
#     else:
#         one_hot_masks = mask.transpose(1, 2, 0)  # Change shape to [H, W, C]

#     rgb_image = np.zeros((one_hot_masks.shape[0], one_hot_masks.shape[1], 3), dtype=np.uint8)  # Shape: [H, W, 3]

#     # print(one_hot_masks.shape)
    
#     # Map the first mask to the red channel and the second to the blue channel
#     rgb_image[..., 0] = one_hot_masks[:,:,0] * 255  # Red channel
#     rgb_image[..., 2] = one_hot_masks[:,:,1] * 255  # Blue channel

#     return Image.fromarray(rgb_image)

# def split_channels(inputs, channels):
#     lists = [[] for _ in range(len(inputs))]
#     for i in range(len(inputs)):
#         for c in range(channels):
#             lists[i].append(inputs[i][c,:,:])
#     return lists
    
# def show_masks(inputs, masks, masks_pred=None, multi=False, cmap='viridis', n=20):
#     nb_rows = min(len(inputs), n)
#     channels = 1
#     # plot images and masks

#     if cmap == 'gray':
#         a = inputs[0]
#         channels = 1 if len(a.shape) == 2 else a.shape[0]
#         if channels != 1:
#             inputs = split_channels(inputs, channels)
    
#     nb_cols = channels + (1 if masks_pred is None else 2)
#     fig, axes = plt.subplots(nb_rows, nb_cols, figsize=(5*nb_cols, 5*nb_rows))
    
#     for idx in range(nb_rows):
#         for c in range(channels):
#             axes[idx][c].imshow(Image.fromarray(inputs[idx][c]*255), cmap=cmap)
#             # axes[idx][c].set_title(inputs[idx][c])
#             axes[idx][channels].imshow(mask_to_rgb(masks[idx]) if multi else masks[idx], cmap="gray")
#         if masks_pred is not None:
#             axes[idx][channels+1].imshow(mask_to_rgb(masks_pred[idx]) if multi else masks_pred[idx][0], cmap="gray")
        
#     # add subtitles
#     for c in range(channels):
#         axes[0][c].set_title('Input')
#     axes[0][channels].set_title('Ground truth masks')
#     if masks_pred is not None:
#         axes[0][channels + 1].set_title('Predicted masks')

#     plt.show()

# def predict_and_show(model, val_loader, cmap='viridis', multi=None, argmax=False, onnx_input_name=None, n=20):
#     # predict masks
#     masks_pred = []
#     inputs = []
#     targets = []
#     multi = multi
#     if onnx_input_name is not None:
#         if isinstance(model, str):
#             model = model_utils.load_onnx_model(model)
#     x,y = next(iter(val_loader))
#     for input, target in iter(val_loader):
#         mask = model(input.cuda()) if onnx_input_name is None else torch.Tensor(model.run(None, {onnx_input_name: input.cpu().numpy()})[0])
#         multi = mask.shape[1] > 1
#         if argmax:
#             mask = torch.argmax(mask, dim=1)
#         else:  # If we have several class, we need to apply a sigmoid and get the predictions
#             mask = torch.sigmoid(mask)
#             mask[mask<0.5] = 0
#             mask[mask>=0.5] = 1
#         inputs.append(input.squeeze(0).cpu().numpy())
#         masks_pred.append(mask.squeeze(0).cpu().detach().numpy())
#         targets.append(target.squeeze(0).cpu().numpy())
#     show_masks(inputs, targets, masks_pred, multi=multi, cmap=cmap, n=n)

# def normalize_image_np(image, mean=0, std=1.0):
#     max_pixel_value = image.max()
#     return (image - mean * max_pixel_value) / (std * max_pixel_value)

# def predict_and_save(model_path, in_dataset, out_dataset, size=(512, 512)):
#     image_path = in_dataset / "data"
#     # mask_path = in_dataset / "masks"

#     measure_list = os.listdir(image_path)

#     model = model_utils.load_onnx_model(model_path)

#     for measure in measure_list:
#         image = np.array(Image.open(image_path / measure).resize(size, Image.BILINEAR)).transpose((2,0,1))
#         # print(image.shape)                         
#         # mask = np.array(Image.open(mask_path / measure))
#         image = normalize_image_np(image, mean=0, std=1)
#         image = torch.Tensor(image).unsqueeze(0)
#         out = model.run(None, {'input': image.cpu().numpy()})[0]
#         pred = torch.argmax(torch.Tensor(out), dim=1).squeeze(0).cpu().numpy()

#         cv2.imwrite(str(out_dataset / measure), pred.astype(np.uint8))

# def get_avi(avi_path):
#     cap = cv2.VideoCapture(avi_path)
#     frames=[]
#     ret = True
#     while ret:
#         ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
#         if ret:
#             frames.append(img[:,:,0])
#             # frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
#     return np.stack(frames, axis=0) # dimensions (T, H, W, C)

# import cv2
# import numpy as np

# def save_numpy_video_as_avi(video: np.ndarray, filename: str, fps: int = 10):
#     """
#     Saves a NumPy video array to an AVI file using OpenCV.

#     Parameters:
#         video (np.ndarray): Shape (T, H, W) for grayscale, or (T, H, W, 3) for RGB.
#         filename (str): Path to output .avi file.
#         fps (int): Frame rate.
#     """
#     T = video.shape[0]
#     is_color = video.ndim == 4

#     H, W = video.shape[1:3]
#     fourcc = cv2.VideoWriter_fourcc(*'XVID')
#     out = cv2.VideoWriter(filename, fourcc, fps, (W, H), isColor=True)

#     for t in range(T):
#         frame = video[t]
        
#         # Normalize and convert to uint8 if needed
#         if frame.dtype != np.uint8:
#             frame = (normalize_image_np(frame) * 255).astype(np.uint8)
        
#         # Convert grayscale to BGR
#         if not is_color:
#             frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
#         else:
#             frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#         out.write(frame)

#     out.release()
#     print(f"Saved video to {filename}")
