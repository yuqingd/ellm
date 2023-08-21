import numpy as np
import torch
from .plot import Image


BG_COLOR = '#000000'
SLOT_COLORS = [
    '#1f77b4',  #tab:blue
    '#ff7f0e',  #tab:orange
    '#2ca02c',  #tab:green
    '#d62728',  #tab:red
    '#9467bd',  #tab:purple
    '#8c564b',  #tab:brown
    '#e377c2',  #tab:pink
    '#7f7f7f',  #tab:gray
    '#bcbd22',  #tab:olive
    '#17becf',  #tab:cyan
    '#ffffff',  #tab:white
]


def color_hex_to_int(hex_color):
    h = hex_color.lstrip('#')
    return np.asarray(tuple(int(h[i:i+2], 16) for i in (0, 2, 4)), dtype=np.int32)


def color_hex_to_float(hex_color):
    color_int = color_hex_to_int(hex_color)
    return color_int / 255.


def post_process_output_imgs(imgs, detach=True, renormalize=True):
    # convert NCHW to NHWC, clamp, and transfer to CPU
    imgs = imgs.transpose(-1, -3).transpose(-3, -2)
    # Clamp to [0, 1] interval
    # imgs = imgs.clamp(0, 1)
    if renormalize:
        imgs = (imgs * 0.5 + 0.5).clamp(0, 1)
    # Transfer to cpu and (optionally) detach gradient
    imgs = imgs.cpu()
    if detach:
        imgs = imgs.detach()
    imgs = imgs.numpy()
    # RGB2BGR for plot
    imgs = np.flip(imgs, -1)
    return imgs


def append_border_pixels(imgs, color):
    """Assumes last 3 dimensions of tensors are images HWC."""
    # Create pad_width for np.pad
    border_pixel_width = 2
    pad_width = [(0, 0) for _ in range(imgs.ndim)]
    pad_width[-3] = (border_pixel_width, border_pixel_width)
    pad_width[-2] = (border_pixel_width, border_pixel_width)
    pad_width = tuple(pad_width)
    
    # Pad with 1s
    imgs = np.pad(imgs, pad_width=pad_width, constant_values=1)

    # Use the appropriate color for the border
    color = color_hex_to_float(color)
    imgs[..., :border_pixel_width, :, :] = color
    imgs[...,-border_pixel_width:, :, :] = color
    imgs[..., :, :border_pixel_width, :] = color
    imgs[..., :,-border_pixel_width:, :] = color
    return imgs


def combine_slot_masks(slot_masks):
    """Assumes last 3 dimensions of tensors are images HWC."""
    shp = list(slot_masks.shape)
    shp[-1] = 3
    slot_imgs = np.zeros(shp, dtype=np.float32)
    
    # Assumes -4 is the slot dimension
    for i in range(shp[-4]):
        slot_imgs[..., i, :, :, :] = color_hex_to_float(SLOT_COLORS[i])
        # slot_imgs[..., i, :, :, :] *= slot_masks[..., i, :, :, :]
    
    combined_imgs = slot_imgs * slot_masks
    combined_imgs = combined_imgs.sum(axis=-4)
    return combined_imgs


def concat_imgs_in_rec_mask_slots_in_a_row(img_in, img_rec, img_slots, img_slot_masks, img_slot_masks_multiplied):
    # generate mask by combining slot masks depending on their RGB-color coded values
    img_combined_mask = combine_slot_masks(img_slot_masks)

    # first append color borders - black to input, rec, and mask - RGB-coded colors to slot images
    # and then sequentially concatenate (append) them along axis=1
    # input image
    img_res = append_border_pixels(img_in, BG_COLOR)
    
    # append reconstructed image
    img_tmp = append_border_pixels(img_rec, BG_COLOR)
    img_res = np.concatenate((img_res, img_tmp), axis=-2)
    
    # append combined slot masks image
    img_tmp = append_border_pixels(img_combined_mask, BG_COLOR)
    img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # append individual slot images
    for i in range(img_slots.shape[-4]):
        # note: the slicing is (this) ugly due to not knowing if the input is a sequence (B, T, H, W, C)
        img_tmp = append_border_pixels(img_slots[..., i, :, :, :], SLOT_COLORS[i])
        img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # append individual slot masks
    imgs = np.repeat(img_slot_masks, 3, axis=-1)
    for i in range(img_slot_masks.shape[-4]):
        img_tmp = append_border_pixels(imgs[:, i], SLOT_COLORS[i])
        img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # append individual slot masks
    for i in range(img_slot_masks_multiplied.shape[-4]):
        img_tmp = append_border_pixels(img_slot_masks_multiplied[..., i, :, :, :], SLOT_COLORS[i])
        img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # img_res = torch.from_numpy(img_res)  # TODO: fix! do everything in torch instead of np?
    return img_res


def concat_imgs_in_rec_mask_slots_in_a_row_CRAFTER(img_in, img_rec, img_slot_masks, img_slot_masks_multiplied):
    # generate mask by combining slot masks depending on their RGB-color coded values
    img_combined_mask = combine_slot_masks(img_slot_masks)

    # first append color borders - black to input, rec, and mask - RGB-coded colors to slot images
    # and then sequentially concatenate (append) them along axis=1
    # input image
    img_res = append_border_pixels(img_in, BG_COLOR)

    # append reconstructed image
    img_tmp = append_border_pixels(img_rec, BG_COLOR)
    img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # append combined slot masks image
    img_tmp = append_border_pixels(img_combined_mask, BG_COLOR)
    img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # append individual slot masks
    imgs = np.repeat(img_slot_masks, 3, axis=-1)
    for i in range(img_slot_masks.shape[-4]):
        img_tmp = append_border_pixels(imgs[:, i], SLOT_COLORS[i])
        img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # append individual slot masks
    for i in range(img_slot_masks_multiplied.shape[-4]):
        img_tmp = append_border_pixels(img_slot_masks_multiplied[..., i, :, :, :], SLOT_COLORS[i])
        img_res = np.concatenate((img_res, img_tmp), axis=-2)

    # img_res = torch.from_numpy(img_res)  # TODO: fix! do everything in torch instead of np?
    return img_res


def batch_to_rowwise_image(imgs):
    # Flatten the first (batch/time) and the second dimension (time/H)
    imgs = imgs.reshape(-1, *imgs.shape[2:])
    return imgs


def batch_to_rowwise_video(videos):
    # converts a Tensor of shape (B, T, H, W, C) to (1, T, B*H, W, C)
    videos = np.swapaxes(videos, 0, 1)
    shp = videos.shape
    videos = videos.reshape(shp[0], 1, np.prod(shp[1:3]), *shp[3:])
    videos = np.swapaxes(videos, 0, 1)
    return videos

