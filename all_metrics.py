import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import torch
from torchvision import transforms
from PIL import Image
import lpips

def calculate_metrics(original_image_path, frames_folder, num_frames):
    # Load the original image
    original_img = cv2.imread(original_image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net='alex')

    # Initialize lists to store metric values
    psnr_values = []
    ssim_values = []
    lpips_values = []

    # Process each frame
    for i in range(num_frames):
        frame_path = os.path.join(frames_folder, f'frame_{i:04d}.png')
        frame = cv2.imread(frame_path)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize frame to match original image size if necessary
        if frame.shape != original_img.shape:
            frame = cv2.resize(frame, (original_img.shape[1], original_img.shape[0]))

        # Calculate PSNR
        psnr_value = psnr(original_img, frame)
        psnr_values.append(psnr_value)

        # Calculate SSIM
        ssim_value, _ = ssim(original_img, frame, multichannel=True, full=True)
        ssim_values.append(ssim_value)

        # Calculate LPIPS
        original_tensor = transforms.ToTensor()(Image.fromarray(original_img)).unsqueeze(0)
        frame_tensor = transforms.ToTensor()(Image.fromarray(frame)).unsqueeze(0)
        lpips_value = loss_fn(original_tensor, frame_tensor).item()
        lpips_values.append(lpips_value)

    # Calculate average metrics
    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    avg_lpips = np.mean(lpips_values)

    return avg_psnr, avg_ssim, avg_lpips

# Usage
original_image_path = 'example_rgba.png'
frames_folder = 'frames'
num_frames = 120  # Total number of frames (0 to 119), update if needed

avg_psnr, avg_ssim, avg_lpips = calculate_metrics(original_image_path, frames_folder, num_frames)
print(original_image_path)
print(f"Average PSNR: {avg_psnr:.2f}")
print(f"Average SSIM: {avg_ssim:.4f}")
print(f"Average LPIPS: {avg_lpips:.4f}")
