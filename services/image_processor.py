import cv2
import logging
import numpy as np
import torch

from config import (
    DEFAULT_GAMMA, DEFAULT_CURVE, DEFAULT_NOISE_AMOUNT, DEFAULT_BRIGHT_FACTOR,
    CLAHE_CLIP_LIMIT, CLAHE_TILE_GRID_SIZE, ENHANCEMENT_GAMMA, DENOISE_STRENGTH,
    ENHANCEMENT_LUMA_LOW, ENHANCEMENT_LUMA_VERY_LOW,
    ENHANCEMENT_CLAHE_CLIP_LIMIT_LOW, ENHANCEMENT_CLAHE_CLIP_LIMIT_VERY_LOW,
    ENHANCEMENT_GAMMA_LOW, ENHANCEMENT_GAMMA_VERY_LOW,
    ENHANCEMENT_DENOISE_LOW, ENHANCEMENT_DENOISE_VERY_LOW,
    ENHANCEMENT_GAIN_LOW, ENHANCEMENT_GAIN_VERY_LOW,
    ENHANCEMENT_BIAS_LOW, ENHANCEMENT_BIAS_VERY_LOW,
    ENHANCEMENT_DEEP_POSTBOOST_LUMA, ENHANCEMENT_DEEP_POSTBOOST_GAMMA,
    ENHANCEMENT_DEEP_POSTBOOST_GAIN, ENHANCEMENT_DEEP_POSTBOOST_BIAS
)
from models import model_manager


def darken_image(
    img_array,
    percent,
    noise_amount=DEFAULT_NOISE_AMOUNT,
    gamma=DEFAULT_GAMMA,
    curve=DEFAULT_CURVE,
    bright_factor=DEFAULT_BRIGHT_FACTOR
):
    arr = img_array.astype("float32") / 255.0
    linear = arr ** gamma
    linear_dark = linear * (1 - percent / 100.0)
    linear_dark = linear_dark ** curve
    linear_dark = linear_dark * (1 - bright_factor * linear)

    noise = np.random.normal(0.0, noise_amount, linear_dark.shape)
    linear_noisy = np.clip(linear_dark + noise, 0.0, 1.0)

    srgb = np.clip(linear_noisy ** (1.0 / gamma), 0, 1)
    out = (srgb * 255).astype("uint8")
    return out


def _mean_luminance_bgr(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, _, _ = cv2.split(lab)
    return float(l.mean())


def _select_enhancement_params(l_mean, base_gamma, base_denoise):
    clahe_clip = CLAHE_CLIP_LIMIT
    gamma = base_gamma
    denoise = base_denoise
    gain = 1.0
    bias = 0

    if l_mean <= ENHANCEMENT_LUMA_VERY_LOW:
        clahe_clip = ENHANCEMENT_CLAHE_CLIP_LIMIT_VERY_LOW
        gamma = max(base_gamma, ENHANCEMENT_GAMMA_VERY_LOW)
        denoise = max(base_denoise, ENHANCEMENT_DENOISE_VERY_LOW)
        gain = ENHANCEMENT_GAIN_VERY_LOW
        bias = ENHANCEMENT_BIAS_VERY_LOW
    elif l_mean <= ENHANCEMENT_LUMA_LOW:
        clahe_clip = ENHANCEMENT_CLAHE_CLIP_LIMIT_LOW
        gamma = max(base_gamma, ENHANCEMENT_GAMMA_LOW)
        denoise = max(base_denoise, ENHANCEMENT_DENOISE_LOW)
        gain = ENHANCEMENT_GAIN_LOW
        bias = ENHANCEMENT_BIAS_LOW

    return clahe_clip, gamma, denoise, gain, bias


def _apply_gamma_and_gain(img, gamma, gain, bias):
    gamma_inv = 1.0 / gamma
    table = np.array([(i / 255.0) ** gamma_inv * 255 for i in range(256)]).astype("uint8")
    enhanced = cv2.LUT(img, table)

    if gain != 1.0 or bias != 0:
        enhanced = cv2.convertScaleAbs(enhanced, alpha=gain, beta=bias)

    return enhanced




def enhance_classical(img, gamma=ENHANCEMENT_GAMMA, denoise_strength=DENOISE_STRENGTH):
    """
    Apply classical enhancement using CLAHE, gamma correction, and denoising.
    
    Args:
        img: Input image (BGR format)
        gamma: Gamma correction value
        denoise_strength: Strength of denoising filter
        
    Returns:
        Enhanced image
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_mean = float(l.mean())

    clahe_clip, gamma, denoise_strength, gain, bias = _select_enhancement_params(
        l_mean,
        gamma,
        denoise_strength,
    )

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=CLAHE_TILE_GRID_SIZE)
    l_clahe = clahe.apply(l)

    lab_clahe = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    enhanced_gamma = _apply_gamma_and_gain(enhanced, gamma, gain, bias)

    denoised = cv2.fastNlMeansDenoisingColored(
        enhanced_gamma,
        None,
        h=denoise_strength,
        hColor=denoise_strength,
        templateWindowSize=7,
        searchWindowSize=21,
    )
    return denoised


def enhance_deep(img):
    """
    Apply deep learning enhancement using Zero-DCE model.
    
    Args:
        img: Input image (BGR format)
        
    Returns:
        Enhanced image or None if model is unavailable
    """
    if not model_manager.is_zero_dce_available():
        return None

    try:
        img_norm = img.astype(np.float32) / 255.0
        img_input = (
            torch.from_numpy(np.transpose(img_norm, (2, 0, 1)))
            .unsqueeze(0)
            .to(model_manager.device)
        )

        with torch.no_grad():
            _, enhanced, _ = model_manager.deep_model(img_input)

        enhanced_img = enhanced.cpu().numpy()
        enhanced_img = np.clip(
            np.transpose(enhanced_img[0], (1, 2, 0)) * 255, 0, 255
        ).astype(np.uint8)

        if _mean_luminance_bgr(enhanced_img) <= ENHANCEMENT_DEEP_POSTBOOST_LUMA:
            enhanced_img = _apply_gamma_and_gain(
                enhanced_img,
                ENHANCEMENT_DEEP_POSTBOOST_GAMMA,
                ENHANCEMENT_DEEP_POSTBOOST_GAIN,
                ENHANCEMENT_DEEP_POSTBOOST_BIAS,
            )

        return enhanced_img
    except Exception as e:
        logging.error(f"Deep enhancement failed: {e}")
        return None

