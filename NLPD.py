import numpy as np
import cv2
from scipy.ndimage import convolve

def NLP_dist(IM_1, IM_2, DN_filts=None):
    """Calcula la métrica de distorsión perceptual entre dos imágenes."""
    if DN_filts is None:
        DN_filts = DN_filters()

    N_levels = 6

    Y_ori, Lap_ori = NLP(IM_1, DN_filts)
    Y_dist, Lap_dist = NLP(IM_2, DN_filts)

    RR_Lap_aux = []
    RR_aux = []

    for N_b in range(N_levels):
        RR_Lap_aux.append(np.sqrt(np.mean((Lap_ori[N_b] - Lap_dist[N_b]) ** 2)))
        RR_aux.append(np.sqrt(np.mean((Y_ori[N_b] - Y_dist[N_b]) ** 2)))

    DMOS_Lap = np.mean(RR_Lap_aux)
    DMOS_Lap_dn2 = np.mean(RR_aux)

    return DMOS_Lap_dn2, DMOS_Lap

def NLP(IM, DN_filts):
    """Aplica la transformación Laplaciana y normalización divisiva a la imagen."""
    N_levels = 6
    Lap_dom = laplacian_pyramid_s(IM, N_levels)
    DN_dom = []

    for N_b in range(N_levels):
        A2 = convolve(np.abs(Lap_dom[N_b]), DN_filts[N_b]["F2"], mode='constant')
        DN_dom.append(Lap_dom[N_b] / (DN_filts[N_b]["sigma"] + A2))

    return DN_dom, Lap_dom

def DN_filters():
    """Define los filtros de normalización divisiva."""
    sigmas = [0.0248, 0.0185, 0.0179, 0.0191, 0.0220, 0.2782]
    filters = [{} for _ in range(6)]

    F2_matrices = [
        np.array([
            [0, 0, 0, 0, 0],
            [0, 0, 0.1011, 0, 0],
            [0, 0.1493, 0, 0.1460, 0.0072],
            [0, 0, 0.1015, 0, 0],
            [0, 0, 0, 0, 0]
        ]),
        np.pad(np.array([
            [0, 0.0757, 0],
            [0.1986, 0, 0.1846],
            [0, 0.0837, 0]
        ]), ((1, 1), (1, 1)), 'constant'),
        np.pad(np.array([
            [0, 0.0477, 0],
            [0.2138, 0, 0.2243],
            [0, 0.0467, 0]
        ]), ((1, 1), (1, 1)), 'constant'),
        np.pad(np.array([
            [0, 0, 0],
            [0.2503, 0, 0.2616],
            [0, 0, 0]
        ]), ((1, 1), (1, 1)), 'constant'),
        np.pad(np.array([
            [0, 0, 0],
            [0.2598, 0, 0.2552],
            [0, 0, 0]
        ]), ((1, 1), (1, 1)), 'constant'),
        np.pad(np.array([
            [0, 0, 0],
            [0.2215, 0, 0.0717],
            [0, 0, 0]
        ]), ((1, 1), (1, 1)), 'constant')
    ]

    for i in range(6):
        filters[i]["sigma"] = sigmas[i]
        filters[i]["F2"] = F2_matrices[i]

    return filters

def laplacian_pyramid_s(I, nlev=None):
    r, c = I.shape[:2]
    
    if nlev is None:
        # Compute the highest possible pyramid level
        nlev = int(np.floor(np.log2(min(r, c))))
    
    # Initialize pyramid
    pyr = []
    f = np.array([0.05, 0.25, 0.4, 0.25, 0.05])  # Burt and Adelson, 1983
    filter_kernel = np.outer(f, f)
    
    J = I.copy()
    for l in range(nlev - 1):
        # Apply low-pass filter and downsample
        I_low = downsample(J, filter_kernel)
        odd = (np.array(I_low.shape[:2]) * 2 - np.array(J.shape[:2]))
        # Store the difference between the image and upsampled low-pass version
        pyr.append(J - upsample(I_low, odd, filter_kernel))

        J = I_low.copy()  # Continue with low-pass image
    
    pyr.append(J)  # The coarsest level contains the residual low-pass image
    return pyr

def downsample(I, filter_kernel):
    # Apply Gaussian filter
    I_blur = cv2.filter2D(I, -1, filter_kernel, borderType=cv2.BORDER_REFLECT)
    
    # Downsample by taking every second pixel
    return I_blur[::2, ::2]


def upsample(I, odd, filter_kernel):
    # Increase resolution
    I = np.pad(I, ((1, 1), (1, 1)), mode='edge')  # Pad the image with a 1-pixel border
    r, c = I.shape[:2]
    I_up = np.zeros((r * 2, c * 2), dtype=I.dtype)
    I_up[::2, ::2] = I * 4  # Scale to compensate for inserted zeros
    # Apply Gaussian filter for interpolation
    I_up = cv2.filter2D(I_up, -1, filter_kernel, borderType=cv2.BORDER_REFLECT)
    # Remove the border
    return I_up[2:I_up.shape[0] - 2 - odd[0], 2:I_up.shape[1] - 2 - odd[1]]
