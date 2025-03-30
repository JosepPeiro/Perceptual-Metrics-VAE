import os
from preprocessing import Preprocessing, SplitAudio
import torchaudio

import torch
from torch.utils.data import DataLoader, TensorDataset

import requests
import io


def LoadAudios(route = "./MusicCaps", limit = None):

    archivos = os.listdir(route)

    X = []
    metadata = []
    sample_rate_red = 16000

    if limit is not None:
        archivos = archivos[:limit]

    for archivo in archivos:
        waveform, samp_rt = torchaudio.load(route + "/" + archivo)
        f, s = SplitAudio(waveform, sample_rate = samp_rt, new_sample_rate = sample_rate_red)

        f_spec, f_maxi, f_mini = Preprocessing(f, 16000, resampler_f = False)
        s_spec, s_maxi, s_mini = Preprocessing(s, 16000, resampler_f = False)

        X += [f_spec, s_spec]
        metadata += [{"nombre":archivo, "parte":"first", "minimum":f_mini, "maximum":f_maxi},
                    {"nombre":archivo, "parte":"second", "minimum":s_mini, "maximum":s_maxi}]
        
    return X, metadata


def Tensoring(X, batch_size = 64):
    X_tensor = torch.stack(X)  # Suponiendo que X es una lista de tensores (N, 256, 256)
    # Añadir la dimensión del canal (1 para monocanal)
    X_tensor = X_tensor.unsqueeze(1)  # (N, 1, 256, 256)

    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def LoadData(route = "./MusicCaps", limit = None, batch_size = 64):
    X, _ = LoadAudios(route = route, limit=limit)
    return Tensoring(X, batch_size = batch_size)


def LoadNoise(route = "random_noise.pt", limit = None, batch_size = 64):
    ruido = torch.load(route)
    if limit is not None:
        ruido = ruido[:limit*2]
    return Tensoring(ruido, batch_size = batch_size)


def LoadAudiosTest(route = "./PQMD", limit = None):

    archivos = os.listdir(route)

    X = []
    metadata = []

    if limit is not None:
        archivos = archivos[:limit]

    for archivo in archivos:
        waveform, samp_rt = torchaudio.load(route + "/" + archivo)
        spec, mx, mn = Preprocessing(waveform, samp_rt)

        X.append(spec)
        metadata.append({"name":archivo, "minimum":mn, "maximum":mx})
        
    return X, metadata


def LoadAudiosTestGithub(usuario = "JosepPeiro",
                         repositorio = "Perceptual-Metrics-VAE",
                         directorio = "PQMD",
                         branch = "main",
                         limit=None):

    url = f"https://api.github.com/repos/{usuario}/{repositorio}/contents/{directorio}?ref={branch}"
    response = requests.get(url)
    content = response.json()
    archivos = [archivo["name"] for archivo in content]

    X = []
    metadata = []

    if limit is not None:
        archivos = archivos[:limit]

    for archivo in archivos:
        url_arch = f"https://raw.githubusercontent.com/{usuario}/{repositorio}/{branch}/{directorio}/" + archivo
        respuesta = requests.get(url_arch)
        waveform, samp_rt = torchaudio.load(io.BytesIO(respuesta.content))
        spec, mx, mn = Preprocessing(waveform, samp_rt)

        X.append(spec)
        metadata.append({"name":archivo, "minimum":mn, "maximum":mx})

    return X, metadata