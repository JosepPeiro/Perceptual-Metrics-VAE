import torch
import torchaudio
from torchaudio.transforms import Resample, MelSpectrogram, InverseMelScale, GriffinLim
import os
from IPython.display import Audio


def log_scale_spectrogram(spectrogram, epsilon=0.001):
    # Aplicar logaritmo con epsilon
    log_spec = torch.log(spectrogram + epsilon)

    # Normalizar al rango [0,1]
    log_spec_min = log_spec.min()
    log_spec_max = log_spec.max()
    log_spec_norm = (log_spec - log_spec_min) / (log_spec_max - log_spec_min)
    
    return log_spec_norm, log_spec_max, log_spec_min


def Reconstruction(log_spectr, maxi, mini,
                   sample_rate = 16000,
                   n_mels = 256, n_fft=1024,
                   hop_length=260, epsilon=0.001,
                   num_iters=500):
    
    # Desnormalizar
    log_spec = log_spectr * (maxi - mini) + mini

    # Deslogarizar
    mel = torch.exp(log_spec) - epsilon

    # Desmelizar
    mel_to_stft = InverseMelScale(n_stft = n_fft // 2 + 1,
                                  n_mels=n_mels,
                                  sample_rate=sample_rate,
                                  driver='gelsd')
    spectrogram = mel_to_stft(mel)

    # Desespectogramizar
    griffin_lim = GriffinLim(n_fft=n_fft, hop_length=hop_length, power=1.0, n_iter=num_iters)
    waveform = griffin_lim(spectrogram)

    return waveform


def ResamplerAudio(waveform, old_sample, new_sample):
    resampler = Resample(orig_freq=old_sample, new_freq=new_sample)
    return resampler(waveform)


def Mono(waveform):
    return waveform.mean(axis=0)


def MelTransform(waveform,
                 sample_rate = 1600,
                 window_size = 1024,
                 hop_size = 260,
                 n_mels = 256,
                 power = 1,
                 freq_min=0,
                 freq_max = 1600 // 2,
                 center = False,
                 window_fn = torch.hann_window,
                 mel_scale = 'htk',
                 norm = None):
    mel_transform = MelSpectrogram(
        sample_rate=sample_rate,   # Tasa de muestreo del audio
        n_fft=window_size,                # Tamaño de la ventana FFT
        win_length=window_size,
        hop_length=hop_size,            # Hop length (desplazamiento entre ventanas)
        n_mels=n_mels,                # Número de bandas Mel
        power=power,                  # Magnitud (espectrograma de potencia)
        f_min=freq_min,
        f_max = freq_max,
        center = center,
        window_fn = window_fn,
        mel_scale=mel_scale,
        norm=norm
    )
    return mel_transform(waveform)


def Preprocessing(waveform, sample_rate,
                  mono_f = True, resampler_f = True, mel_f = True, log_norm_f = True,
                  new_sample_rate = 1600, window_size = 1024, hop_size = 260, n_mels = 256,
                  power = 1, freq_min = 0, freq_max = 1600 // 2,center = False,
                  window_fn = torch.hann_window, mel_scale = 'htk', norm = None,
                  epsilon=0.001):
    
    if mono_f:
        waveform = Mono(waveform)

    if resampler_f:
        waveform = ResamplerAudio(waveform, sample_rate, new_sample_rate)

    if mel_f:
        waveform = MelTransform(waveform,
                    sample_rate = new_sample_rate,
                    window_size = window_size,
                    hop_size = hop_size,
                    n_mels = n_mels,
                    power = power,
                    freq_min = freq_min,
                    freq_max = freq_max,
                    center = center,
                    window_fn = window_fn,
                    mel_scale = mel_scale,
                    norm = norm)
        
    maxi, mini = 0, 0
    if log_norm_f:
        waveform, maxi, mini = log_scale_spectrogram(waveform, epsilon = epsilon)

    return waveform, maxi, mini