import numpy as np
import matplotlib.pyplot as plt
import torch
from config import MelConfig

def plot_mel_spectrogram(mel_spectrogram, mel_config = None, title="Mel Spectrogram", filename="mel_spectrogram.png"):
    """
    Plots a mel spectrogram.

    Args:
        mel_spectrogram: The mel spectrogram tensor (shape: [1, n_mels, T] where T is the time dimension).
        mel_config: Mel configuration used for normalization.
        title: Title of the plot.
    """

    if mel_config is None:
        mel_config = MelConfig()

    # Assuming mel_spectrogram is of shape [1, n_mels, T]
    mel_spectrogram = mel_spectrogram.squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU

    plt.figure(figsize=(12, 4))

    # Normalize if necessary (depends on your mel_config settings)
    #mel_spectrogram = mel_spectrogram ** 0.5  # Optional: square root for better visual representation
    #mel_spectrogram = np.clip(mel_spectrogram, 0, np.max(mel_spectrogram))

    # Display the mel spectrogram
    plt.imshow(mel_spectrogram, aspect='auto', origin='lower', 
               interpolation='none', extent=[0, mel_spectrogram.shape[1], 0, mel_config.n_mels])
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.ylabel('Mel Frequency')
    plt.xlabel('Time Frame')
    #plt.show()

    plt.savefig(filename)  # Save the plot as an image file
    plt.close()  # Close the figure to free memory

# # Example usage (call this function right after inference)
# audio_output, mel_output = inference(text, ref_audio, language, step, temperature, length_scale, solver, cfg)
# plot_mel_spectrogram(mel_output, mel_config)  # Pass the mel_config associated with your mel extractor