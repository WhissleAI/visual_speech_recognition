"""
Author: Lakshmipathi Balaji
Desc: This code is used to mix two audio files with volume normalization to the mean RMS of both audios, resampling, and final volume adjustment.
"""
import numpy as np
import librosa
import soundfile as sf

def calculate_rms(audio):
    """Calculate the RMS (root mean square) level of an audio signal."""
    return np.sqrt(np.mean(audio**2))

def adjust_volume(audio, target_rms):
    """Adjust the audio's volume to a target RMS level."""
    current_rms = calculate_rms(audio)
    return audio * (target_rms / (current_rms + 1e-9))  # Avoid division by zero

def mix_audios(audio_path1, audio_path2, output_path, target_sr=22050, mix_ratio=0.5, final_volume_ratio=1.0):
    """
    Mix two audios with volume normalization to the mean RMS of both audios, resampling,
    and final volume adjustment.
    
    audio_path1: Path to the first audio file.
    audio_path2: Path to the second audio file.
    output_path: Path where the mixed audio will be saved.
    target_sr: Target sampling rate for both audios.
    mix_ratio: The mix ratio for the first audio. The second audio's ratio will be 1 - mix_ratio.
    final_volume_ratio: Final volume adjustment ratio; 1.0 means no change, <1.0 decreases, >1.0 increases.
    """
    audio1, sr1 = librosa.load(audio_path1, sr=target_sr)
    audio2, sr2 = librosa.load(audio_path2, sr=target_sr)
    
    rms1 = calculate_rms(audio1)
    rms2 = calculate_rms(audio2)
    mean_rms = np.mean([rms1, rms2])
    
    audio1 = adjust_volume(audio1, mean_rms)
    audio2 = adjust_volume(audio2, mean_rms)
    
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    mixed_audio = mix_ratio * audio1 + (1 - mix_ratio) * audio2
    
    mixed_audio_rms = calculate_rms(mixed_audio)
    mixed_audio = adjust_volume(mixed_audio, mixed_audio_rms * final_volume_ratio)
    
    sf.write(output_path, mixed_audio, target_sr)

# Example usage
mix_audios('audioset_samples/plane.wav', 'peoples_speech/1.wav', 'mixed_audios/plane_1.wav')
