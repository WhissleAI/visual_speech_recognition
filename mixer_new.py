"""
Author: Lakshmipathi Balaji
Desc: This code is used to mix two audio files with volume normalization to the mean RMS of both audios, resampling, and final volume adjustment.
"""
import numpy as np
import librosa
import soundfile as sf
import json
import os
import time
import logging
import random
import argparse

def setup_logging(mixed_output_base_dir):
    """Setup logging configuration."""
    time_and_date_string = time.strftime("%Y%m%d-%H%M%S")
    logs_base_dir = f'{mixed_output_base_dir}/mixing_logs/{time_and_date_string}'
    os.makedirs(logs_base_dir, exist_ok=True)
    logging.basicConfig(
        filename=f'{logs_base_dir}/error_log.log',
        level=logging.ERROR,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    return time_and_date_string, logs_base_dir

def log_error(message, function_name, sample_details=None):
    """Log an error with detailed information."""
    if sample_details:
        message = f"Function: {function_name} | Error: {message} | Sample Details: {sample_details}"
    else:
        message = f"Function: {function_name} | Error: {message}"
    logging.error(message)

def calculate_rms(audio):
    """Calculate the RMS (root mean square) level of an audio signal."""
    return np.sqrt(np.mean(audio**2))

def adjust_volume(audio, target_rms):
    """Adjust the audio's volume to a target RMS level."""
    current_rms = calculate_rms(audio)
    return audio * (target_rms / (current_rms + 1e-9))  # Avoid division by zero

def mix_audios(audio_path1, audio_path2, output_path, target_sr=16000, mix_ratio=0.5, final_volume_ratio=1.0):
    """
    NOTE: You have to Pass audio_path2 as ps audio.
    """
    # try:
    audio1, sr1 = librosa.load(audio_path1, sr=target_sr)
    audio2, sr2 = librosa.load(audio_path2, sr=target_sr)
    
    rms1 = calculate_rms(audio1)
    rms2 = calculate_rms(audio2)
    mean_rms = np.mean([rms1, rms2])
    
    audio1 = adjust_volume(audio1, mean_rms)
    audio2 = adjust_volume(audio2, mean_rms)
    
    # min_len = min(len(audio1), len(audio2))
    assert len(audio2) > 10*target_sr, f"PS Audio file {audio_path2} is less than 10 seconds"
    if len(audio1) < 10*target_sr:
        audio1 = np.pad(audio1, (0, 10*target_sr - len(audio1)), 'constant')
    min_len = min(160000, len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    mixed_audio = mix_ratio * audio1 + (1 - mix_ratio) * audio2
    
    mixed_audio_rms = calculate_rms(mixed_audio)
    mixed_audio = adjust_volume(mixed_audio, mixed_audio_rms * final_volume_ratio)
    
    # make sure all the folders till the parent folder of output_path exists
    os.makedirs(os.path.join(output_path).rsplit('/', 1)[0], exist_ok=True)
    sf.write(output_path, mixed_audio, target_sr)
    # except Exception as e:
    #     log_error(str(e), "mix_audios", {"audio_path1": audio_path1, "audio_path2": audio_path2, "output_path": output_path})
    #     raise e

def read_json(json_file, line_wise=False):
    try:
        if line_wise:
            data = []
            with open(json_file) as f:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
            return data
        else:
            with open(json_file) as f:
                data = json.load(f)
            return data
    except Exception as e:
        log_error(str(e), "read_json", {"json_file": json_file, "line_wise": line_wise})
        return []

def main(mixing_ratio, numsamples):
    
    # Normalize mixing_ratio to a two-digit format for consistency in directory naming
    mixing_ratio_str = f"{mixing_ratio:.2f}".replace('.', '')
    mixed_output_base_dir = f'/tmp/bld56_dataset_v1/get_wer/mixed_dataset_{mixing_ratio_str}/'
    time_and_date_string, logs_base_dir = setup_logging(mixed_output_base_dir)

    config = {
        'base_dir_as_videos': '/tmp/bld56_dataset_v1/audioset/train/videos',
        'base_dir_ps': '/tmp/bld56_dataset_v1/peoples_speech/',
        'base_dir_as': '/tmp/bld56_dataset_v1/audioset/',
        'ctm_file_base_dir': '/workspace/dataset/validation/',
        'as_samples_json': 'annotations/it1_audioset_anns_final_train.json',
        'mixed_output_base_dir': mixed_output_base_dir,
        'mixed_output_audios_dir': 'mixed_audios/validation_ps/',
        'mixed_ouput_anns_file': 'annotations/mixed_audios_anns.json',
        'ps_ctm_file_manifest': 'manifest_with_output_file_paths.json',
        'mixing_ratio': mixing_ratio,
        'output_mapping_json': 'annotations/mixing_map',
        'logs_base_dir': logs_base_dir
    }
    config['output_mapping_json'] = config['output_mapping_json'] + time_and_date_string + '.json'
    os.makedirs(os.path.join(config['mixed_output_base_dir'], config['output_mapping_json']).rsplit('/', 1)[0], exist_ok=True)
    os.makedirs(os.path.join(config['mixed_output_base_dir'], config['mixed_output_audios_dir']), exist_ok=True)
    ps_manifest = read_json(os.path.join(config['base_dir_ps'], config['ps_ctm_file_manifest']), line_wise=True)
    as_samples = read_json(os.path.join(config['base_dir_as'], config['as_samples_json']), line_wise=False)
    random.seed(42)
    random.shuffle(as_samples)
    unique_video_ids = list(set([sample['YTID'] for sample in as_samples]))
    assert len(unique_video_ids) > numsamples, f"Number of unique video ids is less than the number of samples required. Got {len(unique_video_ids)} unique video ids."
    added_ids = set()
    as_samples_unique = []
    for sample in as_samples:
        video_id = sample['YTID']
        if video_id not in added_ids:
            as_samples_unique.append(sample)
            added_ids.add(video_id)

    print(f"Got {len(as_samples_unique)} unique samples")
    assert len(unique_video_ids) > numsamples, f"Number of unique video ids is less than the number of samples required. Got {len(unique_video_ids)} unique video ids."
    random.shuffle(ps_manifest)
    output_mapping = []

    for i in range(len(unique_video_ids)):
        if len(output_mapping) == numsamples:
            break
        as_sample = as_samples_unique[i]
        video_id = as_sample['YTID']
        # Keep only alnum chars in video_id
        video_id = ''.join(e for e in video_id if e.isalnum())
        video_path = os.path.join(config['base_dir_as_videos'], video_id + '.mp4')

        try:
            if not os.path.exists(video_path):
                with open(os.path.join(config['logs_base_dir'], 'missing_as_videos.txt'), 'a') as f:
                    f.write(video_id + '\n')
                continue

            ps_sample = ps_manifest[i]
            ps_audio_path = ps_sample['audio_filepath']
            # get length of audio file
            ps_audio_path_length = librosa.get_duration(filename=ps_audio_path)
            if ps_audio_path_length < 10:
                with open(os.path.join(config['logs_base_dir'], 'short_ps_audios.txt'), 'a') as f:
                    f.write(ps_audio_path + '\n')
                continue
            if not os.path.exists(ps_audio_path):
                with open(os.path.join(config['logs_base_dir'], 'missing_ps_audios.txt'), 'a') as f:
                    f.write(ps_audio_path + '\n')
                continue

            last_two_subdirs = ps_audio_path.split('/')[-2:]
            ps_output_substring = last_two_subdirs[0][:5] + '_' + last_two_subdirs[1]
            output_path_base_name = video_id + '_' + ps_output_substring.split('.')[0]

            mixed_output_path = os.path.join(config['mixed_output_base_dir'], config['mixed_output_audios_dir'], output_path_base_name + '.wav')

            mix_audios(video_path, ps_audio_path, mixed_output_path, mix_ratio=config['mixing_ratio'])

            gts = {
                'video_id': video_id,
                'video_path': video_path,
                'ps_audio_path': ps_audio_path,
                'mixed_output_path': mixed_output_path,
                'text': ps_sample['text'],
                'noise_label': as_sample['label'],
                'noise_start_time': as_sample['start_time'],
                'noise_end_time': as_sample['end_time'],
                'ps_ctm_file': config['ctm_file_base_dir'] +ps_sample['words_level_ctm_filepath']
            }
            output_mapping.append(gts)
        except Exception as e:
            log_error(str(e), "main_loop", {"video_id": video_id, "video_path": video_path, "ps_audio_path": ps_audio_path})
    try:
        with open(os.path.join(config['mixed_output_base_dir'], config['output_mapping_json']), 'w') as f:
            json.dump(output_mapping, f)
    except Exception as e:
        log_error(str(e), "writing_output_mapping", {"output_mapping_file": config['output_mapping_json']})
    print(f"Got {len(output_mapping)} samples")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audio Mixing Script')
    parser.add_argument('--mixing_ratio', type=float, required=False, help='Mixing ratio for audio files', default=0.0)
    parser.add_argument('--num_samples', type=int, required=False, help='Number of samples to process', default=1000)
    args = parser.parse_args()
    main(args.mixing_ratio, args.num_samples)
