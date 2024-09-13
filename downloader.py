import subprocess
import os
import pandas as pd
import argparse
from tqdm import tqdm
import socks
import socket
import json

def download_clip(ytid, start_seconds, end_seconds, output_path, video_frame_rate=5, audio_sample_rate=16000, proxy_url=None):
    """
    Downloads a clip from a YouTube video within a specified time range.

    Parameters:
    ytid (str): YouTube video ID.
    start_seconds (float): Start time of the clip in seconds.
    end_seconds (float): End time of the clip in seconds.
    output_path (str): Directory where the output video will be saved.
    video_frame_rate (int): Frame rate of the output video in frames per second.
    audio_sample_rate (int): Sample rate of the output audio in Hz.
    proxy_url (str): URL of the proxy to use for fetching the video.

    Returns:
    None
    """
    # Sanitize the YouTube ID to remove any unsafe characters for filenames
    # ytid_strip = ''.join(e for e in ytid if e.isalnum())
    # Convert start and end times to HH:MM:SS format
    start_time = f"{int(start_seconds // 3600):02d}:{int((start_seconds % 3600) // 60):02d}:{int(start_seconds % 60):02d}"
    duration = end_seconds - start_seconds

    # Construct the yt-dlp command to get the video and audio URLs
    command = ["yt-dlp","-i", "--youtube-skip-dash-manifest", "-g", f"https://youtu.be/{ytid}"]
    if proxy_url:
        command += ["--proxy", proxy_url]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if error:
        raise Exception(f"Error fetching URLs for {ytid}: {error.decode()}")

    # Decode the output and split by newlines to get separate URLs
    urls = output.decode("utf-8").strip().split("\n")
    if len(urls) != 2:
        # raise Exception(f"Error fetching URLs for {ytid}: {urls}")
        urls = [urls[0], urls[0]]
    
    # Construct the ffmpeg command to download the clip
    ffmpeg_command = [
        "ffmpeg",
        "-ss", start_time,
        "-i", urls[0],  # Video URL
        "-ss", start_time,
        "-i", urls[1],  # Audio URL
        "-ar", str(audio_sample_rate),  # Set audio sample rate
        "-ac", "1",  # Set audio channels to mono
        "-map", "0:v", "-map", "1:a",
        "-c:v", "libx264", "-c:a", "aac",
        "-t", str(duration),
        f"{output_path}/{ytid}.mp4"
    ]

    process = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if process.returncode != 0:
        raise Exception(f"Error downloading {ytid}: {process.stderr.decode()}")

def read_csv(csv_path, l_idx, u_idx):
    df = pd.read_csv(csv_path)
    df = df.iloc[l_idx:u_idx]
    return df



def download_clips(df, output_path, error_path, video_files, videos_list_file_path):
    # filter out rows that are already downloaded and in video_files
    print(f"Total rows: {len(df)}")
    df = df[~df['YTID'].isin(video_files)]
    print(f"Total rows after filtering: {len(df)}")
    print(f"Downloading {len(df)} clips to {output_path}")
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Downloading Clips"):
        try:
            download_clip(row['YTID'], row['start_time'], row['end_time'], output_path) # proxy_url="socks5://163.53.204.178:9813")
            print(f"Downloaded {row['YTID']}")
        except Exception as e:
            with open(error_path, 'a') as f:
                print(f"ERROR: {e}")
                f.write(f"ERROR: {e}\n")
        if i % 500 == 0:
            with open(videos_list_file_path, 'w') as f:
                json.dump(os.listdir(output_path), f)

def main(l_idx, u_idx,split):
    csv_path = f"/home/bld56/gsoc/general/download_logs/ytid_to_label_download_{split}_v3.csv"
    output_path = f"/tmp/bld56_dataset_v1/audioset/{split}/videos"
    error_path = f"/tmp/bld56_dataset_v1/audioset/{split}/download_logs/error_v3.log"
    videos_list_file_path = f"/home/bld56/gsoc/general/temps/gput.json"
    current_videos_list = json.load(open(videos_list_file_path, 'r'))
    current_videos_list = [f.replace('.mp4', '') for f in current_videos_list]
    print(current_videos_list[:10])
    # make dirs recursively if not exist
    subprocess.run(["mkdir", "-p", output_path])
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.dirname(error_path), exist_ok=True)
    df = read_csv(csv_path, l_idx, u_idx)
    download_clips(df, output_path, error_path, current_videos_list, videos_list_file_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download clips from YouTube videos.')
    parser.add_argument('--l_idx', type=int, help='lower index', required=True)
    parser.add_argument('--u_idx', type=int, help='upper index', required=True)
    parser.add_argument('--split', type=str, help='split', default='eval')
    args = parser.parse_args()
    # Set up the local SOCKS proxy
    # setup_socks_proxy()
    main(args.l_idx, args.u_idx, args.split)