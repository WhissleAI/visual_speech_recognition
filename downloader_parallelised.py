import subprocess
import os
import pandas as pd
import argparse
from tqdm import tqdm
import socks
import socket
import json
from multiprocessing import Pool, cpu_count

def download_clip(ytid, start_seconds, end_seconds, output_path, video_frame_rate=5, audio_sample_rate=16000, proxy_url=None):
    """
    Downloads a clip from a YouTube video within a specified time range.
    """
    start_time = f"{int(start_seconds // 3600):02d}:{int((start_seconds % 3600) // 60):02d}:{int(start_seconds % 60):02d}"
    duration = end_seconds - start_seconds

    command = ["yt-dlp", "-i", "--youtube-skip-dash-manifest", "-g", f"https://youtu.be/{ytid}"]
    if proxy_url:
        command += ["--proxy", proxy_url]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = process.communicate()

    if error:
        raise Exception(f"Error fetching URLs for {ytid}: {error.decode()}")

    urls = output.decode("utf-8").strip().split("\n")
    if len(urls) != 2:
        urls = [urls[0], urls[0]]

    ffmpeg_command = [
        "ffmpeg",
        "-ss", start_time,
        "-i", urls[0],
        "-ss", start_time,
        "-i", urls[1],
        "-ar", str(audio_sample_rate),
        "-ac", "1",
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

def setup_socks_proxy(proxy_host='localhost', proxy_port=1080):
    socks.set_default_proxy(socks.SOCKS5, proxy_host, proxy_port)
    socket.socket = socks.socksocket

def download_clip_wrapper(args):
    try:
        fun_args = args[:-1]
        download_clip(*fun_args)
    except Exception as e:
        error_path = args[-1]
        with open(error_path, 'a') as f:
            print(f"ERROR: {e}")
            f.write(f"ERROR: {e}\n")

def download_clips(df, output_path, error_path, video_files, videos_list_file_path, num_workers):
    df = df[~df['YTID'].isin(video_files)]
    print(f"Downloading {len(df)} clips to {output_path} using {num_workers} workers")
    tasks = [(row['YTID'], row['start_time'], row['end_time'], output_path, 5, 16000, None, error_path) for _, row in df.iterrows()]
    
    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(download_clip_wrapper, tasks), total=len(tasks), desc="Downloading Clips"))

    with open(videos_list_file_path, 'w') as f:
        json.dump(os.listdir(output_path), f)

def main(l_idx, u_idx, split, num_workers):
    csv_path = f"/home/bld56/gsoc/general/download_logs/ytid_to_label_download_{split}_v3.csv"
    output_path = f"/tmp/bld56_dataset_v1/audioset/{split}/videos_1"
    error_path = f"/tmp/bld56_dataset_v1/audioset/{split}/download_logs/error_v3.log"
    videos_list_file_path = f"/home/bld56/gsoc/general/temps/gput.json"
    current_videos_list = json.load(open(videos_list_file_path, 'r'))
    current_videos_list = [f.replace('.mp4', '') for f in current_videos_list]

    subprocess.run(["mkdir", "-p", output_path])
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(os.path.dirname(error_path), exist_ok=True)
    df = read_csv(csv_path, l_idx, u_idx)
    download_clips(df, output_path, error_path, current_videos_list, videos_list_file_path, num_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download clips from YouTube videos.')
    parser.add_argument('--l_idx', type=int, help='lower index', required=True)
    parser.add_argument('--u_idx', type=int, help='upper index', required=True)
    parser.add_argument('--split', type=str, help='split', default='eval')
    parser.add_argument('--workers', type=int, help='number of parallel workers', default=cpu_count())
    args = parser.parse_args()
    main(args.l_idx, args.u_idx, args.split, args.workers)
