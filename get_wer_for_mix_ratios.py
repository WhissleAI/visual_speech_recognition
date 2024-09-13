from torchmetrics.text import WordErrorRate
import nemo.collections.asr as nemo_asr
import json
import subprocess
import os
from read_ctm import read_file

# Local Imports
from mixer_new import main, read_json, mix_audios, log_error, calculate_rms, adjust_volume, setup_logging

def load_model(model_name):
    model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    return model

def load_json_lines(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))
    return data

def get_wer(model, annotation_file):
    wer = WordErrorRate()
    # with open(annotation_file, "r") as f:
    #     file_data = json.load(f)
    file_data = load_json_lines(annotation_file)
    mixed_audio_paths = [line_data['audio_file'] for line_data in file_data]
    transcripts = model.transcribe(mixed_audio_paths)
    preds = transcripts
    targets = [line_data['text'] for line_data in file_data]
    # targets = []
    # for line_data in file_data:
    #     ctm_data = read_file(line_data['ps_ctm_file'])
    #     # ['be', 8.32, 0.08],
    #     #       ['paying', 8.4, 0.32],
    #     #       ['in', 8.8, 0.08],
    #     #       ['two', 8.96, 0.08],
    #     # choose only words that are less than 10 seconds in index 1
    #     # and append them to targets
    #     target = [word[0] for word in ctm_data if float(word[1]) < 10]
    #     targets.append(" ".join(target))
    print(f"Preds: {preds}")
    word_error_rate = wer(preds, targets)
    return word_error_rate.tolist()

    
def merge_and_get_wer(model, ratio, num_samples):
    # run command for mixer.py with given ratio
    # pass model, annotation file for that ratio to get_wer()
    # return wer
    # subprocess.run(["python", "/workspace/gsoc/mixer_new.py", f"--mix_ratio={ratio}", f"--num_samples={num_samples}"])
    # main(ratio, num_samples)
    mixing_ratio_str = f"{ratio:.2f}".replace('.', '')
    ann_file = os.listdir(f"/tmp/bld56_dataset_v1/get_wer/mixed_dataset_{mixing_ratio_str}/annotations/")[0]
    wer = get_wer(model, f"/tmp/bld56_dataset_v1/get_wer/mixed_dataset_{mixing_ratio_str}/annotations/{ann_file}")
    return wer
    
    

# if __name__ == "__main__":
#     model = load_model("stt_en_conformer_ctc_large")
#     # model = load_model("nvidia/canary-1b")
#     # ratios = [0.0, 0.05 , 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
#     # ratios = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
#     ratios = [0.0]
#     num_samples = 1000
#     ratio_to_wer = {}
#     output_file = "wer_for_mix_ratios_fine.json"
#     # Check if the output file exists and read the existing data
#     # if os.path.exists(output_file):
#     #     with open(output_file, 'r') as f:
#     #         ratio_to_wer = json.load(f)
#     for ratio in ratios:
#         wer = merge_and_get_wer(model, ratio, num_samples)
#         ratio_to_wer[ratio] = wer
#         print(f"WER for ratio {ratio}: {wer}")
#         with open(output_file, 'w') as f:
#             json.dump(ratio_to_wer, f)

get_wer(load_model("stt_en_conformer_ctc_large"), "/tmp/bld56_dataset_v1/it1_20/annotations/manifest_eval.json")
        
