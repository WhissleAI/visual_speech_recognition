import json
from read_ctm import read_file
import random
import pandas as pd
import numpy as np
import librosa
import os
from mixer_new import mix_audios


def add_label_to_text(label, endlabel, st_time, end_time, ctm_file):
    """
    Reads CTM file to get text below 10 secs and get time stamps to place labels accordingly.

    Args:
        text : Transcript.
        label : Label to be added to the text.
        st_time : Start time of the label.
        end_time : End time of the label.
        ctm_file : CTM file to get time stamps for the label and text.

    Returns:
        text : Text with label.
    """
    ctm_data = read_file(ctm_file)
    # ['be', 8.32, 0.08],
    #       ['paying', 8.4, 0.32],
    #       ['in', 8.8, 0.08],
    #       ['two', 8.96, 0.08],
    # choose only words that are less than 10 seconds in index 1
    # and append them to targets
    till_10_sec = [word for word in ctm_data if float(word[1]) < 10]
    # compare the start time and end time with the words in the till_10_sec
    # and add the label to the word.
    final_text = ""
    inside_label = False
    for word in till_10_sec:
        word_text = word[0]
        word_start_time = float(word[1])
        word_end_time = word_start_time + float(word[2])
        
        # Check if the word falls within the label's time range
        if word_start_time >= st_time and word_end_time <= end_time:
            if not inside_label:
                final_text += f"{label} "
                inside_label = True
            final_text += f"{word_text} "
        else:
            if inside_label:
                final_text += f"{endlabel} "
                inside_label = False
            final_text += f"{word_text} "
    
    # If the last word was within the label range, close the label
    if inside_label:
        final_text += f"{endlabel} "
    
    return final_text.strip()


def map_current_anns_to_labels_in_audioset(unique_labels, labels_file):
    """
    Converts all labels to a good format, all unique labels should have mapping 
    starting from N1, N2, N3, ... etc.

    Args:
        unique_labels : List of all unique labels.  

    Returns:
        label_mapping : Mapping of all labels to N1, N2, N3, ... etc.
    """
    label_mapping = {}
    for i, label in enumerate(unique_labels):
        label_mapping[label] = f"<N{i+2}>"
    label_mapping["endlabel"] = "<N1>"
    if not os.path.exists(labels_file):
        labels_list = []
    else:
        labels_list = json.load(open(labels_file, 'r'))
    if isinstance(labels_list, list):
        # assert if all labels are present in the labels file.
        for label in unique_labels:
            if label != "endlabel":
                assert label in labels_list, f"{label} not present in the labels file."
        mid_to_label = pd.read_csv('/disk1/audioset/annotations/mid_to_display_name.tsv', sep='\t', names=['mid', 'label'])
        mid_to_coded_label_and_display_name = {}
        for i, row in mid_to_label.iterrows():
            if row['mid'] in label_mapping.keys():
                mid_to_coded_label_and_display_name[row['mid']] = [row['label'], label_mapping[row['mid']] if row['mid'] in label_mapping else None]
        mid_to_coded_label_and_display_name["endlabel"] = ["endlabel", "<N1>"]
        json.dump(mid_to_coded_label_and_display_name, open(labels_file, 'w'))
    elif isinstance(labels_list, dict):
        assert len(labels_list) == len(unique_labels) + 1, f"Labels file should have same number of labels as unique labels. But previous trainset has {len(labels_list)} but current set has {len(unique_labels)}. {set(unique_labels) - set(labels_list.keys()) }extra labels in current set. {set(labels_list.keys()) - set(unique_labels)} extra labels in previous set."

    return label_mapping


def read_video_anns(ann_file, labels_file):
    """
    Reads video annotations file and replaces labels with the mapped labels.
    Repaets a randoom video if a video from label is not present.

    Args:
        ann_file : Annotations file.

    Returns:
        anns : List of all video annotations. 
        [YTID, start_time, end_time, label]
    """
    video_anns = json.load(open(ann_file, 'r'))
    # TODO: @Balu - Check processor .ipynb this is tmp fix.
    # remove the samples with '/m/07qfr4h' as label
    video_anns = [x for x in video_anns if x["label"] != "/m/07qfr4h"]

    unique_labels = list(set([x["label"] for x in video_anns]))
    label_mapping = map_current_anns_to_labels_in_audioset(unique_labels, labels_file)
    for ann in video_anns:
        ann["label"] = label_mapping[ann["label"]]

    label_to_videos = {}
    for ann in video_anns:
        if ann["label"] not in label_to_videos:
            label_to_videos[ann["label"]] = []
        # only append if the video is present.
        if ann["is_present"]:
            label_to_videos[ann["label"]].append(
                {
                    "YTID": ann["YTID"],
                    "start_time": ann["start_time"],
                    "end_time": ann["end_time"],
                    "label": ann["label"]
                })

    video_anns_req = []
    for ann in video_anns:
        if ann["is_present"]:
            video_anns_req.append(
                # ann["YTID"], ann["start_time"], ann["end_time"], ann["label"]
                {
                    "YTID": ann["YTID"],
                    "start_time": ann["start_time"],
                    "end_time": ann["end_time"],
                    "label": ann["label"]
                }
            )
        else:
            # pick the video with the same label any random video.
            random_ann = random.choice(label_to_videos[ann["label"]])
            video_anns_req.append(
                # random_ann["YTID"], random_ann["start_time"], random_ann["end_time"], ann["label"]
                {
                    "YTID": random_ann["YTID"],
                    "start_time": random_ann["start_time"],
                    "end_time": random_ann["end_time"],
                    "label": ann["label"]
                }
            )

    return video_anns_req


def load_ctm_anns(ctm_ann_file):
    """
    Reads CTM annotations file and replaces labels with the mapped labels.

    Args:
        ctm_ann_file : CTM Annotations file.

    Returns:
        ctm_anns : List of all CTM annotations. 
        [YTID, start_time, end_time, label]
    """
    def load_json_lines(file_path):
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line.strip()))
        return data

    # {"audio_filepath": "/disk1/peoples_speech/downloads/extracted/496fee4d8a940f49701c485feccb9cb4adc7bfeeb089790a3121a6130d9583ce/1_16_2018_Williston_Selectboard_SLASH_1_16_2018_Williston_Selectboard_DOT_mp3_00010.flac", "text": "type of thing and certainly brian did a good job explaining sort of where he's coming from i guess my observation is we could have somebody also in the same position who might advocate from a totally different position might", "tokens_level_ctm_filepath": "aligned_peoples_speech/ctm/tokens/1_16_2018_Williston_Selectboard_SLASH_1_16_2018_Williston_Selectboard_DOT_mp3_00010.ctm", "words_level_ctm_filepath": "aligned_peoples_speech/ctm/words/1_16_2018_Williston_Selectboard_SLASH_1_16_2018_Williston_Selectboard_DOT_mp3_00010.ctm", "segments_level_ctm_filepath": "aligned_peoples_speech/ctm/segments/1_16_2018_Williston_Selectboard_SLASH_1_16_2018_Williston_Selectboard_DOT_mp3_00010.ctm"}
    ps_anns = []
    
    file_data = load_json_lines(ctm_ann_file)

    for ann in file_data:
        if "words_level_ctm_filepath" not in ann.keys():
            continue
        ps_anns.append(
            {
                "audio_filepath": ann["audio_filepath"],
                "text": ann["text"],
                "words_level_ctm_filepath": ann["words_level_ctm_filepath"],
            }
        )
    return ps_anns

def write_json_lines(data, file_path):
    with open(file_path, 'w') as file:
        for line in data:
            file.write(json.dumps(line) + '\n')

# def main():
"""
Main function to read all the files and generate the final manifest file.
"""
split_type = 'eval'
config_eval = {
    "split_type": ['eval'],  # ['train', 'eval']
    "ctm_ann_file": "/disk1/peoples_speech/train_clean/aligned_peoples_speech/manifest_with_output_file_paths_eval.json",
    "videos_ann_file": "/disk1/v1/audioset/audioset_anns_final_eval_present.json",
    "videos_folder": "/disk1/audioset/eval/videos/",
    "mixed_audio_folder": "/disk1/it1_20/mixed_audios_eval/",
    "ctm_file_prefix": "/disk1/peoples_speech/train_clean/",
    "labels_list_file": "/disk1/audioset/annotations/it1_audioset_labels.json",
    "output_file": "/disk1/it1_20/annotations/manifest_eval.json",
    "as_info_file": "/disk1/audioset/eval/info.json",
    "video_feats_base_folder": "/disk1/audioset/eval/feats_ViT-B-32_5fps/",
    "as_video_issues_file": "/disk1/audioset/annotations/it1_20_as_issues_eval.json",
    "seed": 42
}
confg_train = {
    "split_type": ['train'],  # ['train', 'eval']
    "ctm_ann_file": "/disk1/peoples_speech/train_clean/aligned_peoples_speech/manifest_with_output_file_paths.json",
    "videos_ann_file": "/disk1/v1/audioset/audioset_anns_final_train_present.json",
    "videos_folder": "/disk1/audioset/train/videos/",
    "mixed_audio_folder": "/disk1/it1_20/mixed_audios_train/",
    "ctm_file_prefix": "/disk1/peoples_speech/train_clean/",
    "labels_list_file": "/disk1/audioset/annotations/it1_audioset_labels.json",
    "output_file": "/disk1/it1_20/annotations/manifest_train.json",
    "as_info_file": "/disk1/audioset/train/info.json",
    "video_feats_base_folder": "/disk1/audioset/train/feats_ViT-B-32_5fps/",
    "as_video_issues_file": "/disk1/audioset/annotations/it1_20_as_issues_train.json",
    "seed": 42
}
if split_type == 'train':
    config = confg_train
elif split_type == 'eval':
    config = config_eval
else:
    raise ValueError(f"Invalid split type. Should be train or eval. Got {split_type}")

video_anns = read_video_anns(config["videos_ann_file"], config["labels_list_file"])
ps_anns = load_ctm_anns(config["ctm_ann_file"])
end_label = "<N1>"
random.seed(config["seed"])
final_manifest = []
# snr_ratios = np.random.uniform(0.2, 0.5, len(video_anns))
snr_ratios = [0.2] * len(video_anns)
ps_ann_idx = 0
as_issue_videos = []
for i, video_ann in enumerate(video_anns):
    try:
        while ps_ann_idx < len(ps_anns):
            if librosa.get_duration(path=ps_anns[ps_ann_idx]["audio_filepath"]) > 10 and len(ps_anns[ps_ann_idx]["audio_filepath"]) < (256 - 20): # TODO: @Balu the 2nd cond. confirms that the final path that is created isn't greater than 256 chars.
                ps_ann = ps_anns[ps_ann_idx]
                ps_ann_idx += 1
                break
            else:
                ps_ann_idx += 1
        text = ps_ann["text"]
        os.makedirs(config["mixed_audio_folder"], exist_ok=True)
        os.makedirs(config["output_file"].rsplit('/', 1)[0], exist_ok=True)
        audio_filepath = os.path.join(config["mixed_audio_folder"], video_ann["YTID"]+'_'+ ps_ann["audio_filepath"].split("/")[-2][:6]+'_'+ ps_ann["audio_filepath"].split("/")[-1].split('.')[0] + ".wav")
        video_feat_path = os.path.join(config["video_feats_base_folder"], video_ann["YTID"] + ".npy")
        as_info = json.load(open(config["as_info_file"], 'r'))
        video_path = os.path.join(config['videos_folder'], video_ann['YTID'] + '.mp4')
        
        if not (as_info[video_path]['has_video'] and as_info[video_path]['has_audio'] and os.path.exists(video_feat_path)):
            as_issue_videos.append(video_path)
            continue

        label = video_ann["label"]
        st_time = video_ann["start_time"]
        end_time = video_ann["end_time"]
        ctm_file = ps_ann["words_level_ctm_filepath"]
        ## V0 -> Should shift mixing audios to dataloader.
        mix_audios(f"{video_path}", ps_ann["audio_filepath"], audio_filepath, mix_ratio=snr_ratios[i])
        text = add_label_to_text(label, end_label, st_time, end_time, config["ctm_file_prefix"] + ctm_file)
        final_manifest.append(
            {
                "audio_filepath": audio_filepath,
                "feature_file": video_feat_path,
                "duration": 10,
                "snr_ratio": snr_ratios[i],
                "text": text,
                "label": label
            }
        )
    except Exception as e:
        print(f"Error: {e}")
        break
    
json.dump(as_issue_videos, open(config["as_video_issues_file"], 'w'))
write_json_lines(final_manifest, config["output_file"])