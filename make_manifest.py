import json
from read_ctm import read_file
import random
import pandas as pd
import numpy as np
import librosa
import os
from mixer_new import mix_audios
from tqdm import tqdm


def add_label_to_text_it1(label, endlabel, st_time, end_time, ctm_file, add_label=True, add_label_at="END"):
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
    if not add_label:
        final_text = " ".join([word[0] for word in till_10_sec])
        return final_text.strip()
    # compare the start time and end time with the words in the till_10_sec
    # and add the label to the word.
    final_text = ""
    if not add_label_at:
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
    elif add_label_at == "END":
        final_text = " ".join([word[0] for word in till_10_sec])
        final_text += f" {label}"
        return final_text.strip()
    elif add_label_at == "START":
        final_text = f"{label} "
        final_text += " ".join([word[0] for word in till_10_sec])
        return final_text.strip()



def map_current_anns_to_labels_in_audioset(unique_labels, labels_file, dataset_version):
    """
    Converts all labels to a good format, all unique labels should have mapping 
    starting from N1, N2, N3, ... etc.

    Args:
        unique_labels : List of all unique labels.  

    Returns:
        label_mapping : Mapping of all labels to N1, N2, N3, ... etc.
    """

    # label_mapping["endlabel"] = "<N1>"
    if not os.path.exists(labels_file):
        labels_list = []
    else:
        labels_list = json.load(open(labels_file, 'r'))

    if(do_mapping) and isinstance(labels_list, list):
        label_mapping = {}
        for i, label in enumerate(unique_labels):
            label_mapping[label] = f"<N{i+1}>"
    if isinstance(labels_list, list):
        # assert if all labels are present in the labels file.
        for label in unique_labels:
            if label != "endlabel":
                assert label in labels_list, f"{label} not present in the labels file."
        if dataset_version == 'v1':
            mid_to_label = pd.read_csv(
                '/tmp/bld56_dataset_v1/audioset/annotations/mid_to_display_name.tsv', sep='\t', names=['mid', 'label'])
        if dataset_version == 'v2':
            mid_to_label_file_path = f'/tmp/bld56_dataset_v1/audioset/annotations/class_labels_indices.csv'
            mid_to_label = pd.read_csv(mid_to_label_file_path, sep=',', names=[
                                       'index', 'mid', 'label'])
            # mid_to_label = mid_to_label.set_index('mid').to_dict()['label']
        mid_to_coded_label_and_display_name = {}
        for i, row in mid_to_label.iterrows():
            if row['mid'] in label_mapping.keys():
                mid_to_coded_label_and_display_name[row['mid']] = [
                    row['label'], label_mapping[row['mid']] if row['mid'] in label_mapping else None]
        # mid_to_coded_label_and_display_name["endlabel"] = ["endlabel", "<N1>"]
        json.dump(mid_to_coded_label_and_display_name, open(labels_file, 'w'))
    elif isinstance(labels_list, dict):
        assert len(labels_list) == len(
            unique_labels), f"Labels file should have same number of labels as unique labels. But previous trainset has {len(labels_list)} but current set has {len(unique_labels)}. {set(unique_labels) - set(labels_list.keys()) }extra labels in current set. {set(labels_list.keys()) - set(unique_labels)} extra labels in previous set."
        label_mapping = {k: v[1]
                         for k, v in labels_list.items() if k in unique_labels}
    return label_mapping


def read_video_anns(ann_file, labels_file, dataset_version):
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
    video_anns = [x for x in video_anns if x["label"]]

    unique_labels = list(set([x["label"] for x in video_anns]))
    label_mapping = map_current_anns_to_labels_in_audioset(
        unique_labels, labels_file, dataset_version)
    for ann in video_anns:
        if ann["label"] not in label_mapping:
            print(f"Label not found: {ann['label']}")
            # remove this annotation from the list.
            video_anns.remove(ann)
            continue
        ann["label"] = label_mapping[ann["label"]]

    label_to_videos = {}
    for ann in video_anns:
        if ann["label"] not in label_to_videos:
            label_to_videos[ann["label"]] = []
        # only append if the video is present.
        # if ann["is_present"]:
        label_to_videos[ann["label"]].append(
            {
                "YTID": ann["YTID"],
                "start_time": ann["start_time"],
                "end_time": ann["end_time"],
                "label": ann["label"]
            })

    video_anns_req = []
    for ann in video_anns:
        # if ann["is_present"]:
        video_anns_req.append(
            # ann["YTID"], ann["start_time"], ann["end_time"], ann["label"]
            {
                "YTID": ann["YTID"],
                "start_time": ann["start_time"],
                "end_time": ann["end_time"],
                "label": ann["label"]
            }
        )
        # else:
        #     # pick the video with the same label any random video.
        #     random_ann = random.choice(label_to_videos[ann["label"]])
        #     video_anns_req.append(
        #         # random_ann["YTID"], random_ann["start_time"], random_ann["end_time"], ann["label"]
        #         {
        #             "YTID": random_ann["YTID"],
        #             "start_time": random_ann["start_time"],
        #             "end_time": random_ann["end_time"],
        #             "label": ann["label"]
        #         }
        #     )

    return video_anns_req


def load_json_lines(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            try:
                data.append(json.loads(line.strip()))
            except:
                print(f"Error: {line}")
                # raise ValueError(f"Error in loading the file {file_path}")
    print(f"Loaded {len(data)} lines from {file_path} out of {len(data)}")
    return data


def load_ctm_anns(ctm_ann_file):
    """
    Reads CTM annotations file and replaces labels with the mapped labels.

    Args:
        ctm_ann_file : CTM Annotations file.

    Returns:
        ctm_anns : List of all CTM annotations. 
        [YTID, start_time, end_time, label]
    """

    # {"audio_filepath": "/tmp/bld56_dataset_v1/peoples_speech/downloads/extracted/496fee4d8a940f49701c485feccb9cb4adc7bfeeb089790a3121a6130d9583ce/1_16_2018_Williston_Selectboard_SLASH_1_16_2018_Williston_Selectboard_DOT_mp3_00010.flac", "text": "type of thing and certainly brian did a good job explaining sort of where he's coming from i guess my observation is we could have somebody also in the same position who might advocate from a totally different position might", "tokens_level_ctm_filepath": "aligned_peoples_speech/ctm/tokens/1_16_2018_Williston_Selectboard_SLASH_1_16_2018_Williston_Selectboard_DOT_mp3_00010.ctm", "words_level_ctm_filepath": "aligned_peoples_speech/ctm/words/1_16_2018_Williston_Selectboard_SLASH_1_16_2018_Williston_Selectboard_DOT_mp3_00010.ctm", "segments_level_ctm_filepath": "aligned_peoples_speech/ctm/segments/1_16_2018_Williston_Selectboard_SLASH_1_16_2018_Williston_Selectboard_DOT_mp3_00010.ctm"}
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


def make_manifest_for_av():
    if split_type == 'train':
        config = config_train
    elif split_type == 'eval':
        config = config_eval
    elif split_type == 'test':
        config = config_test
    else:
        raise ValueError(
            f"Invalid split type. Should be train or eval. Got {split_type}")

    print(f"Using config: {config}")
    video_anns = read_video_anns(
        config["videos_ann_file"], config["labels_list_file"], dataset_version)
    video_anns_test = read_video_anns(
        config_test["videos_ann_file"], config_test["labels_list_file"], dataset_version)
    video_anns = video_anns + video_anns_test
    random.seed(config["seed"])
    random.shuffle(video_anns)
    # ps_anns = load_ctm_anns(config["ctm_ann_file"]) #! EDIT 2 for LS
    ps_anns = load_json_lines(config["ls_ann_file"])
    # remove audio_filepath from ps_anns that are present in pretrained train manifest.
    # remove audio_filepath from ps_anns if it is present in the output_file_eval
    output_train_manifest = load_json_lines(pretrain_config["pretraining_train_manifest_path"])
    pretrain_audio_files = [x["audio_filepath"] for x in output_train_manifest]
    ps_anns_audio_files = [x["audio_filepath"] for x in ps_anns]
    # remove set of pretrain_audio_files from ps_anns_audio_files
    new_ps_anns_audio_files = list(set(ps_anns_audio_files) - set(pretrain_audio_files))
    ps_anns = [x for x in ps_anns if x["audio_filepath"] in new_ps_anns_audio_files]
    if split_type == 'test':
        # remove audio_filepath from ps_anns if it is present in the output_file_eval
        output_eval_manifest = load_json_lines(config["output_file_eval"])
        eval_audio_files = [x["audio_filepath"] for x in output_eval_manifest]
        ps_anns = [x for x in ps_anns if x["audio_filepath"] not in eval_audio_files]
        
    end_label = "<N1>"
    random.seed(config["seed"])
    final_manifest = []
    # snr_ratios = np.random.uniform(0.2, 0.5, len(video_anns))
    snr_ratios = [0.7] * len(video_anns)
    ps_ann_idx = 0
    as_issue_videos = []
    # for i, video_ann in enumerate(video_anns):
    # do with tqdm
    if "_label_at_start" in config["output_file"]:
        add_label_at = "START"
    else:
        add_label_at = "END"
    add_label = True if "no_label" not in config["output_file"] else False
    for i, video_ann in tqdm(enumerate(video_anns), total=len(video_anns)):
        try:
            while ps_ann_idx < len(ps_anns):
                # if librosa.get_duration(path=ps_anns[ps_ann_idx]["audio_filepath"]) > 10 and len(ps_anns[ps_ann_idx]["audio_filepath"]) < (256 - 20): # TODO: @Balu the 2nd cond. confirms that the final path that is created isn't greater than 256 chars.
                if librosa.get_duration(path=ps_anns[ps_ann_idx]["audio_filepath"]) > 10 or True: #! EDIT 4 for LS
                    ps_ann = ps_anns[ps_ann_idx]
                    ps_ann_idx += 1
                    break
                else:
                    ps_ann_idx += 1
            text = ps_ann["text"]
            # os.makedirs(config["mixed_audio_folder"], exist_ok=True)
            os.makedirs(config["output_file"].rsplit('/', 1)[0], exist_ok=True)
            # audio_filepath = os.path.join(config["mixed_audio_folder"], video_ann["YTID"]+'_'+ ps_ann["audio_filepath"].split("/")[-2][:6]+'_'+ ps_ann["audio_filepath"].split("/")[-1].split('.')[0] + ".wav")
            audio_filepath = ps_ann["audio_filepath"]
            video_feat_path = os.path.join(
                config["video_feats_base_folder"], video_ann["YTID"] + ".npy")
            feats = np.load(video_feat_path)
            if feats.shape[0] != 50:
                print(f"Shape mismatch: {feats.shape} for {video_feat_path}")
                continue
            # as_info = json.load(open(config["as_info_file"], 'r'))
            video_path = os.path.join(
                config['videos_folder'], video_ann['YTID'] + '.mp4')

            # if not (as_info[video_path]['has_video'] and as_info[video_path]['has_audio'] and os.path.exists(video_feat_path)):
            #     as_issue_videos.append(video_path)
            #     continue

            label = video_ann["label"]
            # st_time = video_ann["start_time"] #! EDIT 5 for LS
            # end_time = video_ann["end_time"]
            # ctm_file = ps_ann["words_level_ctm_filepath"]
            # V0 -> Should shift mixing audios to dataloader.
            # mix_audios(f"{video_path}", ps_ann["audio_filepath"], audio_filepath, mix_ratio=snr_ratios[i])
            # text = add_label_to_text_it1(label, end_label, st_time, end_time, #! EDIT 1 for LS
            #                              config["ctm_file_prefix"] + ctm_file, add_label=add_label, add_label_at=add_label_at) # START, END, None
            final_manifest.append(
                {
                    "audio_filepath": audio_filepath,
                    "feature_file": video_feat_path,
                    "video_filepath": video_path,
                    "duration": 10,
                    "snr_ratio": snr_ratios[i],
                    "text": text,
                    "label": label
                }
            )

        except Exception as e:
            print(f"Error: {e}")
            break
    print(f"Add label: {add_label}")

    json.dump(as_issue_videos, open(config["as_video_issues_file"], 'w'))
    write_json_lines(final_manifest, config["output_file"])

def make_manifest_for_pretraining_au():
    config = config_train
    print(f"Using config: {config}")
    # keep video_feat_files "None" for pretraining.
    # get all files from the ctm_ann_file, that are not used in output_file.
    ps_anns = load_ctm_anns(config["ctm_ann_file"])
    output_train_manifest = load_json_lines(config["output_file"])
    output_train_audio_files = [x["audio_filepath"]
                                for x in output_train_manifest]
    output_train_audio_files = set(output_train_audio_files)
    # list all ps_anns['audio_filepath'] that are not in output_train_audio_files
    ps_anns = [x for x in ps_anns if x["audio_filepath"]
               not in output_train_audio_files]
    # split this into train and eval.
    random.shuffle(ps_anns)
    train_split = int(len(ps_anns) * 0.9)
    final_manifests = {
        "train": ps_anns[:train_split],
        "eval": ps_anns[train_split:]
    }

    for split in final_manifests.keys():
        final_manifest = []
        print(f"Making manifest for {split}")
        for tqdm_idx, ps_ann in tqdm(enumerate(final_manifests[split]), total=len(final_manifests[split])):
            if os.path.exists(ps_ann["audio_filepath"]) and os.path.exists(config["ctm_file_prefix"] + ps_ann["words_level_ctm_filepath"]):
                audio_filepath = ps_ann["audio_filepath"]
                text = add_label_to_text_it1(None, None, None, None, ctm_file=config["ctm_file_prefix"] + ps_ann["words_level_ctm_filepath"], add_label=False)
                final_manifest.append(
                    {
                        "audio_filepath": audio_filepath,
                        "feature_file": "/tmp/bld56_dataset_v1/it2/-0BIyqJj9ZU.npy",
                        "video_filepath": "/tmp/bld56_dataset_v1/it2/-0BIyqJj9ZU.npy",
                        "duration": 10,
                        "snr_ratio": 0.7,
                        "text": text,
                        "label": "N140"
                    }
                )
        write_json_lines(
            final_manifest, config[f"pretraining_{split}_manifest_path"])



"""
Main function to read all the files and generate the final manifest file.
"""
split_type = 'eval'  # 'train' or 'eval' or 'test'
dataset_version = 'v2'
do_mapping = False
config_eval = {
    "split_type": ['eval'],  # ['train', 'eval']
    "ctm_ann_file": "/tmp/bld56_dataset_v1/peoples_speech/validation/aligned_peoples_speech/manifest_with_output_file_paths.json",
    "ls_ann_file": "/tmp/bld56_dataset_v1/librispeech/test-clean-manifest.json",
    "videos_ann_file": "/tmp/bld56_dataset_v1/audioset/v2/audioset_anns_final_eval_normal.json",
    "videos_folder": "/tmp/bld56_dataset_v1/audioset/train/videos/",
    "mixed_audio_folder": "/tmp/bld56_dataset_v1/it2/mixed_audios_eval/",
    "ctm_file_prefix": "/tmp/bld56_dataset_v1/peoples_speech/validation/",
    "labels_list_file": "/tmp/bld56_dataset_v1/audioset/v2/it2_audioset_labels.json",
    "output_file": "/tmp/bld56_dataset_v1/it2/annotations/ls_test_clean_manifest_eval_no_label.json", #! EDIT 3 for LS
    # "output_file": "/tmp/bld56_dataset_v1/it2/annotations/manifest_eval.json",
    # "output_file": "/tmp/bld56_dataset_v1/it2/annotations/manifest_eval_label_at_start.json",
    "as_info_file": "/tmp/bld56_dataset_v1/audioset/eval/info.json",
    "video_feats_base_folder": "/tmp/bld56_dataset_v1/audioset/train/feats_ViT-L-14_5fps/",
    "as_video_issues_file": "/tmp/bld56_dataset_v1/audioset/annotations/it1_70_as_issues_eval.json",
    "seed": 42,
}
config_test = {
    "split_type": ['test'],  # ['train', 'eval']
    "ctm_ann_file": "/tmp/bld56_dataset_v1/peoples_speech/validation/aligned_peoples_speech/manifest_with_output_file_paths.json",
    "videos_ann_file": "/tmp/bld56_dataset_v1/audioset/v2/audioset_anns_final_test_normal.json",
    "videos_folder": "/tmp/bld56_dataset_v1/audioset/train/videos/",
    "mixed_audio_folder": "/tmp/bld56_dataset_v1/it2/mixed_audios_eval/",
    "ctm_file_prefix": "/tmp/bld56_dataset_v1/peoples_speech/validation/",
    "labels_list_file": "/tmp/bld56_dataset_v1/audioset/v2/it2_audioset_labels.json",
    # "output_file": "/tmp/bld56_dataset_v1/it2/annotations/manifest_test_no_label.json",
    "output_file": "/tmp/bld56_dataset_v1/it2/annotations/manifest_test.json",
    # "output_file": "/tmp/bld56_dataset_v1/it2/annotations/manifest_test_label_at_start.json",
    # "output_file_eval": "/tmp/bld56_dataset_v1/it2/annotations/manifest_eval_no_label.json",
    "output_file_eval": "/tmp/bld56_dataset_v1/it2/annotations/manifest_eval.json",
    # "output_file_eval": "/tmp/bld56_dataset_v1/it2/annotations/manifest_eval_label_at_start.json",
    "as_info_file": "/tmp/bld56_dataset_v1/audioset/eval/info.json",
    "video_feats_base_folder": "/tmp/bld56_dataset_v1/audioset/train/feats_ViT-L-14_5fps/",
    "as_video_issues_file": "/tmp/bld56_dataset_v1/audioset/annotations/it1_70_as_issues_eval.json",
    "seed": 42,
}
config_train = {
    "split_type": ['train'],  # ['train', 'eval']
    "ctm_ann_file": "/tmp/bld56_dataset_v1/peoples_speech/train_clean/aligned_peoples_speech/total_manifest_with_output_file_paths.json",
    "videos_ann_file": "/tmp/bld56_dataset_v1/audioset/v2/audioset_anns_final_train_normal.json",
    "videos_folder": "/tmp/bld56_dataset_v1/audioset/train/videos/",
    "mixed_audio_folder": "/tmp/bld56_dataset_v1/it2/mixed_audios_train/",
    "ctm_file_prefix": "/tmp/bld56_dataset_v1/peoples_speech/train_clean/",
    "labels_list_file": "/tmp/bld56_dataset_v1/audioset/v2/it2_audioset_labels.json",
    # "output_file": "/tmp/bld56_dataset_v1/it2/annotations/manifest_train_no_label.json",
    "output_file": "/tmp/bld56_dataset_v1/it2/annotations/manifest_train.json",
    # "output_file": "/tmp/bld56_dataset_v1/it2/annotations/manifest_train_label_at_start.json",
    "as_info_file": "/tmp/bld56_dataset_v1/audioset/train/info.json",
    "video_feats_base_folder": "/tmp/bld56_dataset_v1/audioset/train/feats_ViT-L-14_5fps/",
    "as_video_issues_file": "/tmp/bld56_dataset_v1/audioset/annotations/it1_70_as_issues_train.json",
    "seed": 42,
}

pretrain_config = {
    "pretraining_train_manifest_path": "/tmp/bld56_dataset_v1/it2/annotations/pretraining_train_manifest.json",
    "pretraining_eval_manifest_path": "/tmp/bld56_dataset_v1/it2/annotations/pretraining_eval_manifest.json"
}

# make_manifest_for_pretraining_au()
make_manifest_for_av()