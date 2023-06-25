import os, pickle, json
import cv2
import logging
import numpy as np
import pandas as pd
from constants import *
from tqdm import tqdm
from typing import List, Tuple, Dict
from utils import *
from functools import lru_cache
from collections import Counter


def video_to_frames(video_path: str = '.', out_dir: str = '.', output_dim: Tuple = (224, 224)) -> None:
    """Converts all videos in video_path to sequence of images to each own directory

    Args:
        video_path (str, optional): . Defaults to '.'.
        out_dir (str, optional): . Defaults to '.'.
        output_dim (Tuple, optional): . Defaults to (224, 224).
    """
    allowed_ext = ['.avi', '.mp4']
    all_videos = [video for video in os.listdir(video_path) if os.path.splitext(video)[-1] in allowed_ext]

    for video in tqdm(all_videos):
        vid_cap = cv2.VideoCapture(f'{video_path}/{video}')
        out_vid_dir = f'{out_dir}/{os.path.splitext(video)[0]}'
        os.makedirs(out_vid_dir, exist_ok=True)

        count = 1
        success, image = vid_cap.read()
        while (success):
            if output_dim:
                image = cv2.resize(image, output_dim)
            cv2.imwrite(f'{out_vid_dir}/{count:03}.jpg', image)
            success, image = vid_cap.read()
            count += 1


def build_video_dict(annotation_file: str, reverse_key: bool = False, start_idx: int = 0) -> Dict:
    """Create video index mapping

    Args:
        annotation_file (str): 

    Returns:
        Dict: 
    """
    video_dict = {}
    with open(annotation_file, 'r') as annot:
        line = annot.readline()
        idx = start_idx

        while line:
            line = line.strip('\n')
            tokens = line.split(' ')
            video_name = tokens[0]

            if video_name not in video_dict:
                video_dict[video_name] = idx
                idx += 1
            line = annot.readline()

    if reverse_key:
        return {val: key for key, val in video_dict.items()}

    return video_dict


def build_answer_dict(annotation_files: List[str], reverse_key: bool = False) -> Dict:
    """Create answer index mapping that contains all possible QA-OE answer

    Args:
        annotation_file (str): 

    Returns:
        Dict: 
    """
    answer_dict = {}
    idx = 0

    for file in annotation_files:
        with open(file, 'r') as f:
            qa_list = json.load(f)

            for qa in qa_list:
                if qa['answer'] not in answer_dict:
                    answer_dict[qa['answer']] = idx
                    idx += 1

    if reverse_key:
        return {val: key for key, val in answer_dict.items()}

    return answer_dict


def build_common_answer_dict(annotation_files: List[str], k: int = 1500, reverse_key: bool = False) -> Dict:
    """Create answer index mapping that contains top-K possible QA-OE answer

    Args:
        annotation_file (str): 

    Returns:
        Dict: 
    """
    answer_list = []

    for file in annotation_files:
        with open(file, 'r') as f:
            qa_list = json.load(f)
            answer_list += list(map(lambda qa: qa['answer'], qa_list))

    answer_occ = Counter(answer_list)
    top_k_answer = answer_occ.most_common(k)
    answer_dict = {val: i for i, (val, _) in enumerate(top_k_answer)}

    if reverse_key:
        return {val: key for key, val in answer_dict.items()}

    return answer_dict


def load_features_to_memory(video_features_path: str, text_features_path: str):
    # requires big RAM depending on the dataset, might need to refactor in the future
    video_features_dict = {}
    text_features_dict = {}

    for file_feature in tqdm(os.listdir(video_features_path)):
        feature_id, _ = os.path.splitext(file_feature)
        video_features_dict[int(feature_id)] = np.load(os.path.join(video_features_path, file_feature))

    for file_feature in tqdm(os.listdir(text_features_path)):
        feature_id, _ = os.path.splitext(file_feature)
        text_features_dict[int(feature_id)] = np.load(os.path.join(text_features_path, file_feature))

    return video_features_dict, text_features_dict


def parse_tgif_annot(file_path: str, task_type: str = 'oe', delimiter: str = '\t', k: int = 1000):
    assert os.path.exists(file_path)

    data = pd.read_csv(file_path, delimiter=delimiter)
    video_dict = pd.Series(data['vid_id'].values, index=data['gif_name']).to_dict()

    if task_type == 'oe':
        all_answer = data['answer'].to_list()
        answer_occ = Counter(all_answer)
        top_k_answer = answer_occ.most_common(k)
        answer_dict = {val: i for i, (val, _) in enumerate(top_k_answer)}
        # unique_answer = set(all_answer)
        # answer_dict = {val: i for i, val in enumerate(unique_answer)}
    else:
        all_answer = data['answer'].to_list()
        answer_dict = {val: val for val in all_answer}
    return answer_dict, video_dict


@lru_cache(maxsize=100000)
def load_npy_with_cache(path: str):
    return np.load(path)


def setup_logging():
    logging.basicConfig(level=logging.INFO)


def get_logger(name: str, rank: int):
    # adapted from https://discuss.pytorch.org/t/ddp-training-log-issue/125808
    class NoOp:
        def __getattr__(self, *args):
            def no_op(*args, **kwargs):
                """Accept every signature by doing non-operation."""
                pass

            return no_op

    if rank == 0:
        return logging.getLogger(name)
    return NoOp()


if __name__ == '__main__':
    # video_to_frames('/mnt/hdd/Dataset/MSVD-QA/video', '/mnt/hdd/Dataset/MSVD-QA/video-frames', None)
    # video_to_frames('/mnt/hdd/Dataset/MSRVTT-QA/video', '/mnt/hdd/Dataset/MSRVTT-QA/video-frames', None)
    # video_dict = build_video_dict('/mnt/hdd/Dataset/MSVD-QA/annotations.txt', start_idx=1)
    # print(video_dict.keys())
    # with open('/mnt/hdd/Dataset/MSVD-QA/video-idx-mapping.pkl', 'wb') as f:
    #     pickle.dump(video_dict, f)
    # answer_dict, max_len = build_answer_dict([
    #     '/mnt/hdd/Dataset/MSVD-QA/test_qa.json',
    #     '/mnt/hdd/Dataset/MSVD-QA/train_qa.json',
    #     '/mnt/hdd/Dataset/MSVD-QA/val_qa.json',
    # ])
    # print(max_len)
    # with open('/mnt/hdd/Dataset/MSVD-QA/answer-idx-mapping.pkl', 'wb') as f:
    #     pickle.dump(answer_dict, f)
    # parse_tgif_annot('/mnt/hdd/Dataset/TGIF-QA/annotations/Total_frameqa_question.csv', False)
    ans_dict = build_common_answer_dict(
        ['/mnt/hdd/Dataset/MSRVTT-QA/train_qa.json', '/mnt/hdd/Dataset/MSRVTT-QA/val_qa.json'])
    print(ans_dict)
