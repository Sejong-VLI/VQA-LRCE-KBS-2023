from lrce.lib import *
import time


#DEPRECATED DO NOT USE
class RawVideoDataset(T.utils.data.Dataset):
    """
    Extracts directly from video files
    """
    def __init__(
            self,
            videos_path: str,
            frames_per_clip: int = 5,
            temporal_scale: List[int] = [1, 2, 3],
            frame_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()

        assert os.path.exists(videos_path), f'Path {videos_path} does not exist'

        self.frame_size = frame_size
        self.all_videos = [video for video in os.listdir(videos_path) if video.endswith(VIDEO_EXT)]
        self.videos_path = videos_path
        self.frames_per_clip = frames_per_clip
        self.temporal_scale = temporal_scale

    def preprocess_frame(self, img):
        w, h = img.size
        img = TV.transforms.Compose([
            # TV.transforms.Pad([0, (w - h) // 2] if w > h else [(h - w) //
            #                                                    2, 0]),  # TODO: might be worth to try disabling
            TV.transforms.Resize(self.frame_size),
            TV.transforms.ToTensor()
        ])(img)
        return img

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        video_name = self.all_videos[idx]
        video_name_no_ext, _ = os.path.splitext(video_name)

        # read all frames from video
        video_frames = []
        vid_cap = cv2.VideoCapture(os.path.join(self.videos_path, video_name))
        success, image = vid_cap.read()

        assert success, f'Error in reading video {video_name}'

        while (success):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            video_frames.append(image)
            success, image = vid_cap.read()

        assert len(video_frames
                   ) >= self.frames_per_clip, f'Error in video {video_name}, too many frames_per_clip, set lower value'

        video_frames = [
            self.preprocess_frame(Image.fromarray(np.uint8(frame)).convert('RGB')) for frame in video_frames
        ]
        video_frames = T.stack(video_frames, dim=0)  # (TOTAL_FRAMES, CHANNEL, WIDTH, HEIGHT)

        multi_scale_res = []
        for scale in self.temporal_scale:
            step_size = max(1, max(1, len(video_frames) // self.frames_per_clip) // scale)
            scale_res_all = video_frames[step_size // 2::step_size]

            scale_res_clips = []
            inner_step_size = (len(scale_res_all) - self.frames_per_clip) // (scale - 1) if scale > 1 else 0
            for i in range(scale):
                clips = scale_res_all[i * inner_step_size:i * inner_step_size +
                                      self.frames_per_clip]  # (FRAMES_PER_CLIP, CHANNEL, WIDTH, HEIGHT)
                assert len(
                    clips
                ) == self.frames_per_clip, f'Mismatch length of clips in scale {scale}, video {video_name}, expected {self.frames_per_clip}, got {len(clips)}'
                scale_res_clips.append(clips)
            scale_res_clips = T.stack(scale_res_clips, dim=0)  # (SCALE, FRAMES_PER_CLIP, CHANNEL, WIDTH, HEIGHT)
            multi_scale_res.append(scale_res_clips)
        multi_scale_res = T.concat(multi_scale_res, dim=0)
        return video_name_no_ext, multi_scale_res  # ((SCALE * (SCALE+1)) / 2, FRAMES_PER_CLIP, CHANNEL, WIDTH, HEIGHT)


class RawVideoFramesDataset(T.utils.data.Dataset):
    """
    Extracts from frames. Videos have to be converted into frames first
    """
    def __init__(
            self,
            video_frames_path: str,
            frames_per_clip: int = 5,
            temporal_scale: List[int] = [1, 2, 3],
            frame_size: Tuple[int, int] = (224, 224),
    ):
        super().__init__()

        assert os.path.exists(video_frames_path), f'Path {video_frames_path} does not exist'

        self.frame_size = frame_size
        self.all_videos = os.listdir(video_frames_path)
        self.video_frames_path = video_frames_path
        self.frames_per_clip = frames_per_clip
        self.temporal_scale = temporal_scale

    def preprocess_frame(self, img):
        w, h = img.size
        img = TV.transforms.Compose([
            TV.transforms.Pad([0, (w - h) // 2] if w > h else [(h - w) //
                                                               2, 0]),  # TODO: might be worth to try disabling
            TV.transforms.Resize(self.frame_size),
            TV.transforms.ToTensor()
        ])(img)
        return img

    def __len__(self):
        return len(self.all_videos)

    def __getitem__(self, idx):
        video_name = self.all_videos[idx]
        video_frames = sorted(os.listdir(f'{self.video_frames_path}/{video_name}'))

        assert len(video_frames
                   ) >= self.frames_per_clip, f'Error in video {video_name}, too many frames_per_clip, set lower value'

        video_frames = [f'{self.video_frames_path}/{video_name}/{frame}' for frame in video_frames]
        video_frames = [self.preprocess_frame(Image.open(frame).convert('RGB')) for frame in video_frames]
        video_frames = T.stack(video_frames, dim=0)  # (TOTAL_FRAMES, CHANNEL, WIDTH, HEIGHT)

        multi_scale_res = []
        for scale in self.temporal_scale:
            step_size = max(1, max(1, len(video_frames) // self.frames_per_clip) // scale)
            scale_res_all = video_frames[step_size // 2::step_size]

            scale_res_clips = []
            inner_step_size = (len(scale_res_all) - self.frames_per_clip) // (scale - 1) if scale > 1 else 0
            for i in range(scale):
                clips = scale_res_all[i * inner_step_size:i * inner_step_size +
                                      self.frames_per_clip]  # (FRAMES_PER_CLIP, CHANNEL, WIDTH, HEIGHT)
                assert len(
                    clips
                ) == self.frames_per_clip, f'Mismatch length of clips in scale {scale}, video {video_name}, expected {self.frames_per_clip}, got {len(clips)}'
                scale_res_clips.append(clips)
            scale_res_clips = T.stack(scale_res_clips, dim=0)  # (SCALE, FRAMES_PER_CLIP, CHANNEL, WIDTH, HEIGHT)
            multi_scale_res.append(scale_res_clips)
        multi_scale_res = T.concat(multi_scale_res, dim=0)
        return video_name, multi_scale_res  # ((SCALE * (SCALE+1)) / 2, FRAMES_PER_CLIP, CHANNEL, WIDTH, HEIGHT)


class RawTextDataset(T.utils.data.Dataset):
    def __init__(self, label_path: str = None, max_token_len: int = 30):
        assert os.path.exists(label_path), 'label_path does not exist'
        self.json_file = json.load(open(label_path, 'r'))
        self.max_token_len = max_token_len
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.json_file)

    def __getitem__(self, idx):
        qa = self.json_file[idx]
        id = qa['id']
        question = qa['question']
        question = self.tokenizer.encode(
            question,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len,
        )
        return id, question.squeeze(0)


class RawTGIFOETextDataset(T.utils.data.Dataset):
    def __init__(self, label_path: str = None, max_token_len: int = 30):
        assert os.path.exists(label_path), f'{label_path} does not exist'
        self.csv_file = pd.read_csv(label_path, delimiter='\t')
        self.max_token_len = max_token_len
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        qa = self.csv_file.iloc[idx]
        id = qa['vid_id']
        question = qa['question']
        question = self.tokenizer.encode(
            question,
            add_special_tokens=False,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len,
        )
        return id, question.squeeze(0)


class RawTGIFMCTextDataset(T.utils.data.Dataset):
    def __init__(self, label_path: str = None, max_token_len: List[int] = 30):
        assert os.path.exists(label_path), f'{label_path} does not exist'
        self.csv_file = pd.read_csv(label_path, delimiter='\t')
        self.max_token_len = max_token_len
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        qa = self.csv_file.iloc[idx]
        id = qa['vid_id']
        question = self.tokenizer.encode(
            qa['question'],
            add_special_tokens=False,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len[0],
        )
        a1 = self.tokenizer.encode(
            qa['a1'],
            add_special_tokens=False,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len[1],
        )
        a2 = self.tokenizer.encode(
            qa['a2'],
            add_special_tokens=False,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len[1],
        )
        a3 = self.tokenizer.encode(
            qa['a3'],
            add_special_tokens=False,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len[1],
        )
        a4 = self.tokenizer.encode(
            qa['a4'],
            add_special_tokens=False,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len[1],
        )
        a5 = self.tokenizer.encode(
            qa['a5'],
            add_special_tokens=False,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len[1],
        )
        sep = self.tokenizer.encode(
            SEP_TOKEN,
            add_special_tokens=False,
            return_tensors='pt',
            padding='max_length',
            max_length=1,
        )
        return (
            id,
            question.squeeze(0),
            a1.squeeze(0),
            a2.squeeze(0),
            a3.squeeze(0),
            a4.squeeze(0),
            a5.squeeze(0),
            sep.squeeze(0),
        )


class RawTGIFMCTextDatasetV2(T.utils.data.Dataset):
    def __init__(self, label_path: str = None, max_token_len: int = 30):
        assert os.path.exists(label_path), f'{label_path} does not exist'
        self.csv_file = pd.read_csv(label_path, delimiter='\t')
        self.max_token_len = max_token_len
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        qa = self.csv_file.iloc[idx]
        id = qa['vid_id']
        qa1 = self.tokenizer.encode(
            qa['question'],
            qa['a1'],
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len,
        )
        qa2 = self.tokenizer.encode(
            qa['question'],
            qa['a2'],
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len,
        )
        qa3 = self.tokenizer.encode(
            qa['question'],
            qa['a3'],
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len,
        )
        qa4 = self.tokenizer.encode(
            qa['question'],
            qa['a4'],
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len,
        )
        qa5 = self.tokenizer.encode(
            qa['question'],
            qa['a5'],
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_token_len,
        )

        return (
            id,
            qa1.squeeze(0),
            qa2.squeeze(0),
            qa3.squeeze(0),
            qa4.squeeze(0),
            qa5.squeeze(0),
        )


if __name__ == '__main__':
    # dataset = RawTGIFTextDataset('/mnt/hdd/Dataset/TGIF-QA/annotations/Total_frameqa_question.csv', 28)
    # max_len = 0
    # max_idx = 0
    # for i, data in enumerate(dataset):
    #     if data[1].shape[0] > max_len:
    #         max_len = data[1].shape[0]
    #         max_idx = i
    #     print(data)
    # print(max_len, max_idx, dataset[max_idx])

    # dataset = RawVideoDataset('/mnt/hdd/Dataset/TGIF-QA/gifs', 5, [1, 2, 3])
    # for i, (name, vid) in enumerate(dataset):
    #     print(i, name)

    video_frames = []
    vid_cap = cv2.VideoCapture('/mnt/hdd/Dataset/TGIF-QA/gifs/tumblr_nippaia6EW1rruv38o1_400.gif')
    success, image = vid_cap.read()

    while (success):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(image.shape)
        video_frames.append(image)
        success, image = vid_cap.read()

    print(len(video_frames))