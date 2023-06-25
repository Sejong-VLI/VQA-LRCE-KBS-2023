from lrce.lib import *


class E2EDatasetBase(T.utils.data.Dataset):
    def __init__(
        self,
        label_path: str,
        videos_path: str,
        frames_per_clip: int = 5,
        temporal_scale: List[int] = [1, 2, 3],
        frame_size: Tuple[int, int] = (224, 224),
        max_text_token_len: int = 30,
        video_dict: Dict = None,
        sanity_check: bool = False,
        is_frame_extracted: bool = False,
    ):
        super().__init__()

        assert os.path.exists(videos_path), f'Path {videos_path} does not exist'
        assert os.path.exists(label_path), f'Path {label_path} does not exist'

        self.label_path = label_path
        self.videos_path = videos_path
        self.frames_per_clip = frames_per_clip
        self.temporal_scale = temporal_scale
        self.frame_size = frame_size
        self.max_text_token_len = max_text_token_len
        self.video_dict = video_dict
        self.sanity_check = sanity_check
        self.is_frame_extracted = is_frame_extracted

        self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
        self._load_label_file()
        self._build_answer_dict()
        self._build_scale_idx()

    def _build_scale_idx(self):
        scale_dict = {
            1: [0],
            2: [1, 2],
            3: [3, 4, 5],
            4: [5, 6, 7, 8],
        }
        self.scale_idx = []
        for scale in self.temporal_scale:
            self.scale_idx += scale_dict[scale]

    def _load_label_file(self):
        raise NotImplementedError()

    def _build_answer_dict(self):
        raise NotImplementedError()

    def __len__(self):
        if self.sanity_check:
            return SANITY_CHECK_SIZE

        return len(self.label_file)

    def _preprocess_frame(self, img):
        img = TV.transforms.Compose([TV.transforms.Resize(self.frame_size), TV.transforms.ToTensor()])(img)
        return img

    def _get_texts(self, idx: int):
        raise NotImplementedError()

    def _get_video_name(self, idx: int):
        raise NotImplementedError()

    def _get_gt(self, idx: int):
        raise NotImplementedError()

    def _get_video_clips(self, video_name: str):
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
            self._preprocess_frame(Image.fromarray(np.uint8(frame)).convert('RGB')) for frame in video_frames
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
        return multi_scale_res  # ((SCALE * (SCALE+1)) / 2, FRAMES_PER_CLIP, CHANNEL, WIDTH, HEIGHT)
    
    def _get_extracted_video_clips(self, video_name:str):
        multi_scale_res = np.load(os.path.join(self.videos_path, f'{video_name}.npy'))
        multi_scale_res = multi_scale_res[self.scale_idx]
        return torch.FloatTensor(multi_scale_res)

    def __getitem__(self, idx):
        video_name = self._get_video_name(idx)
        if self.is_frame_extracted:
            video_clips = self._get_extracted_video_clips(video_name)
        else:
            video_clips = self._get_video_clips(video_name)
        return video_clips, *self._get_texts(idx), self._get_gt(idx)


class E2EMicrosoftDataset(E2EDatasetBase):
    def __init__(
            self,
            train_annotation: str,
            val_annotation: str,
            test_annotation: str,
            videos_path: str,
            video_dict: Dict,
            split: str = 'train',
            frames_per_clip: int = 5,
            temporal_scale: List[int] = [1, 2, 3],
            frame_size: Tuple[int, int] = (224, 224),
            max_text_token_len: int = 30,
            sanity_check: bool = False,
            is_frame_extracted: bool = False,
    ):
        self.split_dict = {'train': train_annotation, 'val': val_annotation, 'test': test_annotation}
        super().__init__(
            self.split_dict[split],
            videos_path,
            frames_per_clip,
            temporal_scale,
            frame_size,
            max_text_token_len,
            video_dict,
            sanity_check,
            is_frame_extracted,
        )

    def _load_label_file(self):
        self.label_file = json.load(open(self.label_path, 'r'))

    def _build_answer_dict(self):
        # self.answer_dict = build_answer_dict([*self.split_dict.values()])
        # self.answer_dict = build_common_answer_dict([self.split_dict['train'], self.split_dict['val']], 1500)
        self.answer_dict = build_common_answer_dict([self.split_dict['train'], self.split_dict['val']], 1000)

    def _get_texts(self, idx: int):
        question = self.tokenizer(
            self.label_file[idx]['question'],
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_text_token_len,
        )

        # (max_text_token_len) each
        return question['input_ids'].squeeze(0), question['attention_mask'].squeeze(0), question['token_type_ids'].squeeze(0)  

    def _get_video_name(self, idx: int):
        video_name = self.video_dict[self.label_file[idx]['video_id']]
        return video_name if self.is_frame_extracted else f'{video_name}.avi'

    def _get_gt(self, idx: int):
        answer = self.label_file[idx]['answer']
        return torch.LongTensor([self.answer_dict.get(answer, IGNORE_INDEX)]).squeeze()


class E2ETGIFDataset(E2EDatasetBase):
    def __init__(
        self,
        split_annotation: str,
        full_annotation: str,
        videos_path: str,
        frames_per_clip: int = 5,
        temporal_scale: List[int] = [1, 2, 3],
        frame_size: Tuple[int, int] = (224, 224),
        max_text_token_len: int = 30,
        task_type: str = 'oe',
        sanity_check: bool = False,
        is_frame_extracted: bool = False,
    ):
        self.full_annotation = full_annotation
        self.task_type = task_type
        super().__init__(
            split_annotation,
            videos_path,
            frames_per_clip,
            temporal_scale,
            frame_size,
            max_text_token_len,
            {},
            sanity_check,
            is_frame_extracted,
        )

    def _load_label_file(self):
        self.label_file = pd.read_csv(self.label_path, delimiter='\t')

    def _build_answer_dict(self):
        self.answer_dict, _ = parse_tgif_annot(self.full_annotation, self.task_type, k=1000)

    def _get_texts(self, idx: int):
        qa = self.label_file.iloc[idx]

        if self.task_type == 'mc':
            qa1 = self.tokenizer(
                qa['question'],
                qa['a1'],
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_text_token_len,
            )
            qa2 = self.tokenizer(
                qa['question'],
                qa['a2'],
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_text_token_len,
            )
            qa3 = self.tokenizer(
                qa['question'],
                qa['a3'],
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_text_token_len,
            )
            qa4 = self.tokenizer(
                qa['question'],
                qa['a4'],
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_text_token_len,
            )
            qa5 = self.tokenizer(
                qa['question'],
                qa['a5'],
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_text_token_len,
            )

            all_qa_ids = T.concat(
                [
                    qa1['input_ids'],
                    qa2['input_ids'],
                    qa3['input_ids'],
                    qa4['input_ids'],
                    qa5['input_ids'],
                ],
                dim=0,
            )
            all_qa_attention_mask = T.concat(
                [
                    qa1['attention_mask'],
                    qa2['attention_mask'],
                    qa3['attention_mask'],
                    qa4['attention_mask'],
                    qa5['attention_mask'],
                ],
                dim=0,
            )
            all_qa_token_type_ids = T.concat(
                [
                    qa1['token_type_ids'],
                    qa2['token_type_ids'],
                    qa3['token_type_ids'],
                    qa4['token_type_ids'],
                    qa5['token_type_ids'],
                ],
                dim=0,
            )

            return all_qa_ids, all_qa_attention_mask, all_qa_token_type_ids  # (5, max_text_token_len) each
        else:
            question = self.tokenizer(
                qa['question'],
                add_special_tokens=True,
                return_tensors='pt',
                padding='max_length',
                max_length=self.max_text_token_len,
            )
            # (max_text_token_len) each
            return question['input_ids'].squeeze(0), question['attention_mask'].squeeze(0), question['token_type_ids'].squeeze(0)  

    def _get_video_name(self, idx: int):
        video_name = self.label_file.iloc[idx]['gif_name']
        return video_name if self.is_frame_extracted else f'{video_name}.gif'

    def _get_gt(self, idx: int):
        answer = self.label_file.iloc[idx]['answer']
        if self.task_type == 'count':
            gt = torch.FloatTensor([self.answer_dict[answer]]).squeeze()
        else:
            gt = torch.LongTensor([self.answer_dict.get(answer, IGNORE_INDEX)]).squeeze()
        return gt



if __name__ == '__main__':
    # video_dict = pickle.load(open('/data/Steve/MSRVTT-QA/idx-video-mapping.pkl', 'rb'))
    # print(video_dict)
    # dataset = E2EMicrosoftDataset(
    #     train_annotation='/mnt/hdd/Dataset/MSVD-QA/train_qa.json',
    #     val_annotation='/mnt/hdd/Dataset/MSVD-QA/val_qa.json',
    #     test_annotation='/mnt/hdd/Dataset/MSVD-QA/test_qa.json',
    #     videos_path='/mnt/hdd/Dataset/MSVD-QA/video',
    #     video_dict=video_dict,
    #     max_text_token_len=40,
    #     split='train',
    #     sanity_check=False,
    # )
    # dataset = E2ETGIFDataset(
    #     split_annotation='/mnt/hdd/Dataset/TGIF-QA/annotations/Train_transition_question.csv',
    #     full_annotation='/mnt/hdd/Dataset/TGIF-QA/annotations/Train_transition_question.csv',
    #     videos_path='/mnt/hdd/Dataset/TGIF-QA/gifs',
    #     max_text_token_len=40,
    #     task_type='mc',
    #     sanity_check=False,
    # )
    # dataloader = T.utils.data.DataLoader(
    #     dataset,
    #     batch_size=32,
    #     shuffle=True,
    #     num_workers=2,
    #     pin_memory=True,
    # )
    # a, b, c = next(iter(dataloader))
    # a_0 = a[0].to(DEVICE)
    # a_1 = a[1].to(DEVICE)
    # a_2 = a[2].to(DEVICE)
    # b = b.to(DEVICE)
    # c = c.to(DEVICE)
    # tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
    
    # print(a_0.shape, a_1.shape, a_2.shape, b.shape, c.shape)
    # print(
    #     a_0[0][0],
    #     a_1[0][0],
    #     a_2[0][0],
    # )
    # print(tokenizer.decode(a_0[0][0]))
    # print(tokenizer.decode(a_0[0][1]))
    # print(tokenizer.decode(a_0[0][2]))
    # print(tokenizer.decode(a_0[0][3]))
    # print(tokenizer.decode(a_0[0][4]))

    # video_dict = pickle.load(open('/data/Steve/MSRVTT-QA/idx-video-mapping.pkl', 'rb'))
    # dataset = E2EMicrosoftDataset(
    #     train_annotation='/data/Steve/MSRVTT-QA/train_qa.json',
    #     val_annotation='/data/Steve/MSRVTT-QA/val_qa.json',
    #     test_annotation='/data/Steve/MSRVTT-QA/test_qa.json',
    #     videos_path='/data/Steve/MSRVTT-QA/video',
    #     video_dict=video_dict,
    #     max_text_token_len=37,
    #     split='train',
    #     sanity_check=False,
    # )

    # video_names = os.listdir('/data/Steve/MSRVTT-QA/video')
    # out_dir = '/data/Steve/MSRVTT-QA/video-frames-scale-1-2-3'
    # os.makedirs(out_dir, exist_ok=True)
    # for i in tqdm(range(len(video_names))): 
    #     vid_frames = dataset._get_video_clips(video_names[i])
    #     vid_arr = vid_frames.detach().cpu().numpy()
    #     vid_name_only, ext = os.path.splitext(video_names[i])
    #     np.save(os.path.join(out_dir, f'{vid_name_only}.npy'), vid_arr)

    video_dict = pickle.load(open('/data/Steve/MSRVTT-QA/idx-video-mapping.pkl', 'rb'))
    dataset = E2EMicrosoftDataset(
        train_annotation='/data/Steve/MSRVTT-QA/train_qa.json',
        val_annotation='/data/Steve/MSRVTT-QA/val_qa.json',
        test_annotation='/data/Steve/MSRVTT-QA/test_qa.json',
        videos_path='/data/Steve/MSRVTT-QA/video',
        video_dict=video_dict,
        max_text_token_len=37,
        split='train',
        sanity_check=False,
        temporal_scale=[1,2,3,4]
    )

    video_names = os.listdir('/data/Steve/MSRVTT-QA/video')
    out_dir = '/data/Steve/MSRVTT-QA/video-frames-scale-1-2-3-4'
    os.makedirs(out_dir, exist_ok=True)
    for i in tqdm(range(len(video_names))): 
        vid_frames = dataset._get_video_clips(video_names[i])
        vid_arr = vid_frames.detach().cpu().numpy()
        vid_name_only, ext = os.path.splitext(video_names[i])
        np.save(os.path.join(out_dir, f'{vid_name_only}.npy'), vid_arr)
