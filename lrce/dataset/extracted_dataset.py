from lrce.lib import *
import time


#DEPRECATED DO NOT USE
class ExtractedDataset(T.utils.data.Dataset):
    def __init__(
        self,
        video_features_path: str,
        text_features_path: str,
        train_annotation: str,
        val_annotation: str,
        test_annotation: str,
        video_dict: Dict,
        split: str = 'train',
        sanity_check: bool = False,
    ) -> None:
        super().__init__()

        assert os.path.exists(video_features_path), f'Path {video_features_path} does not exist'
        assert os.path.exists(text_features_path), f'Path {text_features_path} does not exist'

        self.video_features_path = video_features_path
        self.text_features_path = text_features_path

        assert split in ['train', 'val', 'test'], 'split value can only be train, val, or test'
        self.split = split

        assert os.path.exists(train_annotation), f'{train_annotation} does not exist'
        assert os.path.exists(val_annotation), f'{val_annotation} does not exist'
        assert os.path.exists(test_annotation), f'{test_annotation} does not exist'

        if self.split == 'train':
            self.json_file = json.load(open(train_annotation, 'r'))
        elif self.split == 'val':
            self.json_file = json.load(open(val_annotation, 'r'))
        elif self.split == 'test':
            self.json_file = json.load(open(test_annotation, 'r'))

        # self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
        # requires all split to build the answer dictionary to know how many unique answers
        self.answer_dict = build_answer_dict([train_annotation, val_annotation, test_annotation])
        self.video_dict = video_dict
        self.sanity_check = sanity_check

    def __len__(self):
        if self.sanity_check:
            return SANITY_CHECK_SIZE

        return len(self.json_file)

    def __getitem__(self, idx):
        qa = self.json_file[idx]
        id = qa['id']
        vid_name = self.video_dict[qa['video_id']]
        answer = qa['answer']

        with open(os.path.join(self.video_features_path, f'{vid_name}.pkl'), 'rb') as f:
            #(BATCH, TEMPORAL_SCALE, SEQ_LEN, H*W, FEATURES_DIM)
            video_features = pickle.load(f)
        with open(os.path.join(self.text_features_path, f'{id}.pkl'), 'rb') as f:
            #(BATCH, SEQ_LEN, FEATURES_DIM)
            text_features = pickle.load(f)

        # cls_token = self.tokenizer.encode(CLS_TOKEN, add_special_tokens=False)
        # sep_token = self.tokenizer.encode(SEP_TOKEN, add_special_tokens=False)

        return (
            torch.FloatTensor(video_features),
            torch.FloatTensor(text_features),
            # torch.LongTensor(cls_token),
            # torch.LongTensor(sep_token),
            torch.LongTensor([self.answer_dict[answer]]).squeeze(),
        )


class PureExtracted(T.utils.data.Dataset):
    # support for loading the whole video and text features directly to memory
    # requires very big RAM
    def __init__(
        self,
        video_features_dict: Dict,
        text_features_dict: Dict,
        train_annotation: str,
        val_annotation: str,
        test_annotation: str,
        split: str = 'train',
        sanity_check: bool = False,
    ) -> None:
        super().__init__()

        assert split in ['train', 'val', 'test'], 'split value can only be train, val, or test'
        self.split = split

        assert os.path.exists(train_annotation), f'{train_annotation} does not exist'
        assert os.path.exists(val_annotation), f'{val_annotation} does not exist'
        assert os.path.exists(test_annotation), f'{test_annotation} does not exist'

        if self.split == 'train':
            self.json_file = json.load(open(train_annotation, 'r'))
        elif self.split == 'val':
            self.json_file = json.load(open(val_annotation, 'r'))
        elif self.split == 'test':
            self.json_file = json.load(open(test_annotation, 'r'))

        self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
        # requires all split to build the answer dictionary to know how many unique answers
        self.answer_dict = build_answer_dict([train_annotation, val_annotation, test_annotation])
        self.video_features_dict = video_features_dict
        self.text_features_dict = text_features_dict
        self.sanity_check = sanity_check

        self.logger = logging.getLogger(__name__)

    def __len__(self):
        if self.sanity_check:
            return 100

        return len(self.json_file)

    def __getitem__(self, idx):
        qa = self.json_file[idx]
        id = qa['id']
        vid_id = qa['video_id']
        answer = qa['answer']

        video_features = self.video_features_dict[vid_id]  #(TEMPORAL_SCALE, TEMPORAL, H*W, FEATURES_DIM)
        text_features = self.text_features_dict[id]  #(SEQ_LEN, FEATURES_DIM)

        cls_token = self.tokenizer.encode(CLS_TOKEN, add_special_tokens=False)
        sep_token = self.tokenizer.encode(SEP_TOKEN, add_special_tokens=False)

        return (
            torch.FloatTensor(video_features),
            torch.FloatTensor(text_features),
            torch.LongTensor(cls_token),
            torch.LongTensor(sep_token),
            torch.LongTensor([self.answer_dict[answer]]).squeeze(),
        )


class TGIFExtractedDataset(T.utils.data.Dataset):
    def __init__(
        self,
        video_features_path: str,
        text_features_path: str,
        split_annotation: str,
        full_annotation: str,
        sanity_check: bool = False,
        task_type: str = 'oe',
    ) -> None:
        super().__init__()

        assert os.path.exists(video_features_path), f'Path {video_features_path} does not exist'
        assert os.path.exists(text_features_path), f'Path {text_features_path} does not exist'

        self.video_features_path = video_features_path
        self.text_features_path = text_features_path

        assert os.path.exists(split_annotation), f'{split_annotation} does not exist'
        assert os.path.exists(full_annotation), f'{full_annotation} does not exist'

        self.csv_file = pd.read_csv(split_annotation, delimiter='\t')

        # requires all split to build the answer dictionary to know how many unique answers
        self.answer_dict, _ = parse_tgif_annot(full_annotation, task_type)
        self.sanity_check = sanity_check
        self.task_type = task_type

    def __len__(self):
        if self.sanity_check:
            return SANITY_CHECK_SIZE

        return len(self.csv_file)

    def __getitem__(self, idx):
        qa = self.csv_file.iloc[idx]

        id = qa['vid_id']
        answer = qa['answer']
        vid_name = qa['gif_name']

        with open(os.path.join(self.video_features_path, f'{vid_name}.pkl'), 'rb') as f:
            video_features = pickle.load(f)

        if self.task_type == 'mc':
            # with open(os.path.join(self.text_features_path, f'{id}-q.pkl'), 'rb') as f:
            #     q_features = pickle.load(f)  #(Q_SEQ_LEN, FEATURES_DIM)
            # with open(os.path.join(self.text_features_path, f'{id}-a.pkl'), 'rb') as f:
            #     a_features = pickle.load(f)  #(TOTAL_MC, A_SEQ_LEN, FEATURES_DIM)

            # return (
            #     torch.FloatTensor(video_features),
            #     torch.FloatTensor(q_features),
            #     torch.FloatTensor(a_features),
            #     torch.LongTensor([self.answer_dict[answer]]).squeeze(),
            # )
            with open(os.path.join(self.text_features_path, f'{id}.pkl'), 'rb') as f:
                qa_features = pickle.load(f)  #(QA_SEQ_LEN, FEATURES_DIM)

            return (
                torch.FloatTensor(video_features),
                torch.FloatTensor(qa_features),
                torch.LongTensor([self.answer_dict[answer]]).squeeze(),
            )
        else:
            with open(os.path.join(self.text_features_path, f'{id}.pkl'), 'rb') as f:
                text_features = pickle.load(f)
            if self.task_type == 'oe':
                gt = torch.LongTensor([self.answer_dict[answer]]).squeeze()
            else:
                gt = torch.FloatTensor([self.answer_dict[answer]]).squeeze()

            return torch.FloatTensor(video_features), torch.FloatTensor(text_features), gt