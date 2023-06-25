from lrce.lib import *
from lrce.dataset.raw_dataset import RawTGIFMCTextDataset, RawTextDataset, RawTGIFOETextDataset, RawTGIFMCTextDatasetV2


class TextExtractor(T.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=False,
        ).last_hidden_state


def extract_features(
    dataset_path: str,
    out_path: str,
    batch_size: int = 5,
    max_token_len: Union[int, List[int]] = 30,
    is_tgif: bool = False,
    is_mc: bool = False,
):
    text_extractor = TextExtractor()
    text_extractor.to(DEVICE)
    text_extractor.eval()

    if is_tgif:
        if is_mc:
            dataset = RawTGIFMCTextDatasetV2(dataset_path, max_token_len=max_token_len)
        else:
            dataset = RawTGIFOETextDataset(dataset_path, max_token_len=max_token_len)
    else:
        dataset = RawTextDataset(dataset_path, max_token_len=max_token_len)
    dataloader = T.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # h5file = h5py.File(out_path, 'w', libver='latest')
    os.makedirs(out_path, exist_ok=True)

    with T.no_grad():
        for batch in tqdm(dataloader):

            if is_mc:
                # qa_ids, q, a1, a2, a3, a4, a5, sep = batch
                # q = q.to(DEVICE)
                # sep = sep.to(DEVICE)

                # a1_result = text_extractor(T.concat((q, sep, a1.to(DEVICE)), dim=1))
                # a2_result = text_extractor(T.concat((q, sep, a2.to(DEVICE)), dim=1))
                # a3_result = text_extractor(T.concat((q, sep, a3.to(DEVICE)), dim=1))
                # a4_result = text_extractor(T.concat((q, sep, a4.to(DEVICE)), dim=1))
                # a5_result = text_extractor(T.concat((q, sep, a5.to(DEVICE)), dim=1))

                # a_result = T.stack([a1_result, a2_result, a3_result, a4_result, a5_result], dim=1)

                qa_ids, qa1, qa2, qa3, qa4, qa5 = batch
                qa1_result = text_extractor(qa1.to(DEVICE))
                qa2_result = text_extractor(qa2.to(DEVICE))
                qa3_result = text_extractor(qa3.to(DEVICE))
                qa4_result = text_extractor(qa4.to(DEVICE))
                qa5_result = text_extractor(qa5.to(DEVICE))

                qa_result = T.stack([qa1_result, qa2_result, qa3_result, qa4_result, qa5_result], dim=1)

                for i, id in enumerate(qa_ids):
                    # with open(os.path.join(out_path, f'{id}-q.pkl'), 'wb') as f:
                    #     pickle.dump(q_result[i].detach().cpu().numpy(), f)
                    # with open(os.path.join(out_path, f'{id}.pkl'), 'wb') as f:
                    #     pickle.dump(a_result[i].detach().cpu().numpy(), f)
                    with open(os.path.join(out_path, f'{id}.pkl'), 'wb') as f:
                        pickle.dump(qa_result[i].detach().cpu().numpy(), f)
            else:
                qa_ids, data = batch
                data = data.to(DEVICE)
                result = text_extractor(data)
                for id, q_features in zip(qa_ids, result):
                    with open(os.path.join(out_path, f'{id}.pkl'), 'wb') as f:
                        pickle.dump(q_features.detach().cpu().numpy(), f)

    # h5file.close()


if __name__ == '__main__':
    # extract_features(
    #     '/mnt/hdd/Dataset/TGIF-QA/annotations/Total_frameqa_question.csv',
    #     '/mnt/hdd/Dataset/TGIF-QA/FRAMEQA-question-extracted-no-special-char',
    #     batch_size=2048,
    #     max_token_len=28,
    #     is_tgif=True,
    # )
    extract_features(
        '/mnt/hdd/Dataset/TGIF-QA/annotations/Total_action_question.csv',
        '/mnt/hdd/Dataset/TGIF-QA/ACTION-qa-extracted',
        batch_size=2048,
        max_token_len=40,
        is_tgif=True,
        is_mc=True,
    )
    # extract_features(
    #     '/mnt/hdd/Dataset/TGIF-QA/annotations/Total_count_question.csv',
    #     '/mnt/hdd/Dataset/TGIF-QA/COUNT-question-extracted',
    #     batch_size=2048,
    #     max_token_len=30,
    #     is_tgif=True,
    #     is_mc=False,
    # )
    # extract_features(
    #     '/mnt/hdd/Dataset/TGIF-QA/annotations/Total_transition_question.csv',
    #     '/mnt/hdd/Dataset/TGIF-QA/TRANSITION-qa-extracted-4',
    #     batch_size=2048,
    #     max_token_len=40,
    #     is_tgif=True,
    #     is_mc=True,
    # )

    # extract_features(
    #     '/mnt/hdd/Dataset/MSVD-QA/train_qa.json',
    #     '/mnt/hdd/Dataset/MSVD-QA/question-extracted',
    #     batch_size=2048,
    #     max_token_len=32,
    #     is_tgif=False,
    #     is_mc=False,
    # )
    # extract_features(
    #     '/mnt/hdd/Dataset/MSVD-QA/test_qa.json',
    #     '/mnt/hdd/Dataset/MSVD-QA/question-extracted',
    #     batch_size=2048,
    #     max_token_len=32,
    #     is_tgif=False,
    #     is_mc=False,
    # )
    # extract_features(
    #     '/mnt/hdd/Dataset/MSVD-QA/val_qa.json',
    #     '/mnt/hdd/Dataset/MSVD-QA/question-extracted',
    #     batch_size=2048,
    #     max_token_len=32,
    #     is_tgif=False,
    #     is_mc=False,
    # )

    # extract_features(
    #     '/mnt/hdd/Dataset/MSRVTT-QA/train_qa.json',
    #     '/mnt/hdd/Dataset/MSRVTT-QA/question-extracted',
    #     batch_size=2048,
    #     max_token_len=37,
    #     is_tgif=False,
    #     is_mc=False,
    # )
    # extract_features(
    #     '/mnt/hdd/Dataset/MSRVTT-QA/test_qa.json',
    #     '/mnt/hdd/Dataset/MSRVTT-QA/question-extracted',
    #     batch_size=2048,
    #     max_token_len=37,
    #     is_tgif=False,
    #     is_mc=False,
    # )
    # extract_features(
    #     '/mnt/hdd/Dataset/MSRVTT-QA/val_qa.json',
    #     '/mnt/hdd/Dataset/MSRVTT-QA/question-extracted',
    #     batch_size=2048,
    #     max_token_len=37,
    #     is_tgif=False,
    #     is_mc=False,
    # )
