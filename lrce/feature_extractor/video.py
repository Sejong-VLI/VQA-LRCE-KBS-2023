from lrce.lib import *
from lrce.dataset.raw_dataset import RawVideoDataset
from lrce.feature_extractor.video_swin_ori import SwinTransformer3D


class VideoExtractor(T.nn.Module):
    def __init__(self, ckpt_path):
        super().__init__()

        self.swin = SwinTransformer3D(
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            patch_size=(2, 4, 4),
            window_size=(8, 7, 7),
            drop_path_rate=0.2,
            patch_norm=True,
        )

        checkpoint = torch.load(ckpt_path)
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            if 'backbone' in k:
                name = k[9:]
                new_state_dict[name] = v
        self.swin.load_state_dict(new_state_dict)

    def forward(self, clips):
        batch, n_scale, temporal, channel, height, width = clips.shape
        out_height, out_width = height // 32, width // 32

        f_clips = []
        for i in range(n_scale):
            clip = clips[:, i, :, :, :, :]
            clip = TV.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(clip)

            f_clip = self.swin(clip.transpose(1, 2)).transpose(1, 2)
            f_clip = f_clip.permute(0, 1, 3, 4, 2).view([batch, (temporal + 1) // 2, out_height * out_width, 1024])

            f_clips.append(f_clip)

        f_clips = T.stack(f_clips, dim=1)
        return f_clips


def extract_features(
    video_extractor: T.nn.Module,
    dataset_path: str,
    out_dir: str,
    frames_per_clip: int = 5,
    temporal_scale: List[int] = [1, 2, 3],
    batch_size: int = 5,
):
    video_extractor.to(DEVICE)
    video_extractor.eval()
    dataset = RawVideoDataset(dataset_path, frames_per_clip, temporal_scale)
    dataloader = T.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    os.makedirs(out_dir, exist_ok=True)
    with T.no_grad():
        for batch in tqdm(dataloader):
            video_names, data = batch
            data = data.to(DEVICE)
            result = video_extractor(data)
            for video_name, clip_features in zip(video_names, result):
                with open(os.path.join(out_dir, f'{video_name}.pkl'), 'wb') as f:
                    pickle.dump(clip_features.detach().cpu().numpy(), f)


if __name__ == '__main__':
    video_extractor = VideoExtractor('./pretrained_models/swin_base_patch244_window877_kinetics600_22k.pth')
    # with open('/mnt/hdd/Dataset/MSVD-QA/video-idx-mapping.pkl', 'wb') as f:
    #     pickle.dump(video_dict, f)

    # with open('/mnt/hdd/Dataset/MSVD-QA/video-idx-mapping.pkl', 'rb') as f:
    #     video_idx_mapping = pickle.load(f)
    # video_idx_mapping = build_video_dict('/mnt/hdd/Dataset/MSVD-QA/annotations.txt', start_idx=1)
    # extract_features(
    #     video_extractor,
    #     '/mnt/hdd/Dataset/MSVD-QA/video',
    #     '/mnt/hdd/Dataset/MSVD-QA/video-extracted-kinetics600-no-embedding-no-padding-nframe-5',
    #     video_idx_mapping,
    #     batch_size=32,
    #     frames_per_clip=5,
    # )

    # video_idx_mapping = {f'video{i}': i for i in range(10000)}
    # extract_features(
    #     video_extractor,
    #     '/mnt/hdd/Dataset/MSRVTT-QA/video',
    #     '/mnt/hdd/Dataset/MSRVTT-QA/video-extracted-kinetics600-no-embedding-no-padding-nframe-5',
    #     batch_size=60,
    #     frames_per_clip=5,
    # )
    # extract_features(
    #     video_extractor,
    #     '/mnt/hdd/Dataset/TGIF-QA/gifs',
    #     '/mnt/hdd/Dataset/TGIF-QA/video-extracted-kinetics600-nframe-5',
    #     batch_size=150,
    #     frames_per_clip=5,
    # )
    # extract_features(
    #     video_extractor,
    #     '/mnt/hdd/Dataset/MSVD-QA/video',
    #     '/mnt/hdd/Dataset/MSVD-QA/video-extracted-kinetics600-nframe-5',
    #     batch_size=150,
    #     frames_per_clip=5,
    # )
    extract_features(
        video_extractor,
        '/mnt/hdd/Dataset/MSRVTT-QA/video',
        '/mnt/hdd/Dataset/MSRVTT-QA/video-extracted-kinetics600-nframe-5',
        batch_size=150,
        frames_per_clip=5,
    )
