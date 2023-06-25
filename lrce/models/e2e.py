from lrce.lib import *
from lrce.feature_extractor.video import VideoExtractor
from lrce.feature_extractor.text import TextExtractor
from lrce.models.fusionv3 import LRCEOpenEnded, LRCEMultipleChoice, LRCECount, LRCEMultipleChoiceSim


class E2EBase(T.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        assert os.path.exists('./pretrained_models/swin_base_patch244_window877_kinetics600_22k.pth')

        self.text_extractor = TextExtractor()
        self.video_extractor = VideoExtractor('./pretrained_models/swin_base_patch244_window877_kinetics600_22k.pth')

    def extract_text_features(self, texts: T.tensor, attention_mask: T.tensor, texts_type_ids: T.tensor):
        return self.text_extractor(texts, attention_mask, texts_type_ids)

    def extract_video_features(self, video_clips: T.tensor):
        return self.video_extractor(video_clips)

    def forward(self, video_clips: T.tensor, texts: T.tensor, texts_attention_mask: T.tensor, texts_type_ids: T.tensor):
        video_features = self.extract_video_features(video_clips)
        texts_features = self.extract_text_features(texts, texts_attention_mask, texts_type_ids)
        return self.fusion_model(video_features, texts_features, texts_attention_mask)


class E2EOpenEnded(E2EBase):
    def __init__(
        self,
        feature_dim: int,  # should be same for video and text
        num_classes: int,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        temporal_scale: List[int] = [1, 2, 3],
        text_seq_len: int = 30,
    ) -> None:
        super().__init__()
        self.fusion_model = LRCEOpenEnded(
            feature_dim,
            num_classes,
            drop_out_rate,
            video_feature_res,
            video_feature_dim,
            frame_sample_size,
            temporal_scale,
            text_seq_len,
        )


class E2EMultipleChoice(E2EBase):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        temporal_scale: List[int] = [1, 2, 3],
        text_seq_len: int = 40,
    ) -> None:
        super().__init__()
        self.fusion_model = LRCEMultipleChoice(
            feature_dim,
            num_classes,
            drop_out_rate,
            video_feature_res,
            video_feature_dim,
            frame_sample_size,
            temporal_scale,
            text_seq_len,
        )

    def extract_text_features(self, texts: T.tensor, attention_mask: T.tensor, texts_type_ids: T.tensor):
        batch_size, total_choice, seq_len = texts.shape
        # (BATCH_SIZE*total_choice, seq_len, feature_dim)
        out = self.text_extractor(texts.flatten(0, 1), attention_mask.flatten(0, 1), texts_type_ids.flatten(0, 1))
        return out.view(batch_size, total_choice, seq_len, -1)


class E2ECount(E2EBase):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 1,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        temporal_scale: List[int] = [1, 2, 3],
        text_seq_len: int = 30,
    ) -> None:
        super().__init__()
        self.fusion_model = LRCECount(
            feature_dim,
            num_classes,
            drop_out_rate,
            video_feature_res,
            video_feature_dim,
            frame_sample_size,
            temporal_scale,
            text_seq_len,
        )
