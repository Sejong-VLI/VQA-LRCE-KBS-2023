from lrce.lib import *


def init_weight(size: Tuple):
    weight = torch.empty(size, requires_grad=True)
    weight = torch.nn.init.xavier_normal_(weight)
    return T.nn.Parameter(weight, requires_grad=True)


class TextPosEmbed(T.nn.Module):
    def __init__(self, seq_len: int, feature_dim: int) -> None:
        super().__init__()
        self.emb_cls = init_weight((1, 1, feature_dim))
        self.emb_pos = init_weight((1, 1 + seq_len, feature_dim))
        self.layer_norm = T.nn.LayerNorm(feature_dim, eps=1e-12)

    def forward(self, text_features: T.tensor) -> T.tensor:
        # text_features (BATCH, SEQ_LEN, FEATURES_DIM)
        batch, _, _ = text_features.shape
        text_features = T.cat([self.emb_cls.expand([batch, -1, -1]), text_features], dim=1)
        text_features = text_features + self.emb_pos.expand([batch, -1, -1])
        text_features = self.layer_norm(text_features)
        return text_features


class VideoPosEmbed(T.nn.Module):
    def __init__(
            self,
            feature_dim: int,
            video_feature_res: Iterable[int] = (7, 7),
            frame_sample_size: int = 5,
            clip_size: int = 6,
    ) -> None:
        super().__init__()
        self.emb_cls = init_weight((1, 1, 1, 1, feature_dim))
        self.emb_pos = init_weight((
            1,
            1,
            1,
            1 + video_feature_res[0] * video_feature_res[1],
            feature_dim,
        ))
        self.emb_len = init_weight((1, 1, (frame_sample_size + 1) // 2, 1, feature_dim))
        self.emb_clip = init_weight((1, clip_size, 1, 1, feature_dim))
        self.layer_norm = T.nn.LayerNorm(feature_dim, eps=1e-12)

    def forward(self, video_features: T.tensor) -> T.tensor:
        # video_features (BATCH, TEMPORAL_SCALE, TEMPORAL, FRAME_MUL, FEATURE_DIM)
        batch, temporal_scale, temporal, frame_mul, _ = video_features.shape

        video_features = T.cat(
            [
                self.emb_cls.expand([batch, temporal_scale, temporal, -1, -1]),
                video_features,
            ],
            dim=3,
        )
        video_features = video_features + self.emb_pos.expand([batch, temporal_scale, temporal, -1, -1])
        video_features = video_features + self.emb_len.expand([batch, temporal_scale, -1, 1 + frame_mul, -1])
        video_features = video_features + self.emb_clip.expand([batch, -1, temporal, 1 + frame_mul, -1])
        video_features = self.layer_norm(video_features)
        video_features = video_features.view([batch, temporal_scale, temporal * (1 + frame_mul), -1])
        return video_features