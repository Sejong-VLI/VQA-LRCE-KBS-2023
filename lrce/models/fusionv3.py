from lrce.lib import *
from lrce.models.embedding import *


class FusionTransformer(T.nn.Module):
    def __init__(self, feature_dim: int = 768, drop_out_rate: float = 0.1) -> None:
        super().__init__()
        decoder_layer = T.nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=12,
            dropout=drop_out_rate,
            dim_feedforward=3072,
            batch_first=True,
            layer_norm_eps=1e-12,
            activation=T.nn.functional.gelu,
        )
        self.transformer = T.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=12)
        self.fusion_layer_norm = T.nn.LayerNorm(feature_dim, eps=1e-12)
        self.dropout = T.nn.Dropout(drop_out_rate)
        self.summarization_token = init_weight((1, 1, feature_dim))
        # num layer 12 dim feedforward 3072 88.05
        # num layer 6 dim feedforward 1536 85 on epoch 2, likely to be less than 88 on later epoch
        # num layer 6 dim feedforward 3072 87
        # num layer 12 dim feedforward 3072 nhead 12 similarity 88.04
        # num layer 12 dim feedforward 3072 nhead 8 87.1

    def forward(
        self,
        video_features: T.tensor,
        text_features: T.tensor,
        texts_attention_mask: T.tensor,
    ):
        """
        Args:
            video_features (T.tensor): (BATCH, TEMPORAL_SCALE, video_seq_len, feature_dim)
            text_features (T.tensor): (BATCH, text_seq_len, feature_dim)
            texts_attention_mask (T.tensor): (BATCH, text_seq_len)
        Returns:
            (T.tensor) (BATCH, num_classes)
        """
        batch, temporal_scale, video_seq_len, _ = video_features.shape

        summarization_token = self.summarization_token.expand([batch, -1, -1])
        for i in range(temporal_scale):
            vidl_features = T.concat([video_features[:, i, :, :], text_features], dim=1)
            immediate_res = self.transformer(summarization_token, vidl_features)
            summarization_token = summarization_token + immediate_res
            summarization_token = self.fusion_layer_norm(summarization_token)
            summarization_token = self.dropout(summarization_token)

        return summarization_token  # (BATCH, 1, FEATURE_DIM)


class FusionVideo(T.nn.Module):
    def __init__(self, feature_dim: int = 768, drop_out_rate: float = 0.1) -> None:
        super().__init__()
        decoder_layer = T.nn.TransformerDecoderLayer(
            d_model=feature_dim,
            nhead=12,
            dropout=drop_out_rate,
            dim_feedforward=3072,
            batch_first=True,
            layer_norm_eps=1e-12,
            activation=T.nn.functional.gelu,
        )
        self.transformer = T.nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=12)
        self.fusion_layer_norm = T.nn.LayerNorm(feature_dim, eps=1e-12)
        self.dropout = T.nn.Dropout(drop_out_rate)
        self.summarization_token = init_weight((1, 1, feature_dim))

    def forward(
        self,
        video_features: T.tensor,
    ):
        """
        Args:
            video_features (T.tensor): (BATCH, TEMPORAL_SCALE, video_seq_len, feature_dim)
        Returns:
            (T.tensor) (BATCH, num_classes)
        """
        batch, temporal_scale, _, _ = video_features.shape

        summarization_token = self.summarization_token.expand([batch, -1, -1])
        for i in range(temporal_scale):
            immediate_res = self.transformer(summarization_token, video_features[:, i, :, :])
            summarization_token = summarization_token + immediate_res
            summarization_token = self.fusion_layer_norm(summarization_token)
            summarization_token = self.dropout(summarization_token)

        return summarization_token  # (BATCH, 1, FEATURE_DIM)


class FusionTransformerSelfAttention(FusionTransformer):
    def __init__(self, feature_dim: int = 768, drop_out_rate: float = 0.1) -> None:
        super().__init__(feature_dim, drop_out_rate)

    def forward(self, video_features: T.tensor, text_features: T.tensor):
        # video_features (BATCH, TEMPORAL_SCALE, SEQ_LEN, feature_dim)
        # text_features (BATCH, SEQ_LEN, feature_dim)
        batch, temporal_scale, _, _ = video_features.shape

        # for i in range(temporal_scale):
        separator_token = self.sep_token.detach().clone().expand(batch, -1, -1).to(self.gpu_id)

        vidl_features = T.concat(
            [
                text_features,
                separator_token,
                video_features[:, 0, :, :],
            ],
            dim=1,
        )
        immediate_res = self.transformer(
            encoder_hidden_states=vidl_features,
            inputs_embeds=vidl_features,
            output_hidden_states=True,
        )
        # print(video_features.shape)
        # print(text_features.shape)
        # print(immediate_res.hidden_states[-1].shape)
        summarization_token = immediate_res.hidden_states[-1][:, 0:1, :]
        # summarization_token += immediate_res.hidden_states[-1]  # skip connection
        # summarization_token = self.fusion_layer_norm(summarization_token)
        # summarization_token = self.dropout(summarization_token)

        return summarization_token  # (BATCH, 1, FEATURE_DIM)


class LRCEOpenEnded(T.nn.Module):
    def __init__(
        self,
        feature_dim: int,  # should be same for video and text
        num_classes: int,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        temporal_scale: List[int] = [1, 2, 3],
        question_seq_len: int = 30,
    ) -> None:
        super().__init__()
        self.feature_dim = feature_dim
        self.video_feature_dim = video_feature_dim

        self.video_pos_embed = VideoPosEmbed(
            feature_dim,
            video_feature_res,
            frame_sample_size,
            clip_size=sum(temporal_scale),
        )
        self.question_pos_embed = TextPosEmbed(question_seq_len, feature_dim)

        if video_feature_dim != feature_dim:
            self.projection_layer = T.nn.Linear(video_feature_dim, feature_dim)

        self.video_dropout = T.nn.Dropout(drop_out_rate)
        self.question_dropout = T.nn.Dropout(drop_out_rate)

        self.fusion_transformer = FusionTransformer(feature_dim, drop_out_rate=drop_out_rate)
        self.final_fc = T.nn.Linear(feature_dim, num_classes)
        #TODO: try direct projection, result: for open-ended works way better to use direct projection
        # self.final_fc = T.nn.Sequential(*[
        #     T.nn.Linear(768, feature_dim * 2),
        #     T.nn.GELU(),
        #     T.nn.Linear(feature_dim * 2, num_classes),
        # ])

    def forward(
        self,
        video_features: T.tensor,
        text_features: T.tensor,
        texts_attention_mask: T.tensor,
    ) -> T.tensor:
        """
        Args:
            video_features (T.tensor): (BATCH, TEMPORAL_SCALE, video_seq_len, feature_dim)
            text_features (T.tensor): (BATCH, question_seq_len, feature_dim)
            texts_attention_mask (T.tensor): (BATCH, question_seq_len)
        Returns:
            (T.tensor) (BATCH, num_classes)
        """

        batch = video_features.shape[0]
        if self.video_feature_dim != self.feature_dim:
            video_features = self.projection_layer(video_features)

        video_features = self.video_pos_embed(video_features)
        text_features = self.question_pos_embed(text_features)

        video_features = self.video_dropout(video_features)
        text_features = self.question_dropout(text_features)

        summarized_features = self.fusion_transformer(video_features, text_features, texts_attention_mask)

        final_out = self.final_fc(summarized_features.squeeze())
        final_out = final_out.view(batch, -1)

        return final_out


class LRCEMultipleChoice(LRCEOpenEnded):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        temporal_scale: List[int] = [1, 2, 3],
        qa_seq_len: int = 40,
    ) -> None:
        super().__init__(
            feature_dim,
            num_classes,
            drop_out_rate,
            video_feature_res,
            video_feature_dim,
            frame_sample_size,
            temporal_scale,
            qa_seq_len,
        )

        # self.final_fc = T.nn.Sequential(*[
        #     T.nn.Linear(feature_dim, feature_dim // 2),
        #     T.nn.GELU(),
        #     T.nn.Linear(feature_dim // 2, num_classes),
        # ])

    def forward(
        self,
        video_features: T.tensor,
        text_features: T.tensor,
        texts_attention_mask: T.tensor,
    ) -> T.tensor:
        """
        Args:
            video_features (T.tensor): (BATCH, TEMPORAL_SCALE, video_seq_len, feature_dim)
            text_features (T.tensor): (BATCH, question_seq_len, feature_dim)
            texts_attention_mask (T.tensor): (BATCH, question_seq_len)
        Returns:
            (T.tensor) (BATCH, num_classes)
        """
        batch, total_mc, _, _ = text_features.shape
        text_features = text_features.flatten(0, 1)
        texts_attention_mask = texts_attention_mask.flatten(0, 1)

        if self.video_feature_dim != self.feature_dim:
            video_features = self.projection_layer(video_features)

        video_features = self.video_pos_embed(video_features)
        text_features = self.question_pos_embed(text_features)

        video_features = self.video_dropout(video_features)
        text_features = self.question_dropout(text_features)

        # video_features (BATCH, TEMPORAL_SCALE, SEQ_LEN, feature_dim)
        # text_features (BATCH*TOTAL_MC, SEQ_LEN, feature_dim)
        video_features = video_features.unsqueeze(1).expand([-1, total_mc, -1, -1, -1]).flatten(0, 1)
        summarized_features = self.fusion_transformer(video_features, text_features, texts_attention_mask)

        final_out = self.final_fc(summarized_features.squeeze())
        final_out = final_out.view(batch, total_mc)

        return final_out


class LRCEMultipleChoiceSim(LRCEOpenEnded):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        temporal_scale: List[int] = [1, 2, 3],
        qa_seq_len: int = 40,
    ) -> None:
        super().__init__(
            feature_dim,
            num_classes,
            drop_out_rate,
            video_feature_res,
            video_feature_dim,
            frame_sample_size,
            temporal_scale,
            qa_seq_len,
        )

        self.text_projection = T.nn.Linear(feature_dim, feature_dim)
        self.fusion_transformer = FusionVideo(feature_dim, drop_out_rate=drop_out_rate)
        self.final_fc = None
        self.cosine_sim = T.nn.CosineSimilarity(dim=1)

    def forward(
        self,
        video_features: T.tensor,
        text_features: T.tensor,
        texts_attention_mask: T.tensor,
    ) -> T.tensor:
        """
        Args:
            video_features (T.tensor): (BATCH, TEMPORAL_SCALE, video_seq_len, feature_dim)
            text_features (T.tensor): (BATCH, num_mc, question_seq_len, feature_dim)
            texts_attention_mask (T.tensor): (BATCH, question_seq_len)
        Returns:
            (T.tensor) (BATCH, num_classes)
        """
        batch, total_mc, _, _ = text_features.shape
        text_features = text_features.flatten(0, 1)
        texts_attention_mask = texts_attention_mask.flatten(0, 1)

        if self.video_feature_dim != self.feature_dim:
            video_features = self.projection_layer(video_features)

        video_features = self.video_pos_embed(video_features)
        text_features = self.question_pos_embed(text_features)

        video_features = self.video_dropout(video_features)
        text_features = self.question_dropout(text_features)

        text_fused = T.mean(text_features, dim=1)
        text_fused = self.text_projection(text_fused)  # (BATCH*num_mc, feature_dim)

        # video_features (BATCH, TEMPORAL_SCALE, SEQ_LEN, feature_dim)
        # video_fused (BATCH*num_mc, FEATURE_DIM)
        video_fused = self.fusion_transformer(video_features).expand([-1, total_mc, -1]).flatten(0, 1)
        # final_out = T.einsum('BA,BA->B', [text_fused, video_fused])
        final_out = self.cosine_sim(text_fused, video_fused)
        final_out = final_out.view(batch, total_mc)

        return final_out


class LRCECount(LRCEOpenEnded):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 1,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        temporal_scale: List[int] = [1, 2, 3],
        question_seq_len: int = 30,
    ) -> None:
        super().__init__(
            feature_dim,
            1,  # always single neuron output
            drop_out_rate,
            video_feature_res,
            video_feature_dim,
            frame_sample_size,
            temporal_scale,
            question_seq_len,
        )
        self.relu = T.nn.ReLU()

    def forward(
        self,
        video_features: T.tensor,
        text_features: T.tensor,
        texts_attention_mask: T.tensor,
    ) -> T.tensor:
        batch = video_features.shape[0]
        out = super().forward(video_features, text_features, texts_attention_mask)
        out = self.relu(out.view(batch))
        return out
        # if self.training:
        #     return out.view(batch)
        # else:
        #     return T.round(out).view(batch)