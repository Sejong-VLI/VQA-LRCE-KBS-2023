from lrce.lib import *
from lrce.models.embedding import *
from lrce.models.fusion_transformer import *

#DEPRECATED DO NOT USE


class FusionTransformerNoSelfAttention(T.nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 12,
        expansion_size: int = 4,
        drop_out_rate: float = 0.4,
        num_layers: int = 12,
    ) -> None:
        super().__init__()
        self.layers = T.nn.ModuleList(
            [EncoderBlock(feature_dim, num_heads, expansion_size, drop_out_rate) for _ in range(num_layers)])
        self.fusion_layer_norm = T.nn.LayerNorm(feature_dim)
        self.dropout = T.nn.Dropout(drop_out_rate)
        self.cls_token, self.sep_token = calculate_special_token_embedding()

    def do_fusion(self, query: T.tensor, key: T.tensor, value: T.tensor):
        out = query
        for layer in self.layers:
            out = layer(out, key, value)
        return out

    def forward(self, video_features: T.tensor, text_features: T.tensor):
        batch, temporal_scale, _, _ = video_features.shape

        summarization_token = self.cls_token.detach().clone().expand(batch, -1, -1).to(DEVICE)
        separator_token = self.sep_token.detach().clone().expand(batch, -1, -1).to(DEVICE)

        for i in range(temporal_scale):
            vidl_features = T.concat(
                [
                    video_features[:, i, :, :],
                    separator_token,
                    text_features,
                ],
                dim=1,
            )
            immediate_res = self.do_fusion(summarization_token, vidl_features, vidl_features)

            summarization_token = self.fusion_layer_norm(summarization_token + immediate_res)
            summarization_token = self.dropout(summarization_token)

        return summarization_token  # (BATCH, 1, FEATURE_DIM)


class FusionTransformerSelfAttention(FusionTransformerNoSelfAttention):
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 12,
        expansion_size: int = 4,
        drop_out_rate: float = 0.4,
        num_layers: int = 12,
    ) -> None:
        super().__init__(feature_dim, num_heads, expansion_size, drop_out_rate, num_layers)

    def forward(self, video_features: T.tensor, text_features: T.tensor):
        batch, _, vid_seq_len, _ = video_features.shape

        summarization_token = self.cls_token.detach().clone().expand(batch, -1, -1).to(DEVICE)

        vidl_features = T.concat(
            [
                video_features[:, 0, :, :],
                summarization_token,
                text_features,
            ],
            dim=1,
        )
        raw_res = self.do_fusion(vidl_features, vidl_features, vidl_features)
        res = raw_res[:, vid_seq_len:vid_seq_len + 1, :]
        res = self.dropout(res)

        return res  # (BATCH, 1, FEATURE_DIM)


#TODO: try using summarization token as cls in multiscale transformer
#TODO: try disable dropout in fusion
#TODO: try init embedding with sinusoidal
#TODO: try to use kinetic600 videoswin (nice result)
#TODO: try independent cls each scale then integrate in multiscale transformer (done, qutie good result)
#TODO: try to separate CLS token for fusion and final answer aggregation (useless)
#TODO: try removing multiscale transformer altogether (done)
#TODO: try different MLP head config (done)
#TODO: try freezing embeding layer (done)
class FusionTransformerWithLib(T.nn.Module):
    def __init__(self, feature_dim: int, drop_out_rate: float) -> None:
        super().__init__()
        config = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(
            transformers.BertConfig(),
            transformers.BertConfig(),
        )
        self.transformer = transformers.EncoderDecoderModel(config).get_decoder()
        self.fusion_layer_norm = T.nn.LayerNorm(feature_dim)
        self.dropout = T.nn.Dropout(drop_out_rate)
        self.cls_token, self.sep_token = calculate_special_token_embedding()

    def forward(self, video_features: T.tensor, text_features: T.tensor):
        batch, temporal_scale, _, _ = video_features.shape

        # summarization_token = self.cls_token.detach().clone().expand(batch, -1, -1).to(DEVICE)
        summarization_token = text_features[:, 0:1, :]
        separator_token = self.sep_token.detach().clone().expand(batch, -1, -1).to(DEVICE)

        for i in range(temporal_scale):
            vidl_features = T.concat(
                [
                    video_features[:, i, :, :],
                    separator_token,
                    text_features,
                ],
                dim=1,
            )
            immediate_res = self.transformer(
                encoder_hidden_states=vidl_features,
                inputs_embeds=summarization_token,
                output_hidden_states=True,
            )

            summarization_token = summarization_token + immediate_res.hidden_states[-1]
            summarization_token = self.fusion_layer_norm(summarization_token)
            summarization_token = self.dropout(summarization_token)
            # TODO: check if this affect performance (yes, 1% diff)

        return summarization_token  # (BATCH, 1, FEATURE_DIM)


class FusionTransformerWithLibSelfAtt(T.nn.Module):
    def __init__(self, feature_dim: int, drop_out_rate: float) -> None:
        super().__init__()
        config = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(
            transformers.BertConfig(),
            transformers.BertConfig(),
        )
        self.transformer = transformers.EncoderDecoderModel(config).get_decoder()
        self.fusion_layer_norm = T.nn.LayerNorm(feature_dim)
        self.dropout = T.nn.Dropout(drop_out_rate)
        self.cls_token, self.sep_token = calculate_special_token_embedding()

    def forward(self, video_features: T.tensor, text_features: T.tensor):
        batch, temporal_scale, vid_seq_len, _ = video_features.shape

        summarization_token = self.cls_token.detach().clone().expand(batch, -1, -1).to(DEVICE)
        # separator_token = self.sep_token.detach().clone().expand(batch, -1, -1).to(DEVICE)

        # for i in range(temporal_scale):
        vidl_features = T.concat(
            [
                video_features[:, 0, :, :],
                summarization_token,
                text_features,
            ],
            dim=1,
        )
        res = self.transformer(
            encoder_hidden_states=vidl_features,
            inputs_embeds=vidl_features,
            output_hidden_states=True,
        ).hidden_states[-1][:, vid_seq_len:vid_seq_len + 1, :]

        # summarization_token = summarization_token + immediate_res.hidden_states[-1]
        # summarization_token = self.fusion_layer_norm(summarization_token)
        # summarization_token = self.dropout(summarization_token)
        # TODO: check if this affect performance (yes, 1% diff)

        return res  # (BATCH, 1, FEATURE_DIM)


class LRCE(T.nn.Module):
    def __init__(
        self,
        feature_dim: int,  # should be same for video and text
        num_classes: int,
        drop_out_rate: float = 0.1,
        question_seq_len: int = 30,
    ) -> None:
        super().__init__()

        self.fusion_transformer = FusionTransformerWithLib(feature_dim, drop_out_rate)
        # self.fusion_transformer = FusionTransformerWithLibSelfAtt(feature_dim, drop_out_rate)
        # self.fusion_transformer = FusionTransformerNoSelfAttention(feature_dim, drop_out_rate=drop_out_rate)

        # simple projection or with expansion
        self.final_fc = T.nn.Linear(feature_dim, num_classes)
        # self.final_fc = T.nn.Sequential(*[
        #     T.nn.Linear(768, feature_dim * 2),
        #     T.nn.GELU(),
        #     T.nn.Linear(feature_dim * 2, num_classes),
        # ])

        self.question_pos_embed = TextPosEmbed(question_seq_len, feature_dim)

        self.video_dropout = T.nn.Dropout(drop_out_rate)
        self.question_dropout = T.nn.Dropout(drop_out_rate)

    def get_l2_reg(self):
        l2_reg = T.tensor(0, requires_grad=True, dtype=T.float32)
        for param in self.parameters():
            if param.requires_grad:
                l2_reg = l2_reg + param.norm(2)
        return l2_reg

    def forward(
        self,
        video_feature_tokens: T.tensor,
        question_feature_tokens: T.tensor,
    ) -> T.tensor:
        """
        Args:
            video_feature_tokens (T.tensor): (BATCH, TEMPORAL_SCALE, video_seq_len, feature_dim)
            question_feature_tokens (T.tensor): (BATCH, question_seq_len, feature_dim)
        Returns:
            (T.tensor) (BATCH, num_classes)
        """

        batch = video_feature_tokens.shape[0]

        # Embedding for text only because video embed has been added from the feature extractor
        question_feature_tokens = self.question_pos_embed(question_feature_tokens)

        video_feature_tokens = self.video_dropout(video_feature_tokens)
        question_feature_tokens = self.question_dropout(question_feature_tokens)

        summarized_features = self.fusion_transformer(video_feature_tokens, question_feature_tokens)

        final_out = self.final_fc(summarized_features.squeeze())
        final_out = final_out.view(batch, -1)

        return final_out


class LRCEOpenEnded(LRCE):
    def __init__(
        self,
        feature_dim: int,  # should be same for video and text
        num_classes: int,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        question_seq_len: int = 30,
    ) -> None:
        super().__init__(
            feature_dim=feature_dim,
            num_classes=num_classes,
            drop_out_rate=drop_out_rate,
            question_seq_len=question_seq_len,
        )
        self.feature_dim = feature_dim
        self.video_feature_dim = video_feature_dim
        self.video_pos_embed = VideoPosEmbed(feature_dim, video_feature_res, frame_sample_size)

        if video_feature_dim != feature_dim:
            self.projection_layer = T.nn.Linear(video_feature_dim, feature_dim)

    def forward(
        self,
        video_feature_tokens: T.tensor,
        question_feature_tokens: T.tensor,
    ) -> T.tensor:
        """
        Args:
            video_feature_tokens (T.tensor): (BATCH, TEMPORAL_SCALE, TEMPORAL, H*W, feature_dim)
            question_feature_tokens (T.tensor): (BATCH, question_seq_len, feature_dim)
        Returns:
            (T.tensor) (BATCH, num_classes)
        """
        # Match feature dimension
        if self.video_feature_dim != self.feature_dim:
            video_feature_tokens = self.projection_layer(video_feature_tokens)

        video_feature_tokens = self.video_pos_embed(video_feature_tokens)

        return super().forward(video_feature_tokens, question_feature_tokens)


class LRCEMultipleChoice(LRCEOpenEnded):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        qa_seq_len: int = 40,
    ) -> None:
        super().__init__(
            feature_dim,
            num_classes,
            drop_out_rate,
            video_feature_res,
            video_feature_dim,
            frame_sample_size,
            qa_seq_len,
        )

        self.final_fc = T.nn.Sequential(*[
            T.nn.Linear(feature_dim, feature_dim),
            T.nn.GELU(),
            T.nn.Linear(feature_dim, num_classes),
        ])

    def forward(
        self,
        video_feature_tokens: T.tensor,
        qa_feature_tokens: T.tensor,
    ) -> T.tensor:
        batch, total_mc, _, _ = qa_feature_tokens.shape

        out = T.empty(batch, total_mc, 1).to(DEVICE)
        for i in range(total_mc):
            out[:, i, :] = super().forward(video_feature_tokens, qa_feature_tokens[:, i, :, :])
        out = out.squeeze()
        return out.view(batch, -1)


class LRCEMultipleChoiceV2(LRCEOpenEnded):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        qa_seq_len: int = 40,
    ) -> None:
        super().__init__(
            feature_dim,
            num_classes,
            drop_out_rate,
            video_feature_res,
            video_feature_dim,
            frame_sample_size,
            qa_seq_len,
        )
        self.final_fc = T.nn.Sequential(*[
            T.nn.Linear(feature_dim, feature_dim),
            T.nn.GELU(),
            T.nn.Linear(feature_dim, num_classes),
        ])

    def forward(self, video_feature_tokens: T.tensor, qa_feature_tokens: T.tensor):
        batch, total_mc, _, feature_dim = qa_feature_tokens.shape

        if self.video_feature_dim != self.feature_dim:
            video_feature_tokens = self.projection_layer(video_feature_tokens)

        video_feature_tokens = self.video_pos_embed(video_feature_tokens)
        video_feature_tokens = self.video_dropout(video_feature_tokens)

        out = T.empty(batch, total_mc, feature_dim).to(DEVICE)
        for i in range(total_mc):
            choice_feature_tokens = qa_feature_tokens[:, i, :, :]
            choice_feature_tokens = self.question_pos_embed(choice_feature_tokens)
            choice_feature_tokens = self.question_dropout(choice_feature_tokens)
            out[:, i, :] = self.fusion_transformer(video_feature_tokens, choice_feature_tokens).squeeze()

        final_out = self.final_fc(out)
        final_out = final_out.view(batch, -1)

        return final_out


class LRCEMultipleChoiceV3(LRCEMultipleChoice):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        qa_seq_len: int = 40,
    ) -> None:
        super().__init__(
            feature_dim,
            num_classes,
            drop_out_rate,
            video_feature_res,
            video_feature_dim,
            frame_sample_size,
            qa_seq_len,
        )
        self.fusion_transformer = FusionTransformerNoSelfAttention(feature_dim, drop_out_rate=drop_out_rate)


class LRCECount(LRCEOpenEnded):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 1,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        question_seq_len: int = 30,
    ) -> None:
        super().__init__(
            feature_dim,
            num_classes,
            drop_out_rate,
            video_feature_res,
            video_feature_dim,
            frame_sample_size,
            question_seq_len,
        )

    def forward(self, video_feature_tokens: T.tensor, question_feature_tokens: T.tensor) -> T.tensor:
        out = super().forward(video_feature_tokens, question_feature_tokens)
        batch = video_feature_tokens.shape[0]

        if self.training:
            return out.view(batch)
        else:
            return T.round(out).view(batch)