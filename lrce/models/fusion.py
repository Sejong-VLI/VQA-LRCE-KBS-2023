from lrce.lib import *

#DEPRECATED DO NOT USE


class LRCE(T.nn.Module):
    def __init__(
        self,
        feature_dim: int,  # should be same for video and text
        num_classes: int,
        drop_out_rate: float = 0.1,
        question_seq_len: int = 30,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.question_seq_len = question_seq_len
        self.drop_out_rate = drop_out_rate

        config = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(
            transformers.BertConfig(),
            transformers.BertConfig(),
        )

        self.fusion_transformer = transformers.EncoderDecoderModel(config).get_decoder()
        # self.multiscale_transformer = transformers.EncoderDecoderModel(config).get_decoder()

        # simple projection or with expansion
        self.final_fc = T.nn.Linear(feature_dim, num_classes)
        # self.final_fc = T.nn.Sequential(*[
        #     T.nn.Linear(768, feature_dim * 2),
        #     T.nn.GELU(),
        #     T.nn.Linear(feature_dim * 2, num_classes),
        # ])

        self.video_dropout = T.nn.Dropout(drop_out_rate)
        self.question_dropout = T.nn.Dropout(drop_out_rate)
        self.fusion_dropout = T.nn.Dropout(drop_out_rate)

        self.question_emb_cls = T.nn.Parameter(0.02 * T.randn(1, 1, feature_dim), requires_grad=True)
        self.question_emb_pos = T.nn.Parameter(0.02 * T.randn(1, 1 + question_seq_len, feature_dim), requires_grad=True)
        self.question_layer_norm = T.nn.LayerNorm(feature_dim)
        self.fusion_layer_norm = T.nn.LayerNorm(feature_dim)

        self.calculate_special_token_embedding()

    def calculate_special_token_embedding(self):
        bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        embedding = bert.embeddings
        embedding.eval()
        tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

        with T.no_grad():
            cls_token = torch.LongTensor(tokenizer.encode(CLS_TOKEN, add_special_tokens=False))
            sep_token = torch.LongTensor(tokenizer.encode(SEP_TOKEN, add_special_tokens=False))
            cls_token = cls_token.reshape(1, -1)
            sep_token = cls_token.reshape(1, -1)
            special_token = T.concat([cls_token, sep_token], dim=1)
            embedded_special_token = embedding(special_token)  # BATCH, 2, feature_dim
        self.cls_token = embedded_special_token[:, 0:1, :]
        self.sep_token = embedded_special_token[:, 1:2, :]

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

        batch, temporal_scale, _, feature_dim = video_feature_tokens.shape

        # Embedding for text only because video embed has been added from the feature extractor
        question_feature_tokens = T.cat([self.question_emb_cls.expand([batch, -1, -1]), question_feature_tokens], dim=1)
        question_feature_tokens += self.question_emb_pos.expand([batch, -1, -1])
        question_feature_tokens = self.question_layer_norm(question_feature_tokens)

        video_feature_tokens = self.video_dropout(video_feature_tokens)
        question_feature_tokens = self.question_dropout(question_feature_tokens)

        # summarization_arr = T.empty(batch, temporal_scale, feature_dim).to(DEVICE)
        summarization_token = self.cls_token.detach().clone().expand(batch, -1, -1).to(DEVICE)
        # print(summarization_token.shape)
        separator_token = self.sep_token.detach().clone().expand(batch, -1, -1).to(DEVICE)

        for i in range(temporal_scale):
            vidl_features = T.concat(
                [
                    video_feature_tokens[:, i, :, :],
                    separator_token,
                    question_feature_tokens,
                ],
                dim=1,
            )
            immediate_res = self.fusion_transformer(
                encoder_hidden_states=vidl_features,
                inputs_embeds=summarization_token,
                output_hidden_states=True,
            )

            summarization_token = summarization_token + immediate_res.hidden_states[-1]
            summarization_token = self.fusion_layer_norm(summarization_token)
            summarization_token = self.fusion_dropout(summarization_token)
            # summarization_arr[:, i, :] = immediate_res.hidden_states[-1].squeeze()

        # multiscale_result = self.multiscale_transformer(
        #     encoder_hidden_states=summarization_arr,
        #     inputs_embeds=summarization_token,
        #     output_hidden_states=True,
        # )

        #TODO: try using summarization token as cls in multiscale transformer
        #TODO: try disable dropout in fusion
        #TODO: try init embedding with sinusoidal
        #TODO: try to use kinetic600 videoswin (nice result)
        #TODO: try independent cls each scale then integrate in multiscale transformer (done, qutie good result)
        #TODO: try to separate CLS token for fusion and final answer aggregation (useless)
        #TODO: try removing multiscale transformer altogether (done)
        #TODO: try different MLP head config (done)
        #TODO: try freezing embeding layer (done)

        # multiscale_result = multiscale_result.hidden_states[-1].squeeze()  # (BATCH_SIZE, FEATURES_DIM)
        # multiscale_result = self.fusion_dropout(multiscale_result)
        # final_out = self.final_fc(multiscale_result)

        final_out = self.final_fc(summarization_token.squeeze())
        final_out = final_out.view(batch, -1)

        return final_out


class FullLRCEOpenEnded(LRCE):
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
        self.video_feature_dim = video_feature_dim

        self.vid_emb_cls = T.nn.Parameter(0.02 * T.randn(1, 1, 1, 1, feature_dim))
        self.vid_emb_pos = T.nn.Parameter(0.02 * T.randn(
            1,
            1,
            1,
            1 + video_feature_res[0] * video_feature_res[1],
            feature_dim,
        ))
        self.vid_emb_len = T.nn.Parameter(0.02 * T.randn(1, 1, (frame_sample_size + 1) // 2, 1, feature_dim))
        self.vid_layer_norm = T.nn.LayerNorm(feature_dim)

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
            cls_token (T.tensor): (BATCH, 1)
            sep_token (T.tensor): (BATCH, 1)
        Returns:
            (T.tensor) (BATCH, num_classes)
        """

        batch, temporal_scale, temporal, frame_mul, _ = video_feature_tokens.shape

        # Match feature dimension
        if self.video_feature_dim != self.feature_dim:
            video_feature_tokens = self.projection_layer(video_feature_tokens)

        # Embedding for video
        video_feature_tokens = T.cat(
            [self.vid_emb_cls.expand([batch, temporal_scale, temporal, -1, -1]), video_feature_tokens],
            dim=3,
        )
        video_feature_tokens += self.vid_emb_pos.expand([batch, temporal_scale, temporal, -1, -1])
        video_feature_tokens += self.vid_emb_len.expand([batch, temporal_scale, -1, 1 + frame_mul, -1])
        video_feature_tokens = self.vid_layer_norm(video_feature_tokens)
        video_feature_tokens = video_feature_tokens.view([batch, temporal_scale, temporal * (1 + frame_mul), -1])

        return super().forward(video_feature_tokens, question_feature_tokens)


class FullLRCEMultipleChoice(FullLRCEOpenEnded):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        question_seq_len: int = 30,
        answer_seq_len: int = 12,
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
        self.answer_dropout = T.nn.Dropout(drop_out_rate)
        self.answer_emb_cls = T.nn.Parameter(0.02 * T.randn(1, 1, 1, feature_dim), requires_grad=True)
        self.answer_emb_pos = T.nn.Parameter(0.02 * T.randn(1, 1, 1 + answer_seq_len, feature_dim), requires_grad=True)
        self.answer_layer_norm = T.nn.LayerNorm(feature_dim)

        self.final_fc = T.nn.Sequential(*[
            T.nn.Linear(2 * feature_dim, feature_dim),
            T.nn.GELU(),
            T.nn.Linear(feature_dim, num_classes),
        ])

    def forward(
        self,
        video_feature_tokens: T.tensor,
        q_feature_tokens: T.tensor,
        a_feature_tokens: T.tensor,
    ) -> T.tensor:
        batch, temporal_scale, temporal, frame_mul, _ = video_feature_tokens.shape
        _, total_mc, _, _ = a_feature_tokens.shape

        # Match feature dimension
        if self.video_feature_dim != self.feature_dim:
            video_feature_tokens = self.projection_layer(video_feature_tokens)

        # Embedding for video
        video_feature_tokens = T.cat(
            [self.vid_emb_cls.expand([batch, temporal_scale, temporal, -1, -1]), video_feature_tokens],
            dim=3,
        )
        video_feature_tokens += self.vid_emb_pos.expand([batch, temporal_scale, temporal, -1, -1])
        video_feature_tokens += self.vid_emb_len.expand([batch, temporal_scale, -1, 1 + frame_mul, -1])
        video_feature_tokens = self.vid_layer_norm(video_feature_tokens)
        video_feature_tokens = video_feature_tokens.view([batch, temporal_scale, temporal * (1 + frame_mul), -1])

        # Embedding for question
        q_feature_tokens = T.cat([self.question_emb_cls.expand([batch, -1, -1]), q_feature_tokens], dim=1)
        q_feature_tokens += self.question_emb_pos.expand([batch, -1, -1])
        q_feature_tokens = self.question_layer_norm(q_feature_tokens)

        # # Embedding for answer
        a_feature_tokens = T.cat([self.answer_emb_cls.expand([batch, total_mc, -1, -1]), a_feature_tokens], dim=2)
        a_feature_tokens += self.answer_emb_pos.expand([batch, total_mc, -1, -1])
        a_feature_tokens = self.answer_layer_norm(a_feature_tokens)

        video_feature_tokens = self.video_dropout(video_feature_tokens)
        q_feature_tokens = self.question_dropout(q_feature_tokens)
        a_feature_tokens = self.answer_dropout(a_feature_tokens)

        summarization_token = self.cls_token.detach().clone().expand(batch, -1, -1).to(DEVICE)
        separator_token = self.sep_token.detach().clone().expand(batch, -1, -1).to(DEVICE)
        for i in range(temporal_scale):
            vidl_features = T.concat(
                [
                    video_feature_tokens[:, i, :, :],
                    separator_token,
                    q_feature_tokens,
                ],
                dim=1,
            )
            immediate_res = self.fusion_transformer(
                encoder_hidden_states=vidl_features,
                inputs_embeds=summarization_token,
                output_hidden_states=True,
            )

            summarization_token = summarization_token + immediate_res.hidden_states[-1]
            summarization_token = self.fusion_layer_norm(summarization_token)
            summarization_token = self.fusion_dropout(summarization_token)

        question_out = summarization_token  #(BATCH, 1, FEATURE_DIM)

        final_out = torch.empty(batch, total_mc).to(DEVICE)
        for k in range(total_mc):
            a_features = a_feature_tokens[:, k, :, :]

            # # Embedding for answer
            # a_features = T.cat([self.question_emb_cls.expand([batch, -1, -1]), a_features], dim=2)
            # a_features += self.question_emb_pos.expand([batch, -1, -1])
            # a_features = self.question_layer_norm(a_features)

            summarization_token = self.cls_token.detach().clone().expand(batch, -1, -1).to(DEVICE)
            separator_token = self.sep_token.detach().clone().expand(batch, -1, -1).to(DEVICE)

            for i in range(temporal_scale):
                vidl_features = T.concat(
                    [
                        video_feature_tokens[:, i, :, :],
                        separator_token,
                        a_features,
                    ],
                    dim=1,
                )
                immediate_res = self.fusion_transformer(
                    encoder_hidden_states=vidl_features,
                    inputs_embeds=summarization_token,
                    output_hidden_states=True,
                )

                summarization_token = summarization_token + immediate_res.hidden_states[-1]
                summarization_token = self.fusion_layer_norm(summarization_token)
                summarization_token = self.fusion_dropout(summarization_token)

            # (BATCH, 2*FEATURE_DIM)
            qa_features = T.concat([question_out.squeeze(), summarization_token.squeeze()], dim=1)
            final_out[:, k] = self.final_fc(qa_features).view(batch)

        return final_out  # (BATCH, TOTAL_MC)


class FullLRCEMultipleChoiceV2(FullLRCEOpenEnded):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        drop_out_rate: float = 0.1,
        video_feature_res: Iterable[int] = (7, 7),
        video_feature_dim: int = 768,
        frame_sample_size: int = 5,
        qa_seq_len: int = 30,
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

        # self.final_fc = T.nn.Sequential(*[
        #     T.nn.Linear(2 * feature_dim, feature_dim),
        #     T.nn.GELU(),
        #     T.nn.Linear(feature_dim, num_classes),
        # ])

    def forward(
        self,
        video_feature_tokens: T.tensor,
        qa_feature_tokens: T.tensor,
    ) -> T.tensor:
        return super().forward(video_feature_tokens, qa_feature_tokens)


class FullLRCECount(FullLRCEOpenEnded):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
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


class LRCEWithSim(T.nn.Module):
    def __init__(
            self,
            feature_dim: int,  # should be same for video and text
            answer_dict: Dict,
            drop_out_rate: float = 0.1,
            video_seq_len: int = 250,
            question_seq_len: int = 30,
            video_feature_res: Iterable[int] = (7, 7),
            frame_sample_size: int = 5,
            final_hidden_dim: int = 50,
    ) -> None:
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = len(answer_dict)
        self.video_seq_len = video_seq_len
        self.question_seq_len = question_seq_len
        self.drop_out_rate = drop_out_rate
        self.answer_dict = answer_dict
        self.frame_sample_size = frame_sample_size

        config = transformers.EncoderDecoderConfig.from_encoder_decoder_configs(
            transformers.BertConfig(),
            transformers.BertConfig(),
        )

        self.fusion_transformer = transformers.EncoderDecoderModel(config).get_decoder()
        self.fc_fusion = T.nn.Linear(feature_dim, final_hidden_dim)
        self.fc_text = T.nn.Linear(feature_dim, final_hidden_dim)
        self.fc_answer = T.nn.Linear(feature_dim, final_hidden_dim)

        self.bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')
        self.video_dropout = T.nn.Dropout(drop_out_rate)
        self.question_dropout = T.nn.Dropout(drop_out_rate)
        self.fusion_dropout = T.nn.Dropout(drop_out_rate)

        self.question_emb_cls = T.nn.Parameter(0.02 * T.randn(1, 1, feature_dim), requires_grad=True)
        self.question_emb_pos = T.nn.Parameter(0.02 * T.randn(1, 1 + question_seq_len, feature_dim), requires_grad=True)
        self.question_layer_norm = T.nn.LayerNorm(feature_dim)

        self.vid_emb_cls = T.nn.Parameter(0.02 * T.randn(1, 1, 1, 1, feature_dim))
        self.vid_emb_pos = T.nn.Parameter(0.02 * T.randn(
            1,
            1,
            1,
            1 + video_feature_res[0] * video_feature_res[1],
            feature_dim,
        ))
        self.vid_emb_len = T.nn.Parameter(0.02 * T.randn(1, 1, frame_sample_size, 1, feature_dim))
        self.vid_layer_norm = T.nn.LayerNorm(feature_dim)

        self.fusion_layer_norm = T.nn.LayerNorm(feature_dim)

        self.freeze_embedding()
        self.calculate_answer_features()

    def freeze_embedding(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def calculate_answer_features(self):
        with T.no_grad():
            answer_set = list(self.answer_dict.keys())
            answer_tensor = []
            for answer in answer_set:
                tokenized_ans = self.tokenizer.encode(
                    answer,
                    add_special_tokens=False,
                    return_tensors='pt',
                    padding='max_length',
                    max_length=5,
                )
                answer_tensor.append(tokenized_ans)
            answer_tensor = T.concat(answer_tensor, dim=0)
            bert_features = self.bert(answer_tensor)

            self.answer_features = T.mean(bert_features.last_hidden_state, dim=1).to(DEVICE)

    def get_special_token_embedding(self, cls_token: T.tensor, sep_token: T.tensor):
        with T.no_grad():
            special_token = T.concat([cls_token, sep_token], dim=1)
            embedded_special_token = self.bert.embeddings(special_token)  # BATCH, 2, feature_dim
            return embedded_special_token[:, 0:1], embedded_special_token[:, 1:2]

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
        cls_token: T.tensor,
        sep_token: T.tensor,
    ) -> T.tensor:
        """
        Args:
            video_feature_tokens (T.tensor): (BATCH, TEMPORAL_SCALE, TEMPORAL, H*W, feature_dim)
            question_feature_tokens (T.tensor): (BATCH, question_seq_len, feature_dim)
            cls_token (T.tensor): (BATCH, 1)
            sep_token (T.tensor): (BATCH, 1)
        Returns:
            (T.tensor) (BATCH, num_classes)
        """

        batch, temporal_scale, temporal, frame_mul, _ = video_feature_tokens.shape

        # Embedding for video
        video_feature_tokens = T.cat(
            [self.vid_emb_cls.expand([batch, temporal_scale, temporal, -1, -1]), video_feature_tokens],
            dim=3,
        )
        video_feature_tokens += self.vid_emb_pos.expand([batch, temporal_scale, temporal, -1, -1])
        video_feature_tokens += self.vid_emb_len.expand([batch, temporal_scale, -1, 1 + frame_mul, -1])
        video_feature_tokens = self.vid_layer_norm(video_feature_tokens)
        video_feature_tokens = video_feature_tokens.view([batch, temporal_scale, temporal * (1 + frame_mul), -1])

        # Embedding for text only because video embed has been added from the feature extractor
        question_feature_tokens = T.cat([self.question_emb_cls.expand([batch, -1, -1]), question_feature_tokens], dim=1)
        question_feature_tokens += self.question_emb_pos.expand([batch, -1, -1])
        question_feature_tokens = self.question_layer_norm(question_feature_tokens)

        # (BATCH, TEMPORAL_SCALE, video_seq_len, feature_dim)
        video_feature_tokens = self.video_dropout(video_feature_tokens)
        # (BATCH, question_seq_len, feature_dim)
        question_feature_tokens = self.question_dropout(question_feature_tokens)

        # fuse text and video features
        fuse_attention = T.einsum('BTVD,BQD->BTVQ', [video_feature_tokens, question_feature_tokens])
        fuse_attention = T.nn.functional.softmax(fuse_attention, dim=2)
        att_text_feat = T.einsum('BTVQ,BQD->BTVD', [fuse_attention, question_feature_tokens])
        fused_features = video_feature_tokens + att_text_feat

        text_pooled = T.mean(question_feature_tokens, dim=1)

        summarization_token, _ = self.get_special_token_embedding(cls_token, sep_token)

        for i in range(temporal_scale):
            immediate_res = self.fusion_transformer(
                encoder_hidden_states=fused_features[:, i, :, :],
                inputs_embeds=summarization_token,
                output_hidden_states=True,
            )

            summarization_token = summarization_token + immediate_res.hidden_states[-1]
            summarization_token = self.fusion_layer_norm(summarization_token)
            summarization_token = self.fusion_dropout(summarization_token)

        summarization_token = self.fc_fusion(summarization_token)
        text_pooled = self.fc_text(text_pooled)
        ans_feat = self.fc_answer(self.answer_features)

        question_video_feature = T.einsum('BF,AF->BA', [summarization_token.squeeze(), ans_feat])
        question_feature = T.einsum('BF,AF->BA', [text_pooled, ans_feat])
        final_out = T.einsum('BA,BA->BA', [question_video_feature, question_feature])
        return final_out