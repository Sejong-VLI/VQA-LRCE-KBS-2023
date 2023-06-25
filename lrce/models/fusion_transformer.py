from lrce.lib import *
from transformers.activations import gelu

#DEPRECATED DO NOT USE


def calculate_special_token_embedding():
    bert = transformers.BertModel.from_pretrained('bert-base-uncased')
    embedding = bert.embeddings
    embedding.eval()
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    with T.no_grad():
        cls_token = torch.LongTensor(tokenizer.encode(CLS_TOKEN, add_special_tokens=False))
        sep_token = torch.LongTensor(tokenizer.encode(SEP_TOKEN, add_special_tokens=False))
        cls_token = cls_token.reshape(1, -1)
        sep_token = sep_token.reshape(1, -1)
        special_token = T.concat([cls_token, sep_token], dim=1)
        embedded_special_token = embedding(special_token)  # BATCH, 2, feature_dim
    return embedded_special_token[:, 0:1, :], embedded_special_token[:, 1:2, :]


def calculate_sep_token_embedding():
    bert = transformers.BertModel.from_pretrained('bert-base-uncased')
    embedding = bert.embeddings
    embedding.eval()
    tokenizer = transformers.BertTokenizerFast.from_pretrained('bert-base-uncased')

    with T.no_grad():
        sep_token = torch.LongTensor(tokenizer.encode(SEP_TOKEN, add_special_tokens=False))
        sep_token = sep_token.reshape(1, -1)
        embedded_sep_token = embedding(sep_token)  # BATCH, 1, feature_dim
    return embedded_sep_token


class EncoderBlock(T.nn.Module):
    def __init__(
        self,
        feature_dim: int = 768,
        num_heads: int = 12,
        expansion_size: int = 4,
        drop_out_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.attention = T.nn.MultiheadAttention(
            feature_dim,
            num_heads,
            dropout=0.1,
            bias=True,
            batch_first=True,
        )
        self.layer_norm1 = T.nn.LayerNorm(feature_dim, eps=1e-12)
        self.layer_norm2 = T.nn.LayerNorm(feature_dim, eps=1e-12)
        self.dropout = T.nn.Dropout(drop_out_rate)
        self.fc = T.nn.Sequential(*[
            T.nn.Linear(feature_dim, feature_dim * expansion_size),
            T.nn.GELU(),
            T.nn.Linear(feature_dim * expansion_size, feature_dim),
        ])

    def forward(self, query: T.tensor, key: T.tensor, value: T.tensor):
        attention_res, _ = self.attention(query, key, value, need_weights=True)  # (BATCH, QUERY_LEN, feature_dim)
        out = self.layer_norm1(attention_res + query)
        fc_out = self.fc(out)
        fc_out = self.dropout(out)
        fc_out = self.layer_norm2(fc_out + out)
        return fc_out


class BaseFusionTransformer(T.nn.Module):
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
        self.fusion_layer_norm = T.nn.LayerNorm(feature_dim, eps=1e-12)
        self.dropout = T.nn.Dropout(drop_out_rate)

    def do_fusion(self, query: T.tensor, key: T.tensor, value: T.tensor):
        out = query
        for layer in self.layers:
            out = layer(out, key, value)

        return out

    def forward(self, video_features: T.tensor, text_features: T.tensor):
        raise NotImplementedError()