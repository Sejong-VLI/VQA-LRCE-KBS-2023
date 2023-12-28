import torch as T
import transformers
import torch
import copy
import math
import matplotlib.pyplot as plt
from ptflops import get_model_complexity_info
from lrce.models.fusionv3 import FusionTransformer
from transformers.modeling_outputs import BaseModelOutput
from transformers import DistilBertConfig
from transformers.activations import gelu
from tqdm import tqdm


class MultiHeadSelfAttention(T.nn.Module):
    def __init__(self, config):
        super().__init__()

        self.n_heads = config.n_heads
        self.dim = config.dim
        self.dropout = T.nn.Dropout(p=config.attention_dropout)

        assert self.dim % self.n_heads == 0

        self.q_lin = T.nn.Linear(in_features=config.dim, out_features=config.dim)
        self.k_lin = T.nn.Linear(in_features=config.dim, out_features=config.dim)
        self.v_lin = T.nn.Linear(in_features=config.dim, out_features=config.dim)
        self.out_lin = T.nn.Linear(in_features=config.dim, out_features=config.dim)

        self.pruned_heads = set()

    def forward(self, query, key, value, mask, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        query: torch.tensor(bs, seq_length, dim)
        key: torch.tensor(bs, seq_length, dim)
        value: torch.tensor(bs, seq_length, dim)
        mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            Attention weights
        context: torch.tensor(bs, seq_length, dim)
            Contextualized layer. Optional: only if `output_attentions=True`
        """
        bs, q_length, dim = query.size()
        k_length = key.size(1)
        # assert dim == self.dim, 'Dimensions do not match: %s input vs %s configured' % (dim, self.dim)
        # assert key.size() == value.size()

        dim_per_head = self.dim // self.n_heads

        mask_reshp = (bs, 1, 1, k_length)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return (x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head))

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))  # (bs, n_heads, q_length, k_length)
        mask = ((mask == 0).view(mask_reshp).expand_as(scores))  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = T.nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return (context, weights)
        else:
            return (context, )


class FFN(T.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = T.nn.Dropout(p=config.dropout)
        self.lin1 = T.nn.Linear(in_features=config.dim, out_features=config.hidden_dim)
        self.lin2 = T.nn.Linear(in_features=config.hidden_dim, out_features=config.dim)
        assert config.activation in [
            "relu",
            "gelu",
        ], "activation ({}) must be in ['relu', 'gelu']".format(config.activation)
        self.activation = gelu if config.activation == "gelu" else T.nn.ReLU()

    def forward(self, input):
        x = self.lin1(input)
        x = self.activation(x)
        x = self.lin2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(T.nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.dim % config.n_heads == 0

        self.attention = MultiHeadSelfAttention(config)
        self.sa_layer_norm = T.nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

        self.ffn = FFN(config)
        self.output_layer_norm = T.nn.LayerNorm(normalized_shape=config.dim, eps=1e-12)

    def forward(self, x, attn_mask=None, head_mask=None, output_attentions=False):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
        attn_mask: torch.tensor(bs, seq_length)

        Outputs
        -------
        sa_weights: torch.tensor(bs, n_heads, seq_length, seq_length)
            The attention weights
        ffn_output: torch.tensor(bs, seq_length, dim)
            The output of the transformer block contextualization.
        """
        # Self-Attention
        sa_output = self.attention(
            query=x,
            key=x,
            value=x,
            mask=attn_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
        )
        if output_attentions:
            (
                sa_output,
                sa_weights,
            ) = sa_output  # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
        else:  # To handle these `output_attention` or `output_hidden_states` cases returning tuples
            assert type(sa_output) == tuple
            sa_output = sa_output[0]
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output, )
        if output_attentions:
            output = (sa_weights, ) + output
        return output


class Transformer(T.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers

        layer = TransformerBlock(config)
        self.layer = T.nn.ModuleList([copy.deepcopy(layer) for _ in range(config.n_layers)])

    def forward(
        self,
        x,
        attn_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=None,
    ):
        """
        Parameters
        ----------
        x: torch.tensor(bs, seq_length, dim)
            Input sequence embedded.
        attn_mask: torch.tensor(bs, seq_length)
            Attention mask on the sequence.

        Outputs
        -------
        hidden_state: torch.tensor(bs, seq_length, dim)
            Sequence of hiddens states in the last (top) layer
        all_hidden_states: Tuple[torch.tensor(bs, seq_length, dim)]
            Tuple of length n_layers with the hidden states from each layer.
            Optional: only if output_hidden_states=True
        all_attentions: Tuple[torch.tensor(bs, n_heads, seq_length, seq_length)]
            Tuple of length n_layers with the attention weights from each layer
            Optional: only if output_attentions=True
        """
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_state = x
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state, )
            if head_mask is not None:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=head_mask[i],
                    output_attentions=output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    x=hidden_state,
                    attn_mask=attn_mask,
                    head_mask=None,
                    output_attentions=output_attentions,
                )
            hidden_state = layer_outputs[-1]

            if output_attentions:
                assert len(layer_outputs) == 2
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions, )
            else:
                assert len(layer_outputs) == 1

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state, )

        if not return_dict:
            return tuple(v for v in [hidden_state, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_state,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


class VQAT(T.nn.Module):
    def __init__(self, feature_dim, video_token_length, text_token_length, batch=1) -> None:
        super().__init__()
        config = DistilBertConfig.from_pretrained(
            "distilbert-base-uncased",
            n_layers=12,
            dim=feature_dim,
            dropout=0.1,
            hidden_dim=3072,
            attention_dropout=0.1,
            n_heads=12,
        )
        self.model = Transformer(config)
        self.update_input(batch, video_token_length, text_token_length, batch)

    def update_input(self, feature_dim, video_token_length, text_token_length, batch=1):
        self.vq_features = torch.rand(batch, video_token_length + text_token_length, feature_dim)
        self.mask = torch.ones(batch, video_token_length + text_token_length)

    def forward(self, _):
        return self.model(x=self.vq_features, attn_mask=self.mask)[0]


class VIOLET(T.nn.Module):
    def __init__(self, feature_dim, video_token_length, text_token_length, batch=1):
        super().__init__()
        bert = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.mask_ext, self.trsfr = bert.get_extended_attention_mask, bert.bert.encoder
        self.update_input(batch, video_token_length, text_token_length, batch)

    def update_input(self, feature_dim, video_token_length, text_token_length, batch=1):
        self.video_features = T.rand(batch, video_token_length, feature_dim)
        self.text_features = T.rand(batch, text_token_length, feature_dim)
        self.video_mask = T.rand(batch, video_token_length)
        self.text_mask = T.rand(batch, text_token_length)

    def go_cross(self, feat_img, mask_img, feat_txt, mask_txt):
        feat, mask = T.cat([feat_img, feat_txt], dim=1), T.cat([mask_img, mask_txt], dim=1)
        mask = self.mask_ext(mask, mask.shape, mask.device)
        out = self.trsfr(feat, mask, output_attentions=True)
        return out['last_hidden_state'], out['attentions']

    def forward(self, _):
        return self.go_cross(self.video_features, self.video_mask, self.text_features, self.text_mask)


class LRCE(T.nn.Module):
    def __init__(self, feature_dim, video_token_length, text_token_length, batch=1) -> None:
        super().__init__()
        self.model = FusionTransformer(feature_dim)
        self.update_input(batch, video_token_length, text_token_length, batch)

    def update_input(self, feature_dim, video_token_length, text_token_length, batch=1):
        self.video_features = T.rand(batch, 3, video_token_length, feature_dim)
        self.text_features = T.rand(batch, text_token_length, feature_dim)

    def forward(self, _):
        return self.model(self.video_features, self.text_features, None)


def benchmark(model, video_token_length, text_token_length, feature_dim, batch_size=1):
    model.update_input(feature_dim, video_token_length, text_token_length, batch=batch_size)
    with torch.no_grad(), T.profiler.profile(
            activities=[T.profiler.ProfilerActivity.CPU, T.profiler.ProfilerActivity.CUDA],
            with_flops=True,
            with_modules=True,
            profile_memory=True,
            record_shapes=True,
    ) as prof:
        model(None)
    total_flops = 0
    for evt in prof.key_averages(group_by_input_shape=False):
        if evt.key not in ['aten::addmm', '[memory]', 'cudaDeviceSynchronize']:  # exclude linear layer and memory deallocation
            total_flops += evt.flops
            total_runtime += evt.self_cpu_time_total
            total_memory += evt.self_cpu_memory_usage
    return total_flops / 1000000, total_runtime / 1000, total_memory / 1048576  # MFLOPS, ms, MB


batch_size = 1
feature_dim = 768

lrce = LRCE(feature_dim, 1, 1, batch=batch_size)
vqat = VQAT(feature_dim, 1, 1, batch=batch_size)
violet = VIOLET(feature_dim, 1, 1, batch=batch_size)

lrce_data = {'token_length': [], 'flops': [], 'runtime': [], 'memory': []}
violet_data = {'token_length': [], 'flops': [], 'runtime': [], 'memory': []}
vqat_data = {'token_length': [], 'flops': [], 'runtime': [], 'memory': []}
video_token_length = 31
text_token_length = 14
benchmark(lrce, video_token_length, text_token_length, feature_dim, batch_size)
for _ in tqdm(range(4)):
    video_token_length *= 2
    text_token_length *= 2

    violet_res = benchmark(violet, video_token_length, text_token_length,
                           feature_dim, batch_size)
    vqat_res = benchmark(vqat, video_token_length, text_token_length,
                         feature_dim, batch_size)
    lrce_res = benchmark(lrce, video_token_length, text_token_length,
                         feature_dim, batch_size)

    lrce_data['token_length'].append(video_token_length + text_token_length)
    lrce_data['flops'].append(lrce_res[0])
    lrce_data['runtime'].append(lrce_res[1])
    lrce_data['memory'].append(lrce_res[2])

    violet_data['token_length'].append(video_token_length + text_token_length)
    violet_data['flops'].append(violet_res[0])
    violet_data['runtime'].append(violet_res[1])
    violet_data['memory'].append(violet_res[2])

    vqat_data['token_length'].append(video_token_length + text_token_length)
    vqat_data['flops'].append(vqat_res[0])
    vqat_data['runtime'].append(vqat_res[1])
    vqat_data['memory'].append(vqat_res[2])

lrce_df = pd.DataFrame.from_dict(lrce_data)
violet_df = pd.DataFrame.from_dict(violet_data)
vqat_df = pd.DataFrame.from_dict(vqat_data)

print('LRCE')
print(lrce_df)
print('VIOLET')
print(violet_df)
print('VQAT')
print(vqat_df)
