import math
from typing import Any, Dict, Optional, List, Tuple, Union
import torch
import torch.nn as nn
import numpy as np


from einops import rearrange

from transformers.modeling_outputs import SequenceClassifierOutputWithPast, CausalLMOutputWithPast, BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from transformers.cache_utils import Cache, DynamicCache

from triton_flash_blocksparse_attn import get_local_strided_sparse_attention_op, BlockSparseParams
from positional_embedding import RotaryEmbedding, RevisedYaRNRotaryEmbedding

from my_configuration_tlg import TLGv4Config

logger = logging.get_logger(__name__)

LegacyCache = Tuple[Tuple[torch.FloatTensor]]


@torch.jit.script
def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)


@torch.jit.script
def gegelu(input, limit: Optional[float] = None):
    a_gelu, a_linear = input[..., ::2], input[..., 1::2]
    if limit is not None:
        a_gelu = torch.where(
            torch.isinf(a_gelu), a_gelu, a_gelu.clamp(min=None, max=limit)
        )
        a_linear = torch.where(
            torch.isinf(a_linear), a_linear, a_linear.clamp(min=-limit, max=limit)
        )
    out_gelu = quick_gelu(a_gelu)
    return out_gelu * (a_linear + 1)

class TLGv4MLP(nn.Module):
    def __init__(self, config: TLGv4Config):
        super().__init__()
        self.config = config
        assert self.config.hidden_act == "gegelu", "Only `gegelu` is supported for the 4.7 series of models .."
        self.hidden_size = config.hidden_size
        self.gegelu_limit = config.gegelu_limit
        self.intermediate_size = config.intermediate_size

        self.up_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size)
        self.dropout = nn.Dropout(config.ffn_dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(
            self.down_proj(
                gegelu(self.up_proj(x), limit=self.gegelu_limit)
            )
        )


class TLGv4SelfAttention(nn.Module):
    def __init__(self, config: TLGv4Config, layer_idx: Optional[int] = None) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )
        
        self.hidden_size = config.hidden_size
        # Number of Query Heads
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        # Number of Key Value HEads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_q_per_kv = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_embedding_base = config.rope_embedding_base
        self.rope_position_scale = config.rope_position_scale
        self.is_causal = True

        self.attention_dropout_rate = config.attention_dropout_prob

        norm_factor = None
        if config.mup_use_scaling:
            norm_factor = self.head_dim / config.mup_attn_multiplier
        else:
            norm_factor = math.sqrt(self.head_dim)
        self.softmax_scale = 1.0 / norm_factor

        self.query_key_value = nn.Linear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)

        # BlockSparse related Parameters
        self.blocksparse_params = BlockSparseParams.from_config(config)

        ## Rotary Positional Embeddings
        # self.rotary_emb = RotaryEmbedding(
        #    dim_model=self.head_dim,
        #    max_seq_len=self.max_position_embeddings,
        #    base=self.rope_embedding_base,
        #    position_scale=self.rope_position_scale,
        #)
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_embedding_base)
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                raise ValueError("Linear scaling is not supported for the TLG models")
                self.rotary_emb = LinearScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_embedding_base, scaling_factor=scaling_factor
                )
            elif scaling_type == "dynamic":
                raise ValueError("dynamic scaling is not supported for the TLG models")
                self.rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_embedding_base, scaling_factor=scaling_factor
                )
            elif scaling_type == "vanilla_ntk":
                raise ValueError("vanilla_ntk scaling is not supported for the TLG models")
                self.rotary_emb = VanillaNTKScalingRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_embedding_base, scaling_factor=scaling_factor
                )
            elif scaling_type == "yarn":
                original_max_position_embeddings = self.config.rope_scaling["original_max_position_embeddings"]
                self.rotary_emb = RevisedYaRNRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_embedding_base, scale=scaling_factor, original_max_position_embeddings=original_max_position_embeddings
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    
    def _split_heads(self, mixed_x_layer: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, sq, _ = mixed_x_layer.size()
        r"""
        The main idea is that we group tensors as
        [bs, sq, (q00, q01, ... q0m, k0, v0), (q10, q11, ... q1m, k1, v1), ... (qn0, qn1, ... qnm, kn, vn)]
        That ways, when the MP column sharding happens, this tensor will be sharded keeping all the
        queries and keys intact. In order to get the correct qkv, we first break into groups, and then
        index into the groups.
        """

        intermediate_shape = (bs, sq, -1, (self.num_q_per_kv + 2), self.head_dim)
        mixed_x_layer = mixed_x_layer.view(*intermediate_shape)
        q = mixed_x_layer[:, :, :, :-2]
        k = mixed_x_layer[:, :, :, [-2]]
        v = mixed_x_layer[:, :, :, [-1]]
        q, k, v = [
            rearrange(
                x,
                "bs sq group nh hn -> bs sq (group nh) hn"
            ) for x in (q, k, v)
        ]
        return q, k, v

    def _apply_blocksparse_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        return_attention_probs: bool = False,
    ) -> torch.Tensor:

        assert not return_attention_probs, "return_attention_probs is not supported for blocksparse attention"
        # q, k, v: (bs, sq, np, hn) -> (bs, np, sq, hn)
        q, k, v = q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        blocksparse_attn_fn = get_local_strided_sparse_attention_op(
            n_heads=self.num_heads,
            max_seq_len=self.max_position_embeddings,
            sparse_block_size=self.blocksparse_params.block_size,
            kernel_block_size=self.blocksparse_params.kernel_block_size,
            local_blocks=self.blocksparse_params.num_local_blocks,
            vert_stride=self.blocksparse_params.vert_stride,
            homo_head=self.blocksparse_params.homo_head_pattern,
            device=q.device,
            inference=not self.training
        )
        # [bs, np, sq, hn]
        context_layer = blocksparse_attn_fn(q, k, v, self.softmax_scale)
        # [bs, sq, np, hn]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        return context_layer

    
    def expand_kv_to_q_size(self, kv: torch.Tensor, num_q_per_kv: int) -> torch.Tensor:
        """
        ... note::
            Right now, I am using a repeat_interleave to expand the kv to the size of q.
            This incurs a memory penalty, since the tensors are actually copied.
            TODO: If this does yield benefits, then potentially we can rewrite the
            flash-attn kernel in a way that allows this different head sharing.
        """

        repeats = torch.tensor([num_q_per_kv] * kv.size(3)).to(kv.device)
        total = repeats.sum()
        expanded_kv = torch.repeat_interleave(
            kv,
            repeats=repeats,
            dim=3,
            output_size=total
        )
        return expanded_kv

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """_summary_

        Args:
            hidden_states (torch.Tensor): _description_
            attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
            position_ids (Optional[torch.LongTensor], optional): _description_. Defaults to None.
            past_key_value (Optional[Cache], optional): _description_. Defaults to None.
            output_attentions (bool, optional): _description_. Defaults to False.
            use_cache (bool, optional): _description_. Defaults to False.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]: _description_
        
        Notations:
        ------------
            bs: batch size
            sq_len: sequence length of the entire sequence
            q_len: sequence length of the query
            cache_sq: sequence length in the cache
                If there is no cache then cache_sq = 0
                and sq_len = q_len
                otherwise sq_len = q_len + cache_sq
            h: hidden size
            nq: number of query heads
            nkv: number of key heads
            hn: hidden size per head
                hn = h // nq
            nqp: number of query heads (per MP partition)
                nqp = nq // (num mp partitions)
            nkvp: number of key-value heads (per MP partition)
                nkvp = nk // (num mp partitions)

        """
        # shape: (bs, q_len, h)
        bsz, q_len, _ = hidden_states.size()
        # shape: (bs, q_len, (nqp + 2 * nkvp) * hn)
        mixed_x_layer = self.query_key_value(hidden_states)
        # shape: (bs, q_len, nqp * hn), shape: (bs, q_len, nkvp * hn), shape: (bs, q_len, nkvp * hn)
        q, k, v = self._split_heads(mixed_x_layer)

        attention_dropout_prob = self.attention_dropout_rate if self.training else 0.0
        kv_seq_len = k.size(1)
        if past_key_values is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            if self.rotary_emb is not None:
                seqlen_offset = past_key_values.get_usable_length(kv_seq_len)
                q, k = self.rotary_emb(
                    q, k, seq_dimension=1, seqlen_offset=seqlen_offset
                )
            # shape: (bs, nkvp, cache_sq + q_len, hn), shape: (bs, nkvp, cache_sq + q_len, hn)
            k, v = past_key_values.update(key_states=k.transpose(1, 2), value_states=v.transpose(1, 2), layer_idx=self.layer_idx)
            # shape: (bs, cache_sq + q_len, nkvp, hn), shape: (bs, cache_sq + q_len, nkvp, hn)
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            # shape: (bs, seq_len, 2, nkvp, hn)
            combined_kv = torch.cat((k.unsqueeze(2), v.unsqueeze(2)), dim=2)
            # shape: (bs, seq_len, 2, nqp, hn)
            expanded_kv = self.expand_kv_to_q_size(combined_kv, num_q_per_kv=self.num_q_per_kv)
            expanded_k, expanded_v = expanded_kv[:, :, 0], expanded_kv[:, :, 1]
            blocksparse_output = self._apply_blocksparse_attention(
                q, expanded_k, expanded_v, output_attentions
            )
        else:
            # In this case seq_len = q_len and cache_sq = 0
            if self.rotary_emb is not None:
                # shape: (bs, seq_len, np, hn), shape: (bs, seq_len, nkvp, hn)
                q, k = self.rotary_emb(q, k)
            # shape: (bs, seq_len, 2, nkvp, hn)
            kv = torch.cat((k.unsqueeze(2), v.unsqueeze(2)), dim=2)
            # shape: (bs, seq_len, 2, nqp, hn)
            expanded_kv = self.expand_kv_to_q_size(kv, num_q_per_kv=self.num_q_per_kv)
            expanded_k, expanded_v = expanded_kv[:, :, 0], expanded_kv[:, :, 1]
            blocksparse_output = self._apply_blocksparse_attention(
                q, expanded_k, expanded_v, output_attentions
            )
        attn_weights = None
        if output_attentions:
            attn_output, attn_weights = blocksparse_output
        else:
            attn_output = blocksparse_output
        attn_output = attn_output.view(bsz, q_len, -1)
        attn_output = self.dense(attn_output)
        return attn_output, attn_weights, past_key_values
        

class TLGv4DecoderLayer(nn.Module):
    def __init__(self, config: TLGv4Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = TLGv4SelfAttention(config, layer_idx)
        self.mlp = TLGv4MLP(config)

        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor], Optional[Cache]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_values = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_values,)

        return outputs



class TLGv4PreTrainedModel(PreTrainedModel):
    config_class = TLGv4Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["TLGv4DecoderLayer"]
    skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = False

    def _init_weights(self, module: nn.Module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        # The output projection on the decoder attention layer as well as the down_proj in the MLP are scaled
        # differently (dubbed `output_layer_init_method` in the Megatron code). This is replicated here
        for name, p in module.named_parameters():
            if any(x in name for x in ("c_proj.weight", "down_proj.weight", "o_proj.weight")):
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                p.data.normal_(mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)))


class TLGv4Model(TLGv4PreTrainedModel):

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # Embedding Dropout
        self.embedding_dropout = nn.Dropout(config.embedding_dropout_prob)
        
        # MuP Embedding scaling
        self.mup_embedding_multiplier = config.mup_embedding_multiplier

        self.layers = nn.ModuleList([TLGv4DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])

        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, LegacyCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False
        
        past_key_values_length = 0

        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)
        
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                past_key_values_length, past_key_values_length + seq_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()
        
        if attention_mask is not None:
            if batch_size <= 0:
                raise ValueError("batch_size has to be defined and > 0")
            logger.warning_once(
                "The `attention_mask` for this model in computed directly inside the attention kernel. A custom attention mask will be ignored ..."
            )
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask[:, None, None, :]
            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and the dtype's smallest value for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min
        
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        inputs_embeds = self.embedding_dropout(inputs_embeds)

        if self.mup_embedding_multiplier is not None and self.mup_embedding_multiplier > 0.0:
            inputs_embeds = inputs_embeds * self.mup_embedding_multiplier

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                # Following the Mistral schema for layer return values
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.final_layernorm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        
        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        


class TLGv4ForCausalLM(TLGv4PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = TLGv4Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, self.vocab_size, bias=False)
        self.mup_width_multiplier = config.mup_width_multiplier

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value
    
    def set_decoder(self, decoder):
        self.model = decoder
    
    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,   
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        if self.mup_width_multiplier:
            logits = logits / self.mup_width_multiplier

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def prepare_inputs_for_generation(
        self, 
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) -> Dict[str, Any]:
        # only last token for inputs_ids if past is defined in kwargs
        if past_key_values:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
    
    


class TLGv4ForSequenceClassification(TLGv4PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = TLGv4Model(config)
        self.score = nn.Linear(config.hidden_size, self.num_labels, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]
        logits = self.score(hidden_states)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                # if no pad token found, use modulo instead of reverse indexing for ONNX compatibility
                sequence_lengths = torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(pooled_logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(pooled_logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(pooled_logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = nn.BCEWithLogitsLoss()
                loss = loss_fct(pooled_logits, labels)
        if not return_dict:
            output = (pooled_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            loss=loss,
            logits=pooled_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )
        
        