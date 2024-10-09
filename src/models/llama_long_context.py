"""
Llama style Language Model.
References:
1) Llama inference code:
https://github.com/facebookresearch/llama/blob/main/llama/model.py
2) Mistral one file ref:
https://github.com/mistralai/mistral-src/blob/main/one_file_ref.py
3) Llama paper:
https://arxiv.org/pdf/2302.13971.pdf

Main differences from GPT2:
* Uses RMSNorm instead of LayerNorm
* Uses a slightly different MLP (SwiGLU)
* rotary embeddings (RoPE)
"""
import logging
import math
from typing import Callable, Any, Sequence, List

import tiktoken
import torch
import torch.nn as nn
from torch import Tensor, LongTensor, BoolTensor
from torch.nn import functional as F
from models.base import CausalSelfAttention, GPTBase

# as of 9 Oct 2024, requires nightly build!
from torch.nn.attention.flex_attention import flex_attention, BlockMask, and_masks, create_block_mask

from models.llama import apply_rotary_emb, RMSNorm, LlamaMLP, precompute_freqs_cis

IGNORE_INDEX = -100
_score_mod_signature = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]


class FlexLlamaAttention(CausalSelfAttention):
    def forward(
            self, x,
            *,
            freqs_cis: Tensor | None = None,
            block_mask: BlockMask | None = None,
            score_mod: _score_mod_signature | None = None,
    ):
        # batch size, sequence length, embedding dimensionality (n_embd)
        (
            B,
            T,
            C,
        ) = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # (B, T, nh, hs)
        k = k.view(B, T, self.n_head, C // self.n_head)
        q = q.view(B, T, self.n_head, C // self.n_head)
        q, k = apply_rotary_emb(q, k, freqs_cis)
        # (B, nh, T, hs)
        q, k = q.transpose(1, 2), k.transpose(1, 2)

        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # flex implementation of attention
        y = flex_attention(q, k, v, block_mask=block_mask, score_mod=score_mod)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class FlexLlamaBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.attn = FlexLlamaAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, eps=config.rmsnorm_eps)
        self.mlp = LlamaMLP(config)

    def forward(
            self, x,
            *,
            freqs_cis: Tensor | None = None,
            block_mask: BlockMask | None = None,
            score_mod: _score_mod_signature | None = None,
    ):
        x = x + self.attn(self.ln_1(x), freqs_cis=freqs_cis, block_mask=block_mask, score_mod=score_mod)
        x = x + self.mlp(self.ln_2(x))
        return x


def pad_to_multiple(tensor_seq: Tensor, multiple_of: int, pad_value: Any) -> Tensor:
    batch_size, seq_len = tensor_seq.shape
    pad_amount = (multiple_of - (seq_len % multiple_of)) % multiple_of

    if pad_amount > 0:
        tensor_seq = torch.cat([
            tensor_seq,
            tensor_seq.new_full((batch_size, pad_amount), fill_value=pad_value)
        ], dim=-1)

    return tensor_seq


def find_last_non_special(idx: LongTensor, specials: List[int]):
    mask = torch.isin(idx, idx.new_tensor(specials), invert=True)

    # Reverse the sequences and convert to int64 for argmax compatibility
    reversed_indices = torch.arange(mask.size(1) - 1, -1, -1, device=idx.device)
    reversed_mask = mask[:, reversed_indices].to(torch.int64)

    # Find the first occurrence in the reversed array (batched), and convert it to the original index
    last_pos = torch.argmax(reversed_mask, dim=1)

    # If no non-special token was found, set those positions to -1
    no_non_special_found = reversed_mask.gather(1, last_pos.unsqueeze(1)).squeeze(1) == 0
    last_pos[no_non_special_found] = -1

    # Convert the reversed positions to the original positions
    last_pos[~no_non_special_found] = idx.size(1) - 1 - last_pos[~no_non_special_found]

    return last_pos


class FlexLlama(GPTBase):
    def __init__(self, config):
        super().__init__(config)
        assert config.sequence_length is not None
        self.config = config
        self.tokenizer = tiktoken.get_encoding("gpt2")
        vocab_size = self.tokenizer.n_vocab

        self.pad_token = vocab_size
        vocab_size += 1
        new_special_tokens = {'<|pad|>': self.pad_token}

        self.sink_token = None
        if config.add_sink:
            self.sink_token = vocab_size
            vocab_size += 1
            new_special_tokens['<|sink|>'] = self.sink_token

        self.block_end_token = None
        self.block_size = config.block_size
        if config.add_block_end:
            self.block_end_token = vocab_size
            vocab_size += 1
            new_special_tokens['<|block_end|>'] = self.block_end_token

        self.sink_token = self.tokenizer.n_vocab
        self.tokenizer = tiktoken.Encoding(
            name='gpt2-extended',
            pat_str=self.tokenizer._pat_str,
            mergeable_ranks=self.tokenizer._mergeable_ranks,
            special_tokens={**new_special_tokens, **self.tokenizer._special_tokens}
        )

        vocab_size = self.tokenizer.n_vocab

        # create the token and position embeddings
        self.head_dim = config.n_embd // config.n_head
        self.freqs_cis = precompute_freqs_cis(self.head_dim, config.sequence_length)

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(vocab_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([FlexLlamaBlock(config) for _ in range(config.n_layer)]),
                ln_f=RMSNorm(config.n_embd, eps=config.rmsnorm_eps),
            )
        )

        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # [VS] > I think this warning is not present in the latest torch version
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(
                    p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer)
                )

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default)
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def _is_block_end_token_added(self, idx: LongTensor) -> bool:
        if len(idx.shape) == 2:
            return (idx[:, self.block_size - 1::self.block_size] == self.block_end_token).all()
        if len(idx.shape) == 1:
            return idx[self.block_size - 1::self.block_size] == self.block_end_token
        raise ValueError(f"Unexpected idx shape {idx.shape}")

    def _is_starts_with_sink_token(self, idx: LongTensor) -> bool:
        if len(idx.shape) == 2:
            return (idx[:, 0] == self.sink_token).all()
        if len(idx.shape) == 1:
            return idx[0] == self.sink_token
        raise ValueError(f"Unexpected idx shape {idx.shape}")

    def get_block_mask(self, idx: Tensor) -> BlockMask:
        batch_size, seq_length = idx.shape

        pad_mask: BoolTensor = (idx == self.pad_token)  # noqa

        block_end_mask: BoolTensor = (idx == self.block_end_token)  # noqa
        block_ids = torch.cumsum(block_end_mask, dim=-1, dtype=torch.int)

        # determine which blocks to mask
        unique_block_ids = torch.unique(block_ids)
        keep_blocks = torch.ones_like(unique_block_ids, dtype=torch.bool)
        keep_blocks.bernoulli_(
            # enable random masking only during training (similar to dropout)
            p=self.config.mask_block_prob if self.training else 0.0
        )

        document_end_mask: BoolTensor = (idx == self.tokenizer.eot_token)  # noqa
        document_ids = torch.cumsum(document_end_mask, dim=-1, dtype=torch.int)

        def causal(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            return q_idx >= kv_idx

        def padding(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            # do not attend to padding and padding should not attend to anything
            return ~(pad_mask[b, q_idx] or pad_mask[b, kv_idx])

        def block_masking(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            q_block_id = block_ids[q_idx]
            kv_block_id = block_ids[kv_idx]

            # freely attend to own block and any other unmasked block and any block end token of other blocks
            return ((q_block_id == kv_block_id)
                    or keep_blocks[kv_block_id]
                    or block_end_mask[kv_block_id]
                    or (kv_block_id == 0))

        def document_masking(b: Tensor, h: Tensor, q_idx: Tensor, kv_idx: Tensor) -> Tensor:
            # attend only within same document
            return document_ids[q_idx] == document_ids[kv_idx]

        return create_block_mask(
            and_masks(causal, padding, block_masking, document_masking),
            B=batch_size,
            H=None,  # broadcast over heads
            Q_LEN=seq_length,
            KV_LEN=seq_length,
            BLOCK_SIZE=self.block_size,
            _compile=True
        )

    def prepare_inputs(self, idx: LongTensor, targets: LongTensor | None = None) -> dict:
        # make inputs batched
        if len(idx.shape) == 1:
            idx = idx.unsqueeze(0)

        batch_size, seq_len = idx.shape

        if targets is not None:
            if len(targets.shape) == 1:
                targets = targets.unsqueeze(0)

            assert targets.shape == idx.shape, f"idx and targets shape mismatch: {targets.shape} and {idx.shape}"

        # we will need to mask out all added tokens at the end, so we save which special tokens were in idx before
        special_tokens = idx.new_tensor(list(self.tokenizer._special_tokens.values()))
        special_tokens_mask = torch.isin(idx, special_tokens)

        # add sink token if there is no one
        if not self._is_starts_with_sink_token(idx) and self.config.add_sink:
            idx = torch.concatenate([
                idx.new_full(size=(batch_size, 1), fill_value=self.sink_token),
                idx
            ], dim=-1)

        # add block end token if not added already
        if not self._is_block_end_token_added(idx) and self.config.add_block_end:
            # pad to ensure that all blocks are full
            idx = pad_to_multiple(idx, multiple_of=self.block_size - 1, pad_value=self.pad_token)

            batch_size, seq_len = idx.shape

            # without end block token, the size of blocks is 1 less
            total_blocks = seq_len // (self.block_size - 1)
            idx = torch.concatenate([
                idx.reshape(total_blocks * batch_size, self.block_size - 1),
                # finally, this will add block_end_token to the end of every block
                idx.new_full(size=(total_blocks * batch_size, 1), fill_value=self.block_end_token)
            ], dim=-1).reshape(batch_size, total_blocks * self.block_size)  # return to normal shape

        # final shape
        batch_size, seq_len = idx.shape

        if targets is not None:
            # we've added special tokens to idx, so now we need to fix targets
            new_targets = targets.new_full((batch_size, seq_len), fill_value=IGNORE_INDEX)
            new_special_tokens_mask = torch.isin(idx, special_tokens)

            # we ignore targets for all special tokens, and keep original targets for all non-special tokens
            new_targets[~new_special_tokens_mask] = targets[~special_tokens_mask]
            targets = new_targets

        # pad to avoid any issues with FlexAttention
        idx = pad_to_multiple(idx, multiple_of=self.block_size, pad_value=self.pad_token)
        targets = pad_to_multiple(targets, multiple_of=self.block_size, pad_value=-100)

        return {'idx': idx, 'targets': targets, 'block_mask': self.get_block_mask(idx)}

    def validate_inputs(self, idx: LongTensor, targets: LongTensor | None = None) -> None:
        idx_shape = idx.shape
        batch_size, seq_len = idx_shape

        assert (
                seq_len <= self.config.sequence_length
        ), f"Cannot forward sequence of length {seq_len}, maximum length is {self.config.sequence_length}"

        if self.config.add_sink:
            assert self._is_starts_with_sink_token(idx), (f"Sequences should start with sink token: "
                                                          f"<|sink|> = {self.sink_token}, "
                                                          f"but they start with {idx[:, 0]}")
        if self.config.add_block_end:
            assert self._is_block_end_token_added(idx), (f"Every {self.block_size} token should be a block end token "
                                                         f"<|block_end|>: {self.block_end_token}!")
        if targets is not None:
            assert targets.shape == idx_shape, f"targets and idx shape mismatch: {targets.shape} and {idx_shape}"

    def forward(
            self,
            idx,
            targets=None,
            get_logits=False,
            *,
            block_mask: BlockMask | None = None,
            score_mod: _score_mod_signature | None = None,
    ):
        self.validate_inputs(idx, targets)

        device = idx.device
        batch_size, seq_len = idx.size()

        # shape (1, t)
        pos = torch.arange(0, seq_len, dtype=torch.long, device=device)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)

        x = self.transformer.drop(tok_emb)
        freqs_cis = self.freqs_cis.to(x.device)[pos]

        for block_idx, block in enumerate(self.transformer.h):
            x = block(x, freqs_cis=freqs_cis, block_mask=block_mask, score_mod=score_mod)
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=IGNORE_INDEX
            )
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last non-special token
            logits = self.lm_head(
                x[torch.arange(batch_size), find_last_non_special(idx, list(self.tokenizer._special_tokens.values()))]
            )
            loss = None

        logits = logits if get_logits else None

        return {
            "logits": logits,
            "loss": loss,
        }

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at sequence_length
            idx_cond = idx if idx.size(1) <= self.config.sequence_length else idx[:, -self.config.sequence_length:]
            inputs = self.prepare_inputs(idx_cond)
            # forward the model to get the logits for the index in the sequence
            logits = self(**inputs, get_logits=True)['logits']
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.no_grad()
    def generate_from_string(self, in_str, max_new_tokens, temperature=1.0, top_k=None):
        idx = torch.tensor(
            self.tokenizer.encode(in_str, allowed_special={"<|endoftext|>"})
        ).view(1, -1).to(self.lm_head.weight.device)
        out_idx = self.generate(idx, max_new_tokens, temperature, top_k).view(-1).to('cpu').numpy()
        return self.tokenizer.decode(out_idx)
