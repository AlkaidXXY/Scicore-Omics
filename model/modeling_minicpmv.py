import math
from typing import List, Optional
import json
import os
from threading import Thread
from copy import deepcopy

import torch
import torch.nn as nn
import torchvision
import anndata as ad
from PIL import Image

from transformers import AutoProcessor, Qwen2PreTrainedModel, Qwen2ForCausalLM, TextIteratorStreamer

from .configuration_minicpm import MiniCPMVConfig
from .modeling_navit_siglip import SiglipVisionTransformer
from .resampler import Resampler
from .processing_minicpmv import MiniCPMVProcessor

# gene
from .modeling_nicheformer import NicheformerModel
from .configuration_nicheformer import NicheformerConfig
from .gene_projector_module import GeneProjector
from .gene_qformer_module import GeneQFormerBiomedBERT

def _is_debug_enabled() -> bool:
    return os.getenv("DEBUG_GENE", "0") == "1"


def _assert_finite(x: torch.Tensor, name: str):
    if not torch.is_tensor(x):
        return
    if not torch.isfinite(x).all():
        with torch.no_grad():
            finite_mask = torch.isfinite(x)
            num_bad = (~finite_mask).sum().item()
            msg = (
                f"[NaN/Inf Detected] {name} has non-finite values. "
                f"bad_count={num_bad}, dtype={x.dtype}, device={x.device}, shape={tuple(x.shape)}"
            )
        raise RuntimeError(msg)


class MiniCPMVPreTrainedModel(Qwen2PreTrainedModel):
    config_class = MiniCPMVConfig


class MiniCPMV(MiniCPMVPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.llm = Qwen2ForCausalLM(config)

        self.vpm = self.init_vision_module()
        self.vision_dim = self.vpm.embed_dim
        self.embed_dim = self.llm.config.hidden_size

        self.resampler = self.init_resampler(self.embed_dim, self.vision_dim)

        # ===== Gene modules =====
        self.nicheformer = self.init_gene_module(config)
        self.gene_dim = self.nicheformer.config.dim_model  # e.g. 512

        self.gene_qformer = GeneQFormerBiomedBERT(
            biomedbert_name="/data2/xiaoxinyu/biomedbert",  # 只用来读 config（不会加载权重）
            gene_in_dim=self.gene_dim,  # 512
            hidden=768,
            num_queries=32,
            load_pretrained_bert=False,  # ✅ 关键：不从 biomedbert 加载权重
        )

        # Project: 768 -> LLM hidden (e.g. 3584)
        self.gene_projector = GeneProjector(in_dim=768, out_dim=self.embed_dim)

        self.processor = None
        self.terminators = ['<|im_end|>', '<|endoftext|>']
        self._generate = self.generate

        self._gene_fp32_forced = False

        # self.post_init()

    def init_gene_module(self, config):
        if hasattr(config, "gene_config"):
            return NicheformerModel(config.gene_config)
        else:
            nicheformer_config = NicheformerConfig()
            return NicheformerModel(nicheformer_config)

    def init_vision_module(self):
        if self.config._attn_implementation == 'flash_attention_2':
            self.config.vision_config._attn_implementation = 'flash_attention_2'
        else:
            self.config.vision_config._attn_implementation = 'eager'

        model = SiglipVisionTransformer(self.config.vision_config)
        if self.config.drop_vision_last_layer:
            model.encoder.layers = model.encoder.layers[:-1]

        setattr(model, 'embed_dim', model.embeddings.embed_dim)
        setattr(model, 'patch_size', model.embeddings.patch_size)
        return model

    def init_resampler(self, embed_dim, vision_dim):
        return Resampler(
            num_queries=self.config.query_num,
            embed_dim=embed_dim,
            num_heads=embed_dim // 128,
            kv_dim=vision_dim,
            adaptive=True
        )

    def get_input_embeddings(self):
        return self.llm.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.llm.embed_tokens = value

    def get_output_embeddings(self):
        return self.llm.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.llm.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.llm = decoder

    def get_decoder(self):
        return self.llm


    def get_vllm_embedding(self, data):
        dtype = self.llm.model.embed_tokens.weight.dtype
        device = self.llm.model.embed_tokens.weight.device
        self.gene_qformer = self.gene_qformer.float()
        self.gene_projector = self.gene_projector.float()

        # =========================
        # 1) Vision
        # =========================
        if 'vision_hidden_states' not in data:
            tgt_sizes = data['tgt_sizes']
            pixel_values_list = data['pixel_values']
            vision_hidden_states = []
            all_pixel_values = []
            img_cnt = []

            for pixel_values in pixel_values_list:
                img_cnt.append(len(pixel_values))
                all_pixel_values.extend([i.flatten(end_dim=1).permute(1, 0) for i in pixel_values])

            if all_pixel_values:
                tgt_sizes = [tgt_size for tgt_size in tgt_sizes if isinstance(tgt_size, torch.Tensor)]
                tgt_sizes = torch.vstack(tgt_sizes).type(torch.int32)

                max_patches = torch.max(tgt_sizes[:, 0] * tgt_sizes[:, 1])

                all_pixel_values = torch.nn.utils.rnn.pad_sequence(
                    all_pixel_values, batch_first=True, padding_value=0.0
                )
                B, L, _ = all_pixel_values.shape
                all_pixel_values = all_pixel_values.permute(0, 2, 1).reshape(B, 3, -1, L)

                patch_attn_mask = torch.zeros((B, 1, max_patches), dtype=torch.bool, device=device)
                for i in range(B):
                    patch_attn_mask[i, 0, :tgt_sizes[i][0] * tgt_sizes[i][1]] = True

                vision_batch_size = self.config.vision_batch_size
                all_pixel_values = all_pixel_values.to(device=device, dtype=dtype)

                if B > vision_batch_size:
                    hs = []
                    for i in range(0, B, vision_batch_size):
                        start_idx = i
                        end_idx = i + vision_batch_size
                        tmp_hs = self.vpm(
                            all_pixel_values[start_idx:end_idx],
                            patch_attention_mask=patch_attn_mask[start_idx:end_idx],
                            tgt_sizes=tgt_sizes[start_idx:end_idx]
                        ).last_hidden_state
                        hs.append(tmp_hs)
                    vision_embedding = torch.cat(hs, dim=0)
                else:
                    vision_embedding = self.vpm(
                        all_pixel_values,
                        patch_attention_mask=patch_attn_mask,
                        tgt_sizes=tgt_sizes
                    ).last_hidden_state

                vision_embedding = self.resampler(vision_embedding, tgt_sizes)

                start = 0
                for pixel_values in pixel_values_list:
                    c = len(pixel_values)
                    if c > 0:
                        vision_hidden_states.append(vision_embedding[start: start + c])
                        start += c
                    else:
                        vision_hidden_states.append([])
            else:
                # no image
                if self.training:
                    dummy_image = torch.zeros((1, 3, 224, 224), device=device, dtype=dtype)
                    tgt_sizes_dummy = torch.Tensor(
                        [[(224 // self.config.patch_size), math.ceil(224 / self.config.patch_size)]]
                    ).type(torch.int32)
                    dummy_feature = self.resampler(self.vpm(dummy_image).last_hidden_state, tgt_sizes_dummy)
                else:
                    dummy_feature = []
                for _ in range(len(pixel_values_list)):
                    vision_hidden_states.append(dummy_feature)
        else:
            vision_hidden_states = data['vision_hidden_states']

        # =========================
        # 2) Gene
        # =========================
        bs = len(data['input_ids'])
        gene_hidden_states = [None] * bs

        if 'gene_input_ids' in data and data['gene_input_ids'] is not None:
            gene_input_ids = data['gene_input_ids'].to(device)
            gene_attention_mask = data.get('gene_attention_mask', None)
            if gene_attention_mask is not None:
                gene_attention_mask = gene_attention_mask.to(device)

            # Nicheformer: expect [B, seq_len, gene_dim]
            nicheformer_output = self.nicheformer.forward(
                input_ids=gene_input_ids,
                attention_mask=gene_attention_mask
            )

            gene_tokens = nicheformer_output[:, 3:, :]  # [B, L, 512]

            gene_pad_mask = None
            if gene_attention_mask is not None:
                gene_pad_mask = (gene_attention_mask[:, 3:] == 0)


            qformer_dtype = next(self.gene_qformer.parameters()).dtype
            gene_tokens = gene_tokens.to(dtype=qformer_dtype)


            q_tokens = self.gene_qformer(gene_tokens, gene_pad_mask=gene_pad_mask)

            proj_dtype = next(self.gene_projector.parameters()).dtype
            q_tokens = q_tokens.to(dtype=proj_dtype)
            gene_tokens_llm = self.gene_projector(q_tokens)  # [B, 32, 3584]
            _assert_finite(gene_tokens, "gene_tokens(after nicheformer)")
            _assert_finite(q_tokens, "q_tokens(after qformer)")
            _assert_finite(gene_tokens_llm, "gene_tokens_llm(after projector)")

            gene_bounds = data.get('gene_bound', [None] * bs)

            for i, bounds in enumerate(gene_bounds):
                # bounds can be: None, [], or a Tensor of shape [N,2]
                if bounds is None:
                    continue
                if isinstance(bounds, list) and len(bounds) == 0:
                    continue
                if torch.is_tensor(bounds) and bounds.numel() == 0:
                    continue


                gene_hidden_states[i] = gene_tokens_llm[i]  # [32, embed_dim]

        # =========================
        # 3) Text token embeddings
        # =========================
        if hasattr(self.llm.config, 'scale_emb'):
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids']) * self.llm.config.scale_emb
        else:
            vllm_embedding = self.llm.model.embed_tokens(data['input_ids'])

        new_vllm_embedding = vllm_embedding.clone()

        # dtype/device align
        vision_hidden_states = [
            x.to(dtype=vllm_embedding.dtype, device=vllm_embedding.device) if torch.is_tensor(x) else x
            for x in vision_hidden_states
        ]
        gene_hidden_states = [
            x.to(dtype=vllm_embedding.dtype, device=vllm_embedding.device) if torch.is_tensor(x) else x
            for x in gene_hidden_states
        ]

        # =========================
        # 4) Scatter insert: image + gene
        # =========================
        for i in range(bs):
            # ---- image ----
            cur_vs_hs = vision_hidden_states[i]
            if torch.is_tensor(cur_vs_hs) and cur_vs_hs.numel() > 0:
                cur_vllm_emb = vllm_embedding[i]
                cur_image_bound = data['image_bound'][i]
                if len(cur_image_bound) > 0:
                    image_indices = torch.cat([
                        torch.arange(r[0], r[1], dtype=torch.long, device=vllm_embedding.device)
                        for r in cur_image_bound if (r[1] - r[0]) > 1
                    ])
                    new_vllm_embedding[i] = cur_vllm_emb.scatter(
                        0,
                        image_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_vs_hs.view(-1, cur_vs_hs.shape[-1])
                    )
                elif self.training:
                    new_vllm_embedding[i] += cur_vs_hs[0].mean() * 0

            # ---- gene ----
            cur_gene_hs = gene_hidden_states[i]
            if cur_gene_hs is not None:
                cur_gene_bound = data.get('gene_bound', [[] for _ in range(bs)])[i]
                if len(cur_gene_bound) > 0:
                    r = cur_gene_bound[0]  # [start, end)
                    gene_indices = torch.arange(
                        r[0], r[1], dtype=torch.long, device=vllm_embedding.device
                    )
                    span = gene_indices.numel()
                    if span != cur_gene_hs.shape[0]:
                        raise ValueError(
                            f"[GeneSpanMismatch] gene span={span}, gene tokens={cur_gene_hs.shape[0]} "
                            f"(expect 32). Check processor placeholder length."
                        )

                    cur_vllm_emb = new_vllm_embedding[i]
                    new_vllm_embedding[i] = cur_vllm_emb.scatter(
                        0,
                        gene_indices.view(-1, 1).repeat(1, cur_vllm_emb.shape[-1]),
                        cur_gene_hs.to(cur_vllm_emb.dtype)  # [32, embed_dim]
                    )

        if _is_debug_enabled():
            if "gene_bound" in data and len(data["gene_bound"]) > 0 and len(data["gene_bound"][0]) > 0:
                gb0 = data["gene_bound"][0]
                print("[DEBUG] gene_bound[0]:", gb0)
                print("[DEBUG] gene_span[0]:", gb0[0][1] - gb0[0][0])

        _assert_finite(new_vllm_embedding, "new_vllm_embedding(final inputs_embeds)")

        return new_vllm_embedding, vision_hidden_states

    def forward(self, data, **kwargs):
        vllm_embedding, vision_hidden_states = self.get_vllm_embedding(data)

        position_ids = data["position_ids"]
        if position_ids.dtype != torch.int64:
            position_ids = position_ids.long()

        for key in ['input_ids', 'inputs_embeds', 'position_ids']:
            if key in kwargs:
                del kwargs[key]

        return self.llm(
            input_ids=None,
            position_ids=position_ids,
            inputs_embeds=vllm_embedding,
            **kwargs
        )

    def _decode(self, inputs_embeds, tokenizer, attention_mask, decode_text=False, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        output = self.llm.generate(
            inputs_embeds=inputs_embeds,
            pad_token_id=0,
            eos_token_id=terminators,
            attention_mask=attention_mask,
            **kwargs
        )
        if decode_text:
            return self._decode_text(output, tokenizer)
        return output

    def _decode_stream(self, inputs_embeds, tokenizer, **kwargs):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        streamer = TextIteratorStreamer(tokenizer=tokenizer)
        generation_kwargs = {
            'inputs_embeds': inputs_embeds,
            'pad_token_id': 0,
            'eos_token_id': terminators,
            'streamer': streamer
        }
        generation_kwargs.update(kwargs)

        thread = Thread(target=self.llm.generate, kwargs=generation_kwargs)
        thread.start()
        return streamer

    def _decode_text(self, result_ids, tokenizer):
        terminators = [tokenizer.convert_tokens_to_ids(i) for i in self.terminators]
        result_text = []
        for result in result_ids:
            result = result[result != 0]
            if result[0] == tokenizer.bos_id:
                result = result[1:]
            if result[-1] in terminators:
                result = result[:-1]
            result_text.append(tokenizer.decode(result).strip())
        return result_text

    def generate(
        self,
        input_ids=None,
        pixel_values=None,
        tgt_sizes=None,
        image_bound=None,
        gene_input_ids=None,
        gene_attention_mask=None,
        gene_bound=None,
        attention_mask=None,
        tokenizer=None,
        vision_hidden_states=None,
        return_vision_hidden_states=False,
        stream=False,
        decode_text=False,
        **kwargs
    ):
        assert input_ids is not None
        if pixel_values is not None:
            assert len(input_ids) == len(pixel_values)
        if gene_input_ids is not None:
            assert len(input_ids) == len(gene_input_ids)

        model_inputs = {
            "input_ids": input_ids,
            "image_bound": image_bound,
            "gene_input_ids": gene_input_ids,
            "gene_attention_mask": gene_attention_mask,
            "gene_bound": gene_bound,
        }

        if vision_hidden_states is None:
            model_inputs["pixel_values"] = pixel_values
            model_inputs['tgt_sizes'] = tgt_sizes
        else:
            model_inputs["vision_hidden_states"] = vision_hidden_states

        with torch.inference_mode():
            model_inputs["inputs_embeds"], vision_hidden_states = self.get_vllm_embedding(model_inputs)

            if stream:
                result = self._decode_stream(model_inputs["inputs_embeds"], tokenizer, **kwargs)
            else:
                result = self._decode(
                    model_inputs["inputs_embeds"],
                    tokenizer,
                    attention_mask,
                    decode_text=decode_text,
                    **kwargs
                )

        if return_vision_hidden_states:
            return result, vision_hidden_states
        return result

    def chat(
        self,
        msgs,
        tokenizer,
        image=None,
        gene_sequence=None,
        processor=None,
        vision_hidden_states=None,
        max_new_tokens=2048,
        min_new_tokens=0,
        sampling=True,
        max_inp_length=12000,
        system_prompt='',
        stream=False,
        max_slice_nums=None,
        use_image_id=None,
        **kwargs
    ):
        if isinstance(msgs[0], list):
            batched = True
        else:
            batched = False

        msgs_list = msgs
        images_list = image
        gene_sequences_list = gene_sequence

        if batched is False:
            images_list, msgs_list = [images_list], [msgs_list]
            gene_sequences_list = [gene_sequences_list]
        else:
            assert images_list is None, "Please integrate image to msgs when using batch inference."
            images_list = [None] * len(msgs_list)

        assert len(images_list) == len(msgs_list), "The batch dim of images_list and msgs_list should be the same."

        if processor is None:
            if self.processor is None:
                self.processor = AutoProcessor.from_pretrained(self.config._name_or_path, trust_remote_code=True)
            processor = self.processor

        assert self.config.query_num == processor.image_processor.image_feature_size
        assert self.config.patch_size == processor.image_processor.patch_size
        assert self.config.use_image_id == processor.image_processor.use_image_id
        assert self.config.slice_config.max_slice_nums == processor.image_processor.max_slice_nums
        assert self.config.slice_mode == processor.image_processor.slice_mode

        prompts_lists = []
        input_images_lists = []
        input_gene_sequences_lists = []

        for image, gene_seq, msgs in zip(images_list, gene_sequences_list, msgs_list):
            if isinstance(msgs, str):
                msgs = json.loads(msgs)
            copy_msgs = deepcopy(msgs)

            assert len(msgs) > 0, "msgs is empty"
            assert sampling or not stream, "if use stream mode, make sure sampling=True"

            content_raw = copy_msgs[0]["content"]
            new_content = []
            if image is not None:
                new_content.append(image)
            if gene_seq is not None:
                new_content.append(gene_seq)
            if isinstance(content_raw, str):
                new_content.append(content_raw)
            elif isinstance(content_raw, list):
                new_content.extend(content_raw)
            copy_msgs[0]["content"] = new_content

            images_in_msg = []
            gene_in_msg = []
            for i, msg in enumerate(copy_msgs):
                role = msg["role"]
                content = msg["content"]
                assert role in ["user", "assistant"]
                if i == 0:
                    assert role == "user", "The role of first msg should be user"
                if not isinstance(content, list):
                    content = [content]

                cur_msgs = []
                for c in content:
                    if isinstance(c, Image.Image):
                        images_in_msg.append(c)
                        cur_msgs.append("(<image>./</image>)")
                    elif isinstance(c, ad.AnnData):
                        gene_in_msg.append(c)
                        cur_msgs.append("(<gene>./</gene>)")
                    elif isinstance(c, str):
                        cur_msgs.append(c)
                    else:
                        raise TypeError(f"Unsupported content type: {type(c)}")

                msg["content"] = "\n".join(cur_msgs)

            if system_prompt:
                sys_msg = {'role': 'system', 'content': system_prompt}
                copy_msgs = [sys_msg] + copy_msgs

            prompts_lists.append(
                processor.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True)
            )
            input_images_lists.append(images_in_msg)
            input_gene_sequences_lists.append(gene_in_msg)

        inputs = processor(
            prompts_lists,
            input_images_lists,
            input_gene_sequences_lists,
            max_slice_nums=max_slice_nums,
            use_image_id=use_image_id,
            return_tensors="pt",
            max_length=max_inp_length
        ).to(self.device)

        if sampling:
            generation_config = {
                "top_p": 0.8,
                "top_k": 100,
                "temperature": 0.7,
                "do_sample": True,
                "repetition_penalty": 1.05
            }
        else:
            generation_config = {
                "num_beams": 3,
                "repetition_penalty": 1.2,
            }

        if min_new_tokens > 0:
            generation_config['min_new_tokens'] = min_new_tokens

        generation_config.update((k, kwargs[k]) for k in generation_config.keys() & kwargs.keys())

        inputs.pop("image_sizes")

        with torch.inference_mode():
            res = self.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                vision_hidden_states=vision_hidden_states,
                stream=stream,
                decode_text=True,
                **generation_config
            )

        if stream:
            def stream_gen():
                for text in res:
                    for term in self.terminators:
                        text = text.replace(term, '')
                    yield text
            return stream_gen()
        else:
            if batched:
                answer = res
            else:
                answer = res[0]
            return answer
