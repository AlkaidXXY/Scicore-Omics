import os
import json
import torch
import anndata
import logging
from typing import Dict, List, Any, Optional
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, AutoProcessor
from swift.llm import (
    register_model, ModelMeta, ModelGroup, Model, register_model_arch,
    MultiModelKeys, get_model_tokenizer_with_flash_attn,
    register_template, Template, TemplateMeta
)
from swift.llm.template.utils import findall
from swift.utils import get_logger
import sys

logger = get_logger()

# ==============================================================================
# 0. Helpers: freeze switches (ENV controlled)
# ==============================================================================
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.environ.get(name, str(default)).strip().lower()
    return v in ("1", "true", "yes", "y", "on")

def _freeze_module(m, freeze: bool):
    if m is None:
        return
    for p in m.parameters():
        p.requires_grad = not freeze

# ==============================================================================
# 1. GeneTokenizer (原样保留)
# ==============================================================================
class GeneTokenizer:
    def __init__(self, vocab_file):
        if not os.path.exists(vocab_file):
            raise ValueError(f"Vocab file not found at {vocab_file}")
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f) # vocab is a dict: {"gene_name": id}

        self.gene_to_id = self.vocab
        # Define special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'

        # 确保特殊 token 存在
        if self.pad_token not in self.gene_to_id:
            pad_id = len(self.gene_to_id)
            self.gene_to_id[self.pad_token] = pad_id
        if self.unk_token not in self.gene_to_id:
            unk_id = len(self.gene_to_id)
            self.gene_to_id[self.unk_token] = unk_id

        self.pad_token_id = self.gene_to_id[self.pad_token]
        self.unk_token_id = self.gene_to_id[self.unk_token]

    def __call__(self, gene_list, max_length=None, padding=False, truncation=False):
        input_ids = [self.gene_to_id.get(gene, self.unk_token_id) for gene in gene_list]

        if max_length:
            if truncation and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]

            if padding:
                pad_len = max_length - len(input_ids)
                if pad_len > 0:
                    input_ids.extend([self.pad_token_id] * pad_len)

        return {"input_ids": input_ids}

# ==============================================================================
# 2. Template (原样保留，只做“增量增强”：让 gene span=32，并 mask labels)
# ==============================================================================
class MiniCPMGeneTemplate(Template):
    placeholder_tokens = ['<gene>']  # 原样保留

    def init_processor(self, processor):
        super().init_processor(processor)
        vocab_path = os.environ.get('GENE_VOCAB_PATH', '/data2/xiaoxinyu/project/model/gene_tokenizer/vocab.json')
        try:
            self.gene_tokenizer = GeneTokenizer(vocab_path)
            logger.info(f"Using GeneTokenizer from: {vocab_path}")
        except Exception as e:
            logger.warning(f"Failed to load GeneTokenizer: {e}. Gene data processing will fail.")
            self.gene_tokenizer = None

    def encode(self, data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        # 1) 取 gene 路径（原样）
        gene_path = data.get('gene')

        # 2) 父类处理文本（原样）
        encoded = super().encode(data, **kwargs)

        # 3) 原有 Gene 逻辑：读 h5ad -> var_names -> tokenize（原样）
        gene_input_ids = torch.empty(0, dtype=torch.long)
        gene_bound = []

        if gene_path and self.gene_tokenizer:
            try:
                if os.path.exists(gene_path):
                    adata = anndata.read_h5ad(gene_path)
                    gene_names = adata.var_names.tolist()

                    gene_tokens = self.gene_tokenizer(gene_names, max_length=1500)
                    gene_input_ids = torch.tensor(gene_tokens['input_ids'], dtype=torch.long)

                    input_ids = encoded['input_ids']
                    gene_str = '<gene>'

                    if gene_str in self.tokenizer.vocab or gene_str in self.tokenizer.get_vocab():
                        gene_token_id = self.tokenizer.convert_tokens_to_ids(gene_str)
                        idx_list = findall(input_ids, gene_token_id)
                        if idx_list:
                            start = idx_list[0]
                            end = start + 1
                            gene_bound = torch.tensor([[start, end]], dtype=torch.long)
                else:
                    logger.warning(f"Gene file not found: {gene_path}")

            except Exception as e:
                logger.error(f"Error processing gene file {gene_path}: {e}")

        # 4) ★增量：如果模型启用了 Q-Former，需要把 <gene> 扩成 32 token span
        #    通过环境变量控制：GENE_SPAN_LEN（默认 32）
        gene_span_len = int(os.environ.get("GENE_SPAN_LEN", "32"))
        use_qformer_span = _env_bool("USE_QFORMER_SPAN", True)  # 默认开启
        if use_qformer_span and isinstance(gene_bound, torch.Tensor) and gene_bound.numel() > 0:
            # 只处理第一个 span
            s, e = gene_bound[0].tolist()  # e = s+1
            if (e - s) != gene_span_len:
                # 将 input_ids 中位置 s 的单个 <gene> token 替换成 gene_span_len 个 <unk> token
                # 注意：这里不依赖 tokenizer 是否有 <gene_patch>，保持“最小入侵”
                input_ids = encoded["input_ids"]
                labels = encoded.get("labels", None)
                attn = encoded.get("attention_mask", None)

                # 用 <unk> token id 填充 span（保持你 processor 侧一致）
                unk_id = self.tokenizer.unk_token_id if hasattr(self.tokenizer, "unk_token_id") else self.tokenizer.convert_tokens_to_ids("<unk>")
                patch = torch.full((gene_span_len,), unk_id, dtype=input_ids.dtype)

                new_input_ids = torch.cat([input_ids[:s], patch, input_ids[e:]], dim=0)

                # labels：gene span 全 -100
                if labels is not None:
                    patch_l = torch.full((gene_span_len,), -100, dtype=labels.dtype)
                    new_labels = torch.cat([labels[:s], patch_l, labels[e:]], dim=0)
                    encoded["labels"] = new_labels

                if attn is not None:
                    patch_m = torch.ones((gene_span_len,), dtype=attn.dtype)
                    new_attn = torch.cat([attn[:s], patch_m, attn[e:]], dim=0)
                    encoded["attention_mask"] = new_attn

                encoded["input_ids"] = new_input_ids
                gene_bound = torch.tensor([[s, s + gene_span_len]], dtype=torch.long)

        # 5) 回填（原样）
        encoded['gene_input_ids'] = gene_input_ids
        encoded['gene_bound'] = gene_bound

        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        # 1) 父类处理（原样）
        res = super()._data_collator(batch, padding_to=padding_to)

        # 你原来的 tgt_sizes 修复（原样保留）
        if batch:
            tgt_sizes_list = [b.get('tgt_sizes') for b in batch if b.get('tgt_sizes') is not None]
            if tgt_sizes_list:
                try:
                    if isinstance(tgt_sizes_list[0], torch.Tensor):
                        res['tgt_sizes'] = torch.cat(tgt_sizes_list, dim=0)
                    elif isinstance(tgt_sizes_list[0], list):
                        res['tgt_sizes'] = torch.tensor(tgt_sizes_list)
                except Exception as e:
                    logger.warning(f"Failed to collate tgt_sizes: {e}")

        # 2) Gene Padding（原样）
        gene_ids_list = [b.get('gene_input_ids') for b in batch if b.get('gene_input_ids') is not None]
        has_genes = any(g.numel() > 0 for g in gene_ids_list)
        gene_pad_id = self.gene_tokenizer.pad_token_id if self.gene_tokenizer else 0

        if has_genes:
            max_len = max(len(g) for g in gene_ids_list if g.numel() > 0)
            padded_list = []
            for g in gene_ids_list:
                if g.numel() == 0:
                    padded_list.append(torch.full((max_len,), gene_pad_id, dtype=torch.long))
                else:
                    pad_size = max_len - len(g)
                    if pad_size > 0:
                        padded_list.append(torch.nn.functional.pad(g, (0, pad_size), value=gene_pad_id))
                    else:
                        padded_list.append(g)

            res['gene_input_ids'] = torch.stack(padded_list)
            res['gene_attention_mask'] = (res['gene_input_ids'] != gene_pad_id).long()
        else:
            res['gene_input_ids'] = None
            res['gene_attention_mask'] = None

        # 3) Gene Bound（原样）
        gene_bound_list = [b.get('gene_bound') for b in batch if b.get('gene_bound') is not None]
        if gene_bound_list:
            res['gene_bound'] = gene_bound_list

        return res

# 注册 Template（原样）
register_template(
    TemplateMeta(
        'minicpm_v2_6_gene',
        prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        suffix=['<|im_end|>'],
        template_cls=MiniCPMGeneTemplate
    )
)

# ==============================================================================
# 3. Model Loader（原样保留，增量加入 Q-Former & freeze switches）
# ==============================================================================
def get_model_tokenizer_minicpm_gene(model_dir, *args, **kwargs):
    if model_dir not in sys.path:
        print(f"🛠️ Injecting model_dir into sys.path: {model_dir}")
        sys.path.append(model_dir)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)

    # 添加调试信息
    print(f"🔍 Model modules: {[name for name, _ in model.named_modules()][:20]}")
    if hasattr(model, 'gene_qformer'):
        print("✅ Model has gene_qformer module")
    else:
        print("❌ Model does NOT have gene_qformer module")
    
    if hasattr(model, 'gene_projector'):
        print("✅ Model has gene_projector module")
    else:
        print("❌ Model does NOT have gene_projector module")
        
        
    # 2) 确保 <gene> token 在 tokenizer 中（原样）
    if '<gene>' not in tokenizer.vocab:
        tokenizer.add_tokens(['<gene>'], special_tokens=True)
        if hasattr(model, 'llm'):
            model.llm.resize_token_embeddings(len(tokenizer))
        else:
            model.resize_token_embeddings(len(tokenizer))

    # ----------------------------------------------------------------
    # ★增量：冻结开关（ENV 控制）
    #
    # 你希望运行脚本里类似：
    #   --freeze_vit true (Swift自带)
    #   --freeze_llm true (Swift自带)
    #   --freeze_qformer_project true/false
    #   --freeze_nicheformer true/false
    #
    # 由于 Swift CLI 未必支持新增 flag，推荐用环境变量：
    #   FREEZE_QFORMER_PROJECT=true
    #   FREEZE_NICHEFORMER=true
    #
    # ----------------------------------------------------------------
    freeze_qformer_project = _env_bool("FREEZE_QFORMER_PROJECT", False)  # 默认不冻结（即训练）
    freeze_nicheformer = _env_bool("FREEZE_NICHEFORMER", True)  

    # Nicheformer（原版逻辑：TUNE_NICHEFORMER 控制）
    if hasattr(model, 'nicheformer'):
        if freeze_nicheformer:
            logger.info("❄️ [Config] FREEZE_NICHEFORMER=true. Freezing Nicheformer.")
            model.nicheformer.requires_grad_(False)
        else:
            logger.info("🔥 [Config] FREEZE_NICHEFORMER=false. Unfreezing all layers of Nicheformer.")
            for layer in model.nicheformer.encoder.layers[-4:]:
                for param in layer.parameters():
                    param.requires_grad = True
    else:
        logger.warning("Nicheformer module not found in model.")

    # Gene Projector（原版：始终解冻；现在增量支持冻结开关）
    if hasattr(model, 'gene_projector'):
        _freeze_module(model.gene_projector, freeze_qformer_project)
        logger.info(f"{'❄️' if freeze_qformer_project else '🔥'} [Config] FREEZE_QFORMER_PROJECT={freeze_qformer_project} (gene_projector)")

    # 增量：Gene Q-Former 冻结/解冻
    if hasattr(model, 'gene_qformer'):
        _freeze_module(model.gene_qformer, freeze_qformer_project)
        logger.info(f"{'❄️' if freeze_qformer_project else '🔥'} [Config] FREEZE_QFORMER_PROJECT={freeze_qformer_project} (gene_qformer)")
    else:
        logger.warning("gene_qformer module not found in model (maybe loading non-qformer checkpoint).")

    # (E) Vision 部分：仍交给 Swift 的 --freeze_vit 处理（原样）

    # ========== 原 forward_wrapper 全保留 ==========
    _orig_forward = model.forward

    def forward_wrapper(self, data=None, **kwargs):
        if data is not None:
            return _orig_forward(data, **kwargs)

        data_pack = kwargs

        if 'input_ids' in data_pack:
            input_ids = data_pack['input_ids']
            bs, seq_len = input_ids.shape
        else:
            bs, seq_len = 1, 0

        if 'position_ids' not in data_pack or data_pack['position_ids'] is None:
            device = input_ids.device if 'input_ids' in data_pack else torch.device('cuda')
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            data_pack['position_ids'] = pos_ids.unsqueeze(0).expand(bs, -1)

        if 'pixel_values' not in data_pack or data_pack['pixel_values'] is None:
            data_pack['pixel_values'] = [[] for _ in range(bs)]

        if 'tgt_sizes' not in data_pack or data_pack['tgt_sizes'] is None:
            data_pack['tgt_sizes'] = [None] * bs

        if 'image_bound' not in data_pack or data_pack['image_bound'] is None:
            data_pack['image_bound'] = [[] for _ in range(bs)]

        return _orig_forward(data_pack, **data_pack)

    import types
    model.forward = types.MethodType(forward_wrapper, model)
    logger.info("🛠️ Successfully patched model.forward to accept **kwargs from Swift Trainer.")
    # ===============================================

    return model, tokenizer

# ==============================================================================
# 4. 注册模型架构：只增量加入 gene_qformer
# ==============================================================================
register_model_arch(
    MultiModelKeys(
        'minicpm_v2_6_gene',
        language_model=['llm'],
        vision_tower=['vpm','resampler'],
        # 原有 aligner 只有 gene_projector；现在增量加入 gene_qformer
        aligner=['gene_projector', 'gene_qformer'],
    )
)

register_model(
    ModelMeta(
        'minicpm_v2_6_gene',
        [
            ModelGroup([
                Model('openbmb/MiniCPM-V-2_6', 'openbmb/MiniCPM-V-2_6'),
            ]),
        ],
        'minicpm_v2_6_gene',
        get_model_tokenizer_minicpm_gene,
        is_multimodal=True,
        model_arch='minicpm_v2_6_gene',
    )
)
