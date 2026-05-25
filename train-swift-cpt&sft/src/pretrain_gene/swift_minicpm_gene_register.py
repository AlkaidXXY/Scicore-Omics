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
import os
logger = get_logger()

# ==============================================================================
# 1. GeneTokenizer (直接从你的 gene_tokenizer.py 迁移过来)
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
        
        # 确保特殊 token 存在 (逻辑保持和你的一致)
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
# 2. Template (对应你的 dataset.py 中的 Dataset 和 DataCollator)
# ==============================================================================
class MiniCPMGeneTemplate(Template):
    placeholder_tokens = ['<gene>'] # 对应你代码里的 <gene>

    def init_processor(self, processor):
        super().init_processor(processor)
        # 初始化 GeneTokenizer
        # [配置项] 请修改为你实际的 vocab 路径，或者通过环境变量传入
        vocab_path = os.environ.get('GENE_VOCAB_PATH', '/data2/xiaoxinyu/project/model/gene_tokenizer/vocab.json')
        try:
            self.gene_tokenizer = GeneTokenizer(vocab_path)
            logger.info(f"Using GeneTokenizer from: {vocab_path}")
        except Exception as e:
            logger.warning(f"Failed to load GeneTokenizer: {e}. Gene data processing will fail.")
            self.gene_tokenizer = None

    def encode(self, data: Dict[str, Any],**kwargs) -> Dict[str, Any]:
        """
        重写入口函数 encode。
        参数 data 是原始字典，包含 {'query':..., 'response':..., 'gene':...}
        """
        
        # 1. 【先】从原始数据里把 gene 路径取出来
        gene_path = data.get('gene')
        
        # 2. 【再】调用父类逻辑处理文本和图片
        # 这一步会生成 input_ids, labels, 并且会自动读取图片
        # 如果图片路径不对，报错会在这里发生
        
        encoded = super().encode(data,**kwargs)
        
        # 3. 【后】手动处理 Gene 逻辑
        gene_input_ids = torch.empty(0, dtype=torch.long)
        gene_bound = []

        if gene_path and self.gene_tokenizer:
            try:
                if os.path.exists(gene_path):
                    # 读取 h5ad
                    adata = anndata.read_h5ad(gene_path)
                    gene_names = adata.var_names.tolist()
                    
                    # Tokenize
                    gene_tokens = self.gene_tokenizer(gene_names, max_length=1500)
                    gene_input_ids = torch.tensor(gene_tokens['input_ids'], dtype=torch.long)
                    
                    # 寻找 <gene> 的位置 (在 input_ids 里找)
                    # 注意：此时父类已经把 <gene> 转换成 token id 了
                    input_ids = encoded['input_ids']
                    gene_str = '<gene>'
                    
                    # 确保 <gene> 在词表里
                    if gene_str in self.tokenizer.vocab or '<gene>' in self.tokenizer.get_vocab():
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

        # 4. 把处理好的 gene 数据塞回结果字典
        encoded['gene_input_ids'] = gene_input_ids
        encoded['gene_bound'] = gene_bound
        
        return encoded

    def _data_collator(self, batch: List[Dict[str, Any]], *, padding_to: Optional[int] = None) -> Dict[str, Any]:
        """Batch 组装逻辑"""
        # 1. 父类处理 Text/Image padding
        res = super()._data_collator(batch, padding_to=padding_to)
        
        # =================================================================
        # ★★★ 修复点：手动收集 tgt_sizes ★★★
        # MiniCPM-V 2.6 需要这个参数来处理动态分辨率
        # =================================================================
        if batch:
            # 检查 batch 里有没有 tgt_sizes
            tgt_sizes_list = [b.get('tgt_sizes') for b in batch if b.get('tgt_sizes') is not None]
            
            if tgt_sizes_list:
                # 只有当存在有效图片数据时，才会有 tgt_sizes
                # 通常它是一个 Tensor，我们需要把它拼成一个 Batch Tensor
                try:
                    import torch
                    # 如果已经是 Tensor，用 cat 或 stack
                    if isinstance(tgt_sizes_list[0], torch.Tensor):
                        res['tgt_sizes'] = torch.cat(tgt_sizes_list, dim=0)
                    # 如果是 list，转 Tensor
                    elif isinstance(tgt_sizes_list[0], list):
                        res['tgt_sizes'] = torch.tensor(tgt_sizes_list)
                except Exception as e:
                    logger.warning(f"Failed to collate tgt_sizes: {e}")

        # =================================================================
        # 2. 处理 Gene Padding (保持你原有的逻辑不变)
        # =================================================================
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

        # 3. 处理 Gene Bound
        gene_bound_list = [b.get('gene_bound') for b in batch if b.get('gene_bound') is not None]
        if gene_bound_list:
             res['gene_bound'] = gene_bound_list
             
        return res

# 注册 Template
register_template(
    TemplateMeta(
        'minicpm_v2_6_gene',
        # MiniCPM V 2.6 的标准 Chat 格式
        prefix=['<|im_start|>system\n{{SYSTEM}}<|im_end|>\n'],
        prompt=['<|im_start|>user\n{{QUERY}}<|im_end|>\n<|im_start|>assistant\n'],
        chat_sep=['<|im_end|>\n'],
        suffix=['<|im_end|>'],
        template_cls=MiniCPMGeneTemplate
    )
)


# ==============================================================================
# 3. Model Loader (对应 finetune.py 的加载和冻结逻辑)
# ==============================================================================
def get_model_tokenizer_minicpm_gene(model_dir, *args, **kwargs):
    # 1. 正常加载模型
    if model_dir not in sys.path:
        print(f"🛠️ Injecting model_dir into sys.path: {model_dir}")
        sys.path.append(model_dir)
    model, tokenizer = get_model_tokenizer_with_flash_attn(model_dir, *args, **kwargs)
    
    # 2. 确保 <gene> token 在 tokenizer 中 (对应 dataset.py __init__)
    if '<gene>' not in tokenizer.vocab:
        tokenizer.add_tokens(['<gene>'], special_tokens=True)
        # 调整 embedding 大小
        if hasattr(model, 'llm'):
            model.llm.resize_token_embeddings(len(tokenizer))
        else:
             model.resize_token_embeddings(len(tokenizer))

    # 3. 复现 finetune.py 的冻结/解冻逻辑
    # ----------------------------------------------------------------
    
    # (A) 处理 Nicheformer (Gene Encoder)
    # finetune.py 逻辑: tune_nicheformer 默认为 False -> 冻结
    # 我们可以通过环境变量 TUNE_NICHEFORMER 控制
    tune_nicheformer = os.environ.get('TUNE_NICHEFORMER', 'false').lower() == 'true'
    
    if hasattr(model, 'nicheformer'):
        if tune_nicheformer:
            logger.info("🔥 [Config] TUNE_NICHEFORMER=true. Unfreezing all layers of Nicheformer.")
            # 对应 finetune.py: 解冻 encoder 
            for layer in model.nicheformer.encoder.layers:
                for param in layer.parameters():
                    param.requires_grad = True
        else:
            logger.info("❄️ [Config] TUNE_NICHEFORMER=false. Freezing Nicheformer.")
            model.nicheformer.requires_grad_(False)
    else:
        logger.warning("Nicheformer module not found in model.")

    # (B) 处理 Gene Projector
    # finetune.py 逻辑: 始终解冻 gene_projector
    if hasattr(model, 'gene_projector'):
        logger.info("🔥 [Config] Unfreezing gene_projector.")
        model.gene_projector.requires_grad_(True)

    # (C) 处理 Vision (VPM)
    # Swift 框架会根据 --freeze_vit 参数自动处理 vision_tower 列表里的模块
    # 我们只需要在 register_model_arch 里配好即可
    _orig_forward = model.forward

    # 2. 定义新的兼容 wrapper
    def forward_wrapper(self, data=None, **kwargs):
        # 情况 A: 正常调用
        if data is not None:
            return _orig_forward(data, **kwargs)
        
        # 情况 B: Swift Trainer 调用
        data_pack = kwargs
        
        # 1. 获取基础信息
        if 'input_ids' in data_pack:
            input_ids = data_pack['input_ids']
            bs, seq_len = input_ids.shape
        else:
            # 极端防御
            bs, seq_len = 1, 0
            
        # -----------------------------------------------------------
        # ★★★ 新增修复：构造 position_ids ★★★
        # -----------------------------------------------------------
        if 'position_ids' not in data_pack or data_pack['position_ids'] is None:
            # 生成简单的位置编码: [0, 1, 2, ..., seq_len-1]
            # 并扩展到整个 batch: [bs, seq_len]
            device = input_ids.device if 'input_ids' in data_pack else torch.device('cuda')
            
            # 创建 [0, 1, ..., seq_len-1]
            pos_ids = torch.arange(seq_len, dtype=torch.long, device=device)
            # 扩展为 [batch_size, seq_len]
            data_pack['position_ids'] = pos_ids.unsqueeze(0).expand(bs, -1)

        # -----------------------------------------------------------
        # 2. 修复 pixel_values (必须是 list of lists)
        if 'pixel_values' not in data_pack or data_pack['pixel_values'] is None:
            data_pack['pixel_values'] = [[] for _ in range(bs)]
            
        # 3. 修复 tgt_sizes
        if 'tgt_sizes' not in data_pack or data_pack['tgt_sizes'] is None:
            data_pack['tgt_sizes'] = [None] * bs 
            
        # 4. 修复 image_bound
        if 'image_bound' not in data_pack or data_pack['image_bound'] is None:
            data_pack['image_bound'] = [[] for _ in range(bs)]
        
        
            
        return _orig_forward(data_pack,**data_pack)

    # 3. 将新函数绑定到模型实例上
    import types
    model.forward = types.MethodType(forward_wrapper, model)
    
    logger.info("🛠️ Successfully patched model.forward to accept **kwargs from Swift Trainer.")
    # =================================================================

    return model, tokenizer

# ==============================================================================
# 4. 注册模型架构和元数据
# ==============================================================================

# 这里的配置决定了 LoRA 挂哪里，以及 freeze_vit 冻结谁
register_model_arch(
    MultiModelKeys(
        'minicpm_v2_6_gene',
        # LLM 部分，Swift 会在这里挂 LoRA (对应 finetune.py: lora_target_modules)
        language_model=['llm'], 
        
        # 视觉塔 (对应 finetune.py: tune_vision=False 时冻结 model.vpm)
        # 将 vpm 放入此处，使用 --freeze_vit true 即可冻结它
        vision_tower=['vpm','resampler'],
        
        # 对齐层 (对应 finetune.py: 始终解冻 gene_projector, resampler)
        # 放入这里可以防止被意外冻结，且通常不需要加 LoRA
        aligner=[ 'gene_projector'],
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