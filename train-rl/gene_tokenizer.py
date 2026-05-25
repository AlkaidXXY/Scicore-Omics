
import json
import os

class GeneTokenizer:
    def __init__(self, vocab_file):
        if not os.path.exists(vocab_file):
            raise ValueError(f"Vocab file not found at {vocab_file}")
        with open(vocab_file, 'r') as f:
            self.vocab = json.load(f) # vocab is a dict: {"gene_name": id}
        
        self.gene_to_id = self.vocab
        self.id_to_gene = {v: k for k, v in self.vocab.items()}
        
        # Define special tokens
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        
        if self.pad_token not in self.gene_to_id:
            # Add special tokens if they don't exist
            pad_id = len(self.gene_to_id)
            self.gene_to_id[self.pad_token] = pad_id
            self.id_to_gene[pad_id] = self.pad_token

        if self.unk_token not in self.gene_to_id:
            unk_id = len(self.gene_to_id)
            self.gene_to_id[self.unk_token] = unk_id
            self.id_to_gene[unk_id] = self.unk_token
            
        self.pad_token_id = self.gene_to_id[self.pad_token]
        self.unk_token_id = self.gene_to_id[self.unk_token]

    def __call__(self, gene_list, max_length=None, padding=False, truncation=False):
        """
        Tokenizes a list of gene names.
        """
        input_ids = [self.gene_to_id.get(gene, self.unk_token_id) for gene in gene_list]
        
        attention_mask = [1] * len(input_ids)

        if max_length:
            if truncation and len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            
            if padding:
                pad_len = max_length - len(input_ids)
                if pad_len > 0:
                    input_ids.extend([self.pad_token_id] * pad_len)
                    attention_mask.extend([0] * pad_len)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

    @property
    def vocab_size(self):
        return len(self.gene_to_id)
