import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig

tokenizer_ours = AutoTokenizer.from_pretrained(
    './merged_tokenizer/merged_tokenizer/', 
    local_files_only=True) # LlamaTokenizerFast

