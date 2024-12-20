import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig


# empty model load
config = AutoConfig.from_pretrained('bert-base-uncased')
model = AutoModel.from_config(config)


# tokenizer load
tokenizer_ours = AutoTokenizer.from_pretrained('./merged_tokenizer/merged_tokenizer/', 
                                          local_files_only=True) # LlamaTokenizerFast

tokenizer_bert = AutoTokenizer.from_pretrained('bert-base-uncased')
print(len(tokenizer_bert)) # 30522

# resize token embeddings
print(len(tokenizer_ours)) # 57101 

model.resize_token_embeddings(len(tokenizer_ours))
print(model) # 이 상태에서 embedid

print("task done!")
quit()






# pretrained weight vs. empty weight (compared for checking if the empty model loaded successfully)
layer0_attention_query_weight_empty = (
model
._modules['encoder']
._modules['layer']
._modules['0']
._modules['attention']
._modules['self']
._modules['query']
._parameters['weight']
.detach()
.numpy())

model_pre = AutoModel.from_pretrained('bert-base-uncased')


layer0_attention_query_weight_pre = (
model_pre
._modules['encoder']
._modules['layer']
._modules['0']
._modules['attention']
._modules['self']
._modules['query']
._parameters['weight']
.detach()
.numpy())

# print(layer0_attention_query_weight_empty)
print("===================================")
# print(layer0_attention_query_weight_pre)




# resize token embedding










