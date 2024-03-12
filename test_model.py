from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, AutoModelForPreTraining
import torch


# gen = AutoModel.from_pretrained('tunib/electra-ko-en-base')
# dis = AutoModel.from_pretrained('google/electra-base-generator')

# from transformers import ElectraForPreTraining, ElectraTokenizerFast
# import torch

# # discriminator = ElectraForPreTraining.from_pretrained("google/electra-base-discriminator")
# tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-base-generator")

# text = "the quick brown [MASK] fake over the lazy dog"
# input_ids = tokenizer(text)

# output = gen(text)
# print(output)
# quit()


### 그냥 떼고 Electra weight -> Bert에 붓기


tunib = AutoModel.from_pretrained('tunib/electra-ko-en-base')
model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')\


# print(len(tunib.state_dict().keys())) # 197
# print(len(model.state_dict().keys())) # 204

tunib_dict = tunib.state_dict().copy()
bert_dict = model.state_dict().copy()

print(bert_dict['bert.embeddings.word_embeddings.weight'].shape) # 30522, 768
print(tunib_dict['embeddings.word_embeddings.weight'].shape) # 61790, 768

quit()


# tunib_keys = tunib_dict.keys()

# for key in tunib_dict.keys(): # 이렇게 쓰면 iterate 하면서는 modify 불가능. keys()가 copy를 return하는 게 아니기 때문.
for key, value in list(tunib_dict.items()):
    new_key = "bert." + key
    tunib_dict[new_key] = tunib_dict[key]
    del tunib_dict[key]

'''
        - Missing key(s) in state_dict: "cls.predictions.bias", "cls.predictions.transform.dense.weight", "cls.predictions.transform.dense.bias", "cls.predictions.transform.LayerNorm.weight", "cls.predictions.transform.LayerNorm.bias", "cls.predictions.decoder.weight", "cls.predictions.decoder.bias".
        
        - size mismatch for bert.embeddings.word_embeddings.weight: copying a param with shape torch.Size([61790, 768]) from checkpoint, the shape in current model is torch.Size([30522, 768]).
'''
model.load_state_dict(tunib_dict)




quit()

# sentence = "The quick brown fox jumps over the lazy dog"
# fake_sentence = "The quick brown fox fake over the lazy dog"

# fake_tokens = tokenizer.tokenize(fake_sentence)
# fake_inputs = tokenizer.encode(fake_sentence, return_tensors="pt")
# discriminator_outputs = dis(fake_inputs)

# # print(discriminator_outputs) # 

# predictions = torch.round((torch.sign(discriminator_outputs[0]) + 1) / 2)

# [print("%7s" % token, end="") for token in fake_tokens]
# [print("%7s" % int(prediction), end="") for prediction in predictions.tolist()]






from transformers import pipeline

fill_mask = pipeline(
    "fill-mask",
    model="google/electra-base-generator",
    tokenizer="google/electra-base-generator"
)

print(
    fill_mask(f"HuggingFace is creating a {fill_mask.tokenizer.mask_token} that the community uses to solve NLP tasks.")
)

quit()



'''
비교해야 하는 것 (layer nubmer 출력하고, layer list, 모델 구조 출력)


- ElectraForPretraining
- AutoModel (tunib)
- ElectraForMaskedLM

- AutoModel (Bert)
- BertForPretraining


'''

arch_list = [
    'AutoModelForPreTraining',
    'AutoModel',
    'AutoModelForMaskedLM'
]

model_id = [
    "tunib/electra-ko-en-base",
    "bert-base-uncased"
]

item_list = []
for id in model_id:

    print(f"id: {id}")

    arch_item = []
    for arch in arch_list:
        
        print(f"architecutre: {arch}")
        model = eval(arch).from_pretrained(id)
        model_keys = model.state_dict().keys()
        print(model)
        print(f"layer num: {len(model_keys)}")
        for name, value in model.state_dict().items():
            print(name)
            
        print("=======================================")
        
    print("==================================================================")
    print("==================================================================")
    print("==================================================================")

quit()

# tunib_electra = AutoModel.from_pretrained('tunib/electra-ko-en-base')

# tunib_electra_mlm = AutoModelForMaskedLM.from_pretrained('tunib/electra-ko-en-base')

# tunib_electra_pre = AutoModelForPreTraining.from_pretrained('tunib/electra-ko-en-base')


# tunib_electra_mlm_keys = tunib_electra_mlm.state_dict().keys()


# electra = AutoModelForPreTraining.from_pretrained('tunib/electra-ko-en-base')

# print(electra)
# quit()





# google_electra_dis = AutoModel.from_pretrained('google/electra-base-discriminator')
# google_electra_gen = AutoModel.from_pretrained('google/electra-base-generator')

bert = AutoModel.from_pretrained('bert-base-uncased')

# # print(type(bert))
# print(bert)
# quit()

bert_ = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')



electra_keys = tunib_electra.state_dict().keys()
bert_keys = bert.state_dict().keys()

bert_keys_ = bert_.state_dict().keys()

print("bert: ", len(bert_keys))
print(bert_keys)

print("=========================================")
print("bert for masekd lm: ", len(bert_keys_))
print(bert_keys_)

print("=========================================")
print("electra: ", len(electra_keys))
print(electra_keys)

quit()


# print(tunib_electra.state_dict().keys())



# # print(type(electra)) # ElectraModel
# print(type(bert)) # BertModel


# print("======================= tunib =======================") # discriminator 
# print(tunib_electra)

# # print(bert)
# print("===================== discriminator ========================")
# print(google_electra_dis)

# print("===================== generator ========================")
# print(google_electra_gen)


# # for m in bert.modules():
# #     for name, value in list(m.named_parameters(recurse=False)):
# #         # print(f"{name}:\t{value}")
# #         print(name)



# ### test for comparing output
# # from Condenser.modeling import CondenserForPretraining

# # bert_condenser = CondenserForPretraining.from_pretrained(
    
# # )





print(len(electra_keys)) # 235
print(len(bert_keys)) # 197 