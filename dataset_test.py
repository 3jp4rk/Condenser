
# from marco_dataset import ComplicatedDataset as NewDataset
from transformers import AutoTokenizer



def test():
    
    a = 1
    b = True
    
    return b, None

if __name__ == "__main__":
    
    import random
    # a = [1]
    # print(random.sample(a, 1))
    # quit()
    # # b, a = test()
    # # print(b)
    # print(a)
    
    # quit()
    
    
    
    
    
    
    
    tokenizer = AutoTokenizer.from_pretrained('tunib/electra-ko-en-base')
    
    # from enko_dataset import ComplicatedDataset
    # origin = ComplicatedDataset(train_ratio=0.9, tokenizer=tokenizer, max_length=512)
    
    # # print(origin.__getitem__(10))
    # origin.__getitem__(300000)
    
    
    my = {'a': [1, 2, 3, 4],
          'b': [2, 3, 4, 5],
          'c': [1, 2, 3, 4],
          'd': [1, 2, 3, 4],}
    
    
    subset = list(my.items())[:3]
    subset = dict(subset)
    
    print(subset)
    quit()
    
    # # k, v = list(my.items())[0]
    
    # print(list(my.items())[-1])
    
    # # print(k)
    # # print(v)
    
    # quit()
    
    
    # # print(list(my.values())[0][3])
    # # print(list(my.keys())[0])
    # quit()
    import time
    start = time.time()
    a = NewDataset(train_ratio=0.9, tokenizer=tokenizer, max_length=512)
    
    # print(a.train_data_start)
    
    # ds_key = ['ko']
    ds_key = ['en', 'ko'] 
    ds_key = ['ko-en-diff']
    d = a.train_data_start
    indicies = [a.train_data_start[ds_key[idx]] for idx in range(len(ds_key))]
    
    # p = a.train_data_start[ds_key[2]]
    # print(a.__getitem__(p))
    
    for i in indicies:
        print(a.__getitem__(i))


    # a = 1
    