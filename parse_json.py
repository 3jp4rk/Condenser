import time
import json 
import ijson

def _check(doc_list, doc_id):
    found = False
    target_idx = 0
    for idx, doc in enumerate(doc_list):
        if doc['doc_id'] == doc_id:
            found = True
            target_idx = idx
            break
    return found, target_idx

def get_doc_idx(file_path): #, lang):
    
    # key_list = ['passage_id', 'offset', 'length', 'line_no', 'doc_id']
    # doc_list = []
    # doc_dict = {}
    # passage_dict = {}
    
    counter = 1
    with open(file_path, 'r') as file:
        parser = ijson.parse(file)
        for prefix, event, value in parser:
            passage_item = []
            if counter == 2:
                break
            if event == 'map_key':
                key = value
            elif event == 'start_array':
                items = []
                for idx, value in enumerate(parser):  # Iterate over the values in the array
                    if value[1] == 'end_array':
                        break               
                    ###
                    items.append(value[2])
                    # passage_item.append(value[2])
                    # if idx == 3: # doc_id
                    #     print(key, '\t', passage_item)
                    #     '''
                    #     doc_id = value[2]
                    #     if doc_id in doc_dict:
                    #         doc_dict[doc_id].append(passage_item)
                    #     else:
                    #         doc_dict[doc_id] = [passage_item]
                    #     '''
                counter += 1
                print(key)
                print(items)
                


def get_idx(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

def create_new_index(json_dict, lang):
    
    doc_id_dict = {}
    counter = 0
    with open(f'new_json_{lang}.json', 'w') as newfile:
        for k, v in json_dict.items():
            offset, length, line_no, doc_id = v
            if doc_id in doc_id_dict:
                doc_id_dict[doc_id].append((k, v))
            else:
                doc_id_dict[doc_id] = [(k, v)]
            counter += 1
            if counter % 5000 == 1:
                print(len(doc_id_dict[doc_id]))
        json.dump(doc_id_dict, newfile, ensure_ascii=False)
        
def test():
    
    target = tuple()
    found = False
    
    return (found, target)

def test(sample, items):
    
    for i in items:
        if str(i) in sample:
            sample[str(i)] = i+1
        else:
            sample[str(i)] = i

    # return sample

if __name__ == "__main__":
    
    # get_doc_idx('./new_index_diet.json')
    # print("ko: ")
    # get_doc_idx('./msmarco/ko_index.json')
    # print("================================")
    # print("en: ")
    # get_doc_idx('./msmarco/en_index.json')
    
    with open('./msmarco/ko_index.json', 'rb') as f:
        data = json.load(f)
    
    
    print(data.get("38_283710562"))
    
    # new_dict = {}
    # items = [[1, 2, 3, 4], [3, 4, 5, 6]]
    
    
    
    # for item in items:
    #     test(new_dict, item)
    
    # print(new_dict)
    quit()

    a = {
        "lang": "en",
        "data": {
            'k': [1, 2, 3, 4],
            'p': [1, 2, 3, 4],
        }
    }
    
    b = {
        "lang": "ko",
        "data": {
            'j': [1, 2, 3, 4],
            'h': [1, 2, 3, 4],
        }
    }
    
    
    new_dict = {}
    
    for idx, origin in enumerate(a["data"].items()):

        k, v = origin
        new_item = (k, (v[0], v[1]))
        
        if 'test' in new_dict:
            if a["lang"] in new_dict['test']:
                new_dict['test'][a["lang"]].append(new_item)
        else:
            new_dict['test'] = {
                f"{a['lang']}": [new_item]
            }
        
    print(new_dict)    

    # a, b = test()
    # print(a, b)    
    
    # t = a["data"].items()
    # for idx in t:
    #     print(idx)
    # # print(t)
    # # print(list(t))
    
    
    quit()
    
    
    
    
    
    
    
    for idx, (k, v) in enumerate(a["data"].items()):
        
        print(f"a item at idx {idx}\t{k}: {v}")
        
        k_, v_ = list(b["data"].items())[idx]
        # k_, v_ = other_pair
        
        print(f"b item at idx {idx}\t{k_}: {v_}")
        
        
    quit()
    
    # create_new_index('msmarco/ko_index.json')
    # sample = {'a': [ 1, 2, 3, 4], 'b': [ 5, 6, 7, 8]}
    # sample2 = {'c': [9, 10, 11, 12], 'd': [13, 14, 15, 16]}
    # sample_ = {}
    # sample_['en'] = sample
    # sample_['ko'] = sample2
    
    # for k, v in sample_['en'].items():
    #     print(k)
    #     print(v)
    langs = ['en', 'ko']
    for lang in langs:
        path = f'msmarco/{lang}_index.json'
        path = f'./new_json_{lang}.json'
        a = get_doc_idx(path,lang)

        print(f'working on {lang}. . . ')
        start = time.time()
        data_idx = get_idx(path)
        print(f'loading took: {time.time() - start} seconds. \n')
        
        start = time.time()
        create_new_index(data_idx, lang)
        print(f'creating took: {time.time() - start} seconds. ')
    
    
    
    
