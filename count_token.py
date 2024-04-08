import os
import struct
from torch.utils.data import Dataset

def unescape_csv(bdata) :
    bcopy = bytearray()
    dquote = False

    for i, b in enumerate(bdata) :
        if b == 34 : # '"' is chr(34)
            if i == 0 or i == len(bdata) - 1 :
                continue
            elif dquote :
                dquote = False
                continue
            else :
                dquote = True
        elif dquote :
            dquote = False
        bcopy.append(b)

    return bcopy.decode('utf-8')

def get_csv_len(fname) :
    idx_name = fname + '.idx'
    fsize = os.stat(idx_name).st_size
    return int((fsize - 8) / 8) - 1

def get_csv_item(fname, idx, len) :
    try :
        idx_name = fname + '.idx'
        with open(idx_name, 'rb') as h :
            h.seek((idx + 1) * 8)
            start = struct.unpack('>Q', h.read(8))[0]
            if idx == len - 1 :
                end = -1
            else :
                h.seek((idx + 2) * 8)
                end = struct.unpack('>Q', h.read(8))[0]
        # print(f'start : { start }, end : { end }')
        with open(fname, 'rb') as h :
            h.seek(start)
            if end >= 0 :                row_data = h.read(end - start)
            else :
                row_data = h.read(os.stat(fname).st_size - start)
        return unescape_csv(row_data[:-1])
    except FileNotFoundError as e :
        print(f'fname : {fname}, idx: {idx}, len: {len}, error : {str(e)} ')
        raise e
    except struct.error as e :
        print(f'fname : {fname}, idx: {idx}, len: {len}, error : {str(e)} ')
        raise e


def find_dataset(idx) :
    for ds in reversed(data_order) :
        if data_start[ds] <= idx :
            return ds

def is_in_csv_data(idx) :
    return idx >= data_start['kowiki']

def get_non_csv(ds, idx) :
    fname = ds + '/' + str(idx - data_start[ds]) + '.txt'
    with open(fname, mode='r', encoding='utf-8') as f :
        return f.read()

def get_csv_fname(ds, ds_pos) :
    if ds == 'kowiki' :
        return f'kowiki/part_{{num:05d}}.csv'.format(num=ds_pos)
    elif ds == 'culko' :
        return f'CulturaX/ko/ko_part_{{num:05d}}.csv'.format(num=ds_pos)
    return f'CulturaX/en/en_part_{{num:05d}}.csv'.format(num=ds_pos)

def find_in_list(ds_list, idx) :
    for i, v in enumerate(ds_list) :
        idx -= v
        if idx < 0 :
            return i, idx + v

# en_ratio = { 0:0.7, 1:0.6, 2:0.5 , 3:0.6, 4:0.5 }
# en_ratio = { 0 : 0.65, 1: 0.5 }
en_ratio = { 0 : 0.3, 1: 0.2 }


if __name__ == "__main__":

    data_len = {}
    data_len['sciencedaily'] = 168329 # 0 ~ 168328
    data_len['namuwiki'] = 1041222 # 0 ~ 1041221
    data_len['newspepper'] = 7954 # 0 ~ 7953
    
    data_len_ko_txt = data_len['namuwiki'] + data_len['newspepper']
    data_len_en_txt = data_len['sciencedaily']
    data_len_txt = data_len_ko_txt + data_len_en_txt
    
    data_len['kowiki'] = []
    data_len['culko'] = []
    data_len['culen'] = []
    
    data_len_csv = 0
    
    data_len['kowiki'].clear()
    
    # 'kowiki/part_00000.csv' ~ 'kowiki/part_00003.csv'
    for i in range(4) :
        f = f'kowiki/part_{{num:05d}}.csv'.format(num=i)
        data_count = get_csv_len(f)
        data_len['kowiki'].append(data_count)
    
    print(data_len['kowiki'])
    
    data_len['culko'].clear()
    
    # 'CulturaX/ko/ko_part_00000.csv' ~ 'CulturaX/ko/ko_part_00031.csv'
    for i in range(32) :
        f = f'CulturaX/ko/ko_part_{{num:05d}}.csv'.format(num=i)
        data_count = get_csv_len(f)
        data_len['culko'].append(data_count)
    
    print(data_len['culko'])
    
    data_len['culen'].clear()
    
    # 'CulturaX/en/en_part_00000.csv' ~ 'CulturaX/en/en_part_00031.csv'
    # for i in range(3072) :
    for i in range(2460) :
        f = f'CulturaX/en/en_part_{{num:05d}}.csv'.format(num=i)
        try :
            data_count = get_csv_len(f)
            data_len['culen'].append(data_count)
        except FileNotFoundError :
            data_len['culen'].append(0)
    
    # print(data_len['culen'])
    
    # need to handle CulturaX en shuffling
    # the constants are the data count before CulturaX en (including CulturaX ko)
    # CulturaX en range should be different on each epochs
    # (won't randomize across epochs. Will use different sets on different epochs)
    
    print(f'txt data : {data_len_txt}, en txt data : { data_len_en_txt }, ko txt data : { data_len_ko_txt }')
    print(sum(data_len['kowiki']))
    print(sum(data_len['culko']))
    
    data_len_ko = data_len_ko_txt + sum(data_len['kowiki']) + sum(data_len['culko'])
    print(data_len_ko)
    
    data_len_en_except_culen = data_len_en_txt
    data_len_except_culen = data_len_txt + sum(data_len['kowiki']) + sum(data_len['culko'])
    print(f'total ko data : { data_len_ko }, current en data : { data_len_en_except_culen }, (ko - current en) data : { data_len_ko - data_len_en_except_culen }')
    
    def get_culen_size(en_ratio) :
        return int(data_len_ko * en_ratio / (1.0 - en_ratio) - data_len_en_except_culen)
    
    print(f'when 90% en data, cul en size should be { get_culen_size(0.9) }')
    print(f'when 80% en data, cul en size should be { get_culen_size(0.8) }')
    print(f'when 70% en data, cul en size should be { get_culen_size(0.7) }')
    print(f'when 60% en data, cul en size should be { get_culen_size(0.6) }')
    print(f'when 50% en data, cul en size should be { get_culen_size(0.5) }')
    
    
    # csv with idx
    #   - CulturaX/en/en_part_00000.csv ~ en_part_03071.csv
    #   - CulturaX/ko/ko_part_00000.csv ~ ko_part_00031.csv
    #   - kowiki/part_00000.csv ~ part_00003.csv
    # txt (idx is file name)
    #   - sciencedaily/0.txt ~ 168328.txt
    #   - namuwiki/0.txt ~ 1041292.txt
    #   - newspepper/0.txt ~ 7953.txt
    
    data_start = {}
    data_start['sciencedaily'] = 0
    data_start['namuwiki'] = data_len['sciencedaily']
    data_start['newspepper'] = data_start['namuwiki'] + data_len['namuwiki']
    
    data_start['kowiki'] = data_start['newspepper'] + data_len['newspepper']
    data_start['culko'] = data_start['kowiki'] + sum(data_len['kowiki'])
    data_start['culen'] = data_start['culko'] + sum(data_len['culko'])
    
    data_order = [ 'sciencedaily', 'namuwiki', 'newspepper', 'kowiki', 'culko', 'culen' ]


    root = "/root"
    ko_data = ["kowiki", "newspepper", "ready_for_zip/ko"]

    print("counting korean data total tokens . . . ")
    for data in ko_data:
        print(f"1. {data}")
        data_dir = os.path.join(root, data)


        
    


    # English data











