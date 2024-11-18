from torch.nn.utils.rnn import pad_sequence
import torch
import random

"""
data_label_list form:

list[tuple(length, data, label)]

"""


def split_data_random(data_label_list, train_ratio=0.95):

    # 創建索引列表並隨機打亂
    indices = list(range(len(data_label_list)))
    random.shuffle(indices)

    # 計算分割點
    train_size = int(len(data_label_list) * train_ratio)

    # 分配數據
    train_indices = indices[:train_size]
    valid_indices = indices[train_size:]

    # 根據索引分割數據
    train_data_label_list = [data_label_list[i] for i in train_indices]
    valid_data_label_list = [data_label_list[i] for i in valid_indices]


    return train_data_label_list, valid_data_label_list



def sliding_window(data_label_list: list[tuple[list]], look_front: int, step=1):
    data_label_list = sorted(data_label_list, key=lambda x: x[0])
    
    data_list = []
    label_list = []
    length_list = []

    for l, data, label in data_label_list:
        for i in range(0, l, step):
            real_end = min(l, i + look_front)
            length = real_end - i
            data_list.append(data[i:real_end])
            label_list.append(label[i:real_end])
            length_list.append(length)
            if (length < look_front):
                break
        
    return data_list, label_list, length_list


def sort_by_length(data_label_list):


    data_label_list = sorted(data_label_list, key=lambda x: x[0])

    data_list = []
    label_list = []
    length = []

    for l, data, label in data_label_list:
        data_list.append(data)
        label_list.append(label)
        length.append(l)

    return data_list, label_list, length


def padding(data_list, label_list, length, batch=64):
    """
    將輸入進來的資料包成batch
    length不足就padding
    
    """
    

    batch_data_list = []
    batch_label_list = []
    batch_length = []

    for i in range(0, len(data_list), batch):
        upper = min(len(data_list), i + batch)
        data = pad_sequence(data_list[i:upper], batch_first=True, padding_value=0)
        label = pad_sequence(label_list[i:upper], batch_first=True, padding_value=0)
        batch_data_list.append(data)
        batch_label_list.append(label)
        batch_length.append(torch.tensor(length[i:upper]))
    return batch_data_list, batch_label_list, batch_length

