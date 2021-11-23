# coding:utf-8
import math
import json
from tqdm import tqdm
import pickle
import numpy as np

def _truncate_seq_tuple(tokens_a, tokens_b, tokens_c, max_length):
    """Truncates a sequence tuple in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b) + len(tokens_c)
        if total_length <= max_length:
            break
        if len(tokens_a) >= len(tokens_b) and len(tokens_a) >= len(tokens_c):
            tokens_a.pop()
        elif len(tokens_b) >= len(tokens_a) and len(tokens_b) >= len(tokens_c):
            tokens_b.pop()
        else:
            tokens_c.pop()


def get_entity_mask(src_ids, entity_ids):
    mask = []
    idx_src_s, idx_src_e = 0, 0
    while idx_src_s < len(src_ids):
        idx_ent = 0
        if src_ids[idx_src_s] == entity_ids[idx_ent]:
            idx_src_e = idx_src_s
            idx_ent = 0
            while (
                idx_src_e < len(src_ids)
                and idx_ent < len(entity_ids)
                and src_ids[idx_src_e] == entity_ids[idx_ent]
            ):
                idx_src_e += 1
                idx_ent += 1
            if idx_ent == len(entity_ids):  # matched
                mask += [1] * len(entity_ids)
                idx_src_s = idx_src_e
            else:
                mask.append(0)
                idx_src_s += 1
        else:
            mask.append(0)
            idx_src_s += 1
    assert len(mask) == len(src_ids)
    return mask


def replace_speaker(s, e1="", e2="", tokenizer=None):
    res = []
    if "roberta" in tokenizer.name_or_path:
        unused_tokens = ["madeupword0001", "madeupword0002"]
    elif "bert" in tokenizer.name_or_path:
        unused_tokens = ["[unused1]", "[unused10]"]
    for itm in s.split(): 
        if itm.startswith("speaker") and itm[7:].isdigit():
            if itm == e1:                       # replace speaker information
                new_itm = unused_tokens[0]
            elif itm == e2:
                new_itm = unused_tokens[1]
            else:
                # new_itm = "[unused{}]".format(itm[7:])
                new_itm = itm
            res.append(new_itm)
        else:
            res.append(itm)
    return " ".join(res)

def convert_input(src, e1, e2, max_seq_length=512, max_d_len=491, tokenizer=None):
    src = replace_speaker(src, e1, e2, tokenizer)
    e1_new = replace_speaker(e1, e1, e2, tokenizer)
    e2_new = replace_speaker(e2, e1, e2, tokenizer)
    # print('e1:', e1)
    # print('e2:', e2)
    # print('e1_new:', e1_new)
    # print('e2_new:', e2_new)
    # print("Replaced_src:", src)
    tokens_a = tokenizer.tokenize(src)
    # print("Replaced_src_tok:", tokens_a)
    _tokens_b = tokenizer.tokenize(e1_new)
    _tokens_c = tokenizer.tokenize(e2_new)

    _truncate_seq_tuple(tokens_a, _tokens_b, _tokens_c, max_seq_length - 6)
    tokens_b = _tokens_b + [tokenizer.sep_token] + _tokens_c

    tokens = []
    segment_ids = []
    tokens.append(tokenizer.cls_token)
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    input_len = len(tokens)
    tokens.append(tokenizer.sep_token)
    segment_ids.append(0)

    for token in tokens_b:              
        tokens.append(token)
        segment_ids.append(1)               # set seg_id = 1 for entities
    tokens.append(tokenizer.sep_token)
    segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    assert len(input_ids) <= max_seq_length, "Invalid instance {}, length: {}".format(' '.join(tokens), len(input_ids))
    b_ids = tokenizer.convert_tokens_to_ids(_tokens_b)
    c_ids = tokenizer.convert_tokens_to_ids(_tokens_c)
    # print('b_ids', b_ids)
    # print('c_ids', c_ids)
    # print('a_idx', input_ids)
    mask_b = get_entity_mask(input_ids[: len(tokens_a) + 1], b_ids)
    mask_c = get_entity_mask(input_ids[: len(tokens_a) + 1], c_ids)
    # print('mask_b', mask_b)
    # exit()
    while len(mask_b) < len(input_ids):
        mask_b.append(0)
        mask_c.append(0)
    
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(tokenizer.pad_token_id)
        input_mask.append(0)
        segment_ids.append(0)
        mask_b.append(0)
        mask_c.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(mask_b) == max_seq_length
    assert len(mask_c) == max_seq_length
    
    return input_ids, input_len, input_mask, segment_ids, mask_b, mask_c


class DialogREDataSet():
    def __init__(self, data_path, tokenizer, save_path='', data_type='std'):
        self.tokenizer = tokenizer
        RE_data = json.load(open(data_path + '.json', 'r', encoding='utf-8'))
        self.build_data(RE_data, 0, save_path, data_type)

    def __len__(self):
        return len(self.rid_lst)

    def cal_max_len(self, ids, curdepth, maxdepth):
        """calculate max sequence length"""
        assert curdepth <= maxdepth
        if isinstance(ids[0], list):
            res = max([self.cal_max_len(k, curdepth + 1, maxdepth) for k in ids])
        else:
            res = len(ids)
        return res
    
    def build_data(self, RE_data, id, save_path, dtype='std'):
        word_inp_lst, context_len_lst, inp_mask_lst, seg_mask_lst, e1_mask_lst, e2_mask_lst, rid_lst = [], [], [], [], [], [], []
        if not id:
            pbar = tqdm(total=len(RE_data))
        all_res, idx, split_idx = 0, 0, 0                            # ith dialog history
        for re_data in RE_data:
            dialog_itm, re_item =  re_data[0], re_data[1]
            dialog_concated = " ".join(dialog_itm)
            for j in range(len(re_item)):                       # Iterate relations
                rid = []
                for k in range(36):
                    if k + 1 in re_item[j]["rid"]:
                        rid += [1]
                    else:
                        rid += [0]
                e1, e2 = re_item[j]["x"].lower().replace("speaker ", "speaker"), re_item[j]["y"].lower().replace("speaker ", "speaker")
                src_tok = dialog_concated.strip().lower().replace("speaker ", "speaker").replace(':', ' :')
                # print("ori_src_tok:", src_tok)
                new_inp, context_len, input_mask, seg_mask, e1_mask, e2_mask = convert_input(src_tok, e1, e2, max_seq_length=512, max_d_len=491, tokenizer=self.tokenizer)
                word_inp_lst.append(new_inp)
                # print("new_src_tok:", new_inp)
                # exit()
                context_len_lst.append(context_len)
                inp_mask_lst.append(input_mask)
                seg_mask_lst.append(seg_mask)
                e1_mask_lst.append(e1_mask)
                e2_mask_lst.append(e2_mask)
                rid_lst.append(rid)
                all_res += 1
                if len(rid_lst) % 50000 == 0:
                    tmp_res = [word_inp_lst, context_len_lst, inp_mask_lst, seg_mask_lst, e1_mask_lst, e2_mask_lst, rid_lst]
                    pkl_file = "{:s}-{}.pkl".format(save_path, split_idx)
                    pickle.dump(tmp_res, open(pkl_file, 'wb'))
                    word_inp_lst, context_len_lst, inp_mask_lst, seg_mask_lst, e1_mask_lst, e2_mask_lst, rid_lst = [], [], [], [], [], [], []
                    split_idx += 1

            if not id:
                pbar.update(1)

        tmp_res = [word_inp_lst, context_len_lst, inp_mask_lst, seg_mask_lst, e1_mask_lst, e2_mask_lst, rid_lst]
        pkl_file = "{:s}-{}.pkl".format(save_path, split_idx)
        pickle.dump(tmp_res, open(pkl_file, 'wb'))
        if not id:
            pbar.close()
        print('All {} instances'.format(all_res))
        return