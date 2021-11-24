# coding:utf-8
import os
import torch
import argparse
from dataset import DialogREDataSet
from transformers import AutoTokenizer
import time

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--train_path", type=str, help="base train data path")
    argparser.add_argument("--dev_path", type=str, help="base dev data path")
    argparser.add_argument("--test_path", type=str, help="base test data path")
    argparser.add_argument("--devc_path", type=str, help="base dev data path")
    argparser.add_argument("--testc_path", type=str, help="base test data path")
    argparser.add_argument("--save_data", type=str, help="saved data path")
    argparser.add_argument("--tokenizer_name_or_path", type=str, default="bert-base-uncased", help="saved data path")
    FLAGS, unparsed = argparser.parse_known_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if not os.path.exists(FLAGS.save_data):
        os.makedirs(FLAGS.save_data)

    if "roberta" in FLAGS.tokenizer_name_or_path:
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained(FLAGS.tokenizer_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": ["madeupword0001", "madeupword0002"]})
        # tokenizer.add_special_tokens({"additional_special_tokens": ["madeupword0000", "madeupword0010","madeupword0001", "madeupword0002", "madeupword0003","madeupword0004", "madeupword0005","madeupword0006","madeupword0007","madeupword0008","madeupword0009"] })
    elif "bert" in FLAGS.tokenizer_name_or_path:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(FLAGS.tokenizer_name_or_path)
        tokenizer.add_special_tokens({"additional_special_tokens": ["[unused1]", "[unused2]"]})
        # tokenizer.add_special_tokens({"additional_special_tokens": ["[unused0]", "[unused10]", "[unused1]", "[unused2]", "[unused3]", "[unused4]", "[unused5]", "[unused6]", "[unused7]", "[unused8]", "[unused9]"] })

    else:
        print(f'Unsupported tokenizer {FLAGS.tokenizer}.')
        exit()

    tokenize_fn = tokenizer
    print("Loading train data ...")
    s_time = time.time()
    train_set = DialogREDataSet(FLAGS.train_path, tokenize_fn, save_path=FLAGS.save_data+'/train', data_type='std')
    print("Loading trainset takes {:.3f}s".format(time.time() - s_time))

    print("Loading dev data ...")
    s_time = time.time()
    dev_set = DialogREDataSet(FLAGS.dev_path, tokenize_fn, save_path=FLAGS.save_data+'/dev', data_type='std')
    print("Loading devset takes {:.3f}s".format(time.time() - s_time))

    print("Loading test data ...")
    s_time = time.time()
    test_set = DialogREDataSet(FLAGS.test_path, tokenize_fn, save_path=FLAGS.save_data+'/test', data_type='std')
    print("Loading testset takes {:.3f}s".format(time.time() - s_time))

    ## TODO supply for conversational evaluate setting
    # print("Loading devc data ...")
    # s_time = time.time()
    # devc_set = DialogREDataSet(FLAGS.devc_path, tokenize_fn, word_vocab=word2id, save_path=FLAGS.save_data+'/devc', data_type='stdc')
    # print("Loading devcset takes {:.3f}s".format(time.time() - s_time))

    # print("Loading testc data ...")
    # s_time = time.time()
    # testc_set = DialogREDataSet(FLAGS.testc_path, tokenize_fn, word_vocab=word2id, save_path=FLAGS.save_data+'/testc', data_type='stdc')
    # print("Loading testcset takes {:.3f}s".format(time.time() - s_time))
