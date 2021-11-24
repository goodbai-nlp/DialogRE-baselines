# !/bin/bash
datapath=../data/
datacate=data-v1
#datacate=data-v2
tokenizer_path=$1

case $tokenizer_path in
    *"roberta"*)
        tokenizer="roberta"
        ;;
    *"bert"*)
        tokenizer="bert"
        ;;
    *)
        echo "Invalid tokenizer type:"$tokenizer_path
esac

save_dir=workplace
setting=$datacate-$tokenizer-bin

mkdir -p $save_dir
python preprocess.py \
    --train_path $datapath/$datacate/train \
    --dev_path $datapath/$datacate/dev \
    --test_path $datapath/$datacate/test \
    --devc_path $datapath/$datacate/devc \
    --testc_path $datapath/$datacate/testc \
    --tokenizer_name_or_path $tokenizer_path \
    --save_data $save_dir/$setting
