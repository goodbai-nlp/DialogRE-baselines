BASE=data-v1
BASE=data-v2
tokenizer="bert"
tokenizer="roberta"
save_dir=workplace
setting=$BASE-$tokenizer-bin
setting=$BASE-$tokenizer-new-bin
mkdir -p $save_dir
python preprocess.py \
    --train_path $BASE/train \
    --dev_path $BASE/dev \
    --test_path $BASE/test \
    --devc_path $BASE/devc \
    --testc_path $BASE/testc \
    --tokenizer $tokenizer \
    --save_data $save_dir/$setting
