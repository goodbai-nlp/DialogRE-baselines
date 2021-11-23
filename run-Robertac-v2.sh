BERT_BASE_DIR=/public/home/zhangyuegroup/baixuefeng/data/pretrained-model/roberta-base

model=STD
dev=2
mode=$2
databin=$1
seed=42
setting=v2-old
setting=v2
if [ "$mode" == "train" ]
then
echo "Start Training..."
for seed in 42
do

save_path=workplace/Roberta_f1_max-512-${model}-seed-${seed}-$setting-baseline
save_path=workplace/Roberta_f1_max-512-${model}-seed-${seed}-$setting-baseline-datanew

mkdir -p $save_path
CUDA_VISIBLE_DEVICES=$dev python run.py --task_name bert --do_train --do_eval \
	--architecture $model \
	--seed $seed \
	--model_name_or_path $BERT_BASE_DIR \
	--max_seq_length 512   \
	--train_batch_size 24   \
	--eval_batch_size 1   \
	--learning_rate 3e-5   \
	--num_train_epochs 30   \
	--output_dir $save_path  \
	--model_type "entity-max" \
	--entity_drop 0.1 \
	--save_data $databin \
	--gradient_accumulation_steps 2 2>&1 | tee $save_path/run.log
done
elif [ "$mode" == "test" ]
then
echo "Start Testing..."
CUDA_VISIBLE_DEVICES=$dev python run.py --task_name bert --do_eval \
	--architecture $model \
	--seed $seed \
	--model_name_or_path $BERT_BASE_DIR \
	--max_seq_length 512   \
	--train_batch_size 24   \
	--eval_batch_size 1   \
	--learning_rate 3e-5   \
	--num_train_epochs 30   \
	--output_dir $save_path  \
	--model_type "entity-max" \
	--entity_drop 0.1 \
	--save_data $databin \
	--gradient_accumulation_steps 2 2>&1 | tee $save_path/eval.log
else
	echo "Invalid mode $mode!!!"
fi
