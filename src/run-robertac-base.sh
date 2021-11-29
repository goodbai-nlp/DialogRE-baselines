BERT_BASE_DIR=/public/home/zhangyuegroup/baixuefeng/data/pretrained-model/roberta-base

model=STD
dev=3
mode=$2
databin=$1
datacate=v2
seed=42
if [ "$mode" == "train" ]
then
echo "Start Training..."

save_path=workplace/output/roberta-base-512-seed-${seed}-$datacate-baseline
mkdir -p $save_path
CUDA_VISIBLE_DEVICES=$dev python run.py --do_train --do_eval \
	--architecture $model \
	--seed $seed \
	--model_name_or_path $BERT_BASE_DIR \
	--max_seq_length 512   \
	--num_labels 36 \
	--train_batch_size 24   \
	--eval_batch_size 1   \
	--learning_rate 2e-5   \
	--num_train_epochs 30   \
	--output_dir $save_path  \
	--model_type "entity-max" \
	--entity_drop 0.1 \
	--save_data $databin \
	--gradient_accumulation_steps 2 2>&1 | tee $save_path/run.log
elif [ "$mode" == "test" ]
then
echo "Start Testing..."
save_path=workplace/output/roberta-base-512-seed-${seed}-$datacate-baseline
CUDA_VISIBLE_DEVICES=$dev python run.py --task_name bert --do_eval \
	--architecture $model \
	--seed $seed \
	--model_name_or_path $BERT_BASE_DIR \
	--max_seq_length 512   \
	--train_batch_size 24   \
	--eval_batch_size 1   \
	--learning_rate 2e-5   \
	--num_train_epochs 30   \
	--output_dir $save_path  \
	--model_type "entity-max" \
	--entity_drop 0.1 \
	--save_data $databin \
	--gradient_accumulation_steps 2 2>&1 | tee $save_path/eval.log
else
	echo "Invalid mode $mode!!!"
fi

