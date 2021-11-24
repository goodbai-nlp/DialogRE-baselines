save_dir=$1
datacate=data-v2
python evaluate.py --devdata ../data/$datacate/dev.json --testdata ../data/$datacate/test.json --f1dev $save_dir/logits_dev.txt --f1test $save_dir/logits_test.txt --f1cdev $save_dir/logits_devc.txt --f1ctest $save_dir/logits_testc.txt 2>&1 | tee $save_dir/eval-f1.log
