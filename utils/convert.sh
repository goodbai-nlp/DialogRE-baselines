#! /bin/bash
basepath=path/to/unzipedfiles/
python convert_bert_tf_checkpoint_to_pytorch.py --tf_checkpoint_path $basepath/bert_model.ckpt --bert_config_file $basepath/bert_config.json --pytorch_dump_path $basepath/pytorch_model.bin
