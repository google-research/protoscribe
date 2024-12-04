# Test data in this directory:

* `rabbit_tf_example.textproto`: Text protocol buffer in Tensorflow `tf.Example`
  format containing single corpus document concerning `rabbit` concept.

* `{train,validation,test}_example.textproto`: Sample documents in `tf.Example`
  format from the three dataset splits.

* `{train,validation,test}.tfr-1-of-1`: Small dataset splits (50 documents
  overall) in Tensorflow `TFRecord` format. Produced by

  ```shell
  PYTHONPATH=. python protoscribe/corpus/builder/build_dataset_main.py \
    --num_texts 50 \
    --max_local_workers=1 \
    --output_dir=/tmp/protoscribe \
    --logtostderr
  ```
