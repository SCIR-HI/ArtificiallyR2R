tag=tag_name_as_you_like
train_path=./data/ChEBI-20/train.json
valid_path=./data/ChEBI-20/dev.json
test_path=./data/ChEBI-20/test.json
batch_size=32
eval_batch_size=512
steps=50000
lr=1e-3
accumulate_grad_batches=1
cmd="python run.py 
    --use_cuda 
    --shuffle 
    --mix_precision
    --steps $steps
    --batch_size $batch_size 
    --eval_batch_size $eval_batch_size 
    --train_path $train_path
    --valid_path $valid_path
    --test_path $test_path
    --lr $lr 
    --tag $tag 
    --accumulate_grad_batches $accumulate_grad_batches"

cmd=${cmd}" 2>&1 | tee ${tag}.log"

echo ${cmd}

eval $cmd