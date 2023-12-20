tag=tag_name_as_you_like
train_path=./data/train.json
batch_size=32
steps=100000
lr=1e-3
accumulate_grad_batches=4
cmd="python run.py 
    --use_cuda 
    --shuffle 
    --mix_precision
    --steps $steps
    --batch_size $batch_size 
    --train_path $train_path
    --lr $lr 
    --tag $tag 
    --accumulate_grad_batches $accumulate_grad_batches"

cmd=${cmd}" 2>&1 | tee ${tag}.log"

echo ${cmd}

eval $cmd