tag=tag_name_as_you_like
test_path=./data/ChEBI-20/test.json
load_path=
eval_batch_size=512
cmd="python test.py 
    --use_cuda 
    --load_model
    --load_path $load_path
    --eval_batch_size $eval_batch_size 
    --test_path $test_path
    --tag $tag"

cmd=${cmd}" 2>&1 | tee ${tag}.log"

echo ${cmd}

eval $cmd