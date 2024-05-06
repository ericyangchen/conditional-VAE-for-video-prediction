tensorboard --logdir="/home_nfs/ericyangchen/DLP/lab4/src/outputs" --port 9090


# train
python Trainer.py --DR ./data --batch_size 2 --save_root ./outputs/monotonic \
    --kl_anneal_type Monotonic \
    --num_epoch 30 \
    --fast_train \
    --fast_train_epoch 10 \
    --tfr 0.0 

python Trainer.py --DR ./data --batch_size 2 --save_root ./outputs/cyclical \
    --kl_anneal_type Cyclical \
    --kl_anneal_ratio 1.5 \
    --num_epoch 30 \
    --fast_train \
    --fast_train_epoch 10 \
    --tfr 0.0 \

python Trainer.py --DR ./data --batch_size 2 --save_root ./outputs/without \
    --kl_anneal_type Without \
    --num_epoch 30 \
    --fast_train \
    --fast_train_epoch 30 \
    --tfr 0.0


# eval
model_dir="monotonic/20240507-0202"
model_epoch="best"
python Trainer.py --DR ./data --batch_size 2 --test \
    --ckpt_path ./outputs/${model_dir}/ckpt/epoch=${model_epoch}.ckpt \
    --save_root ./outputs/${model_dir}

    

# test
model_dir="best"
model_epoch="best"
python Tester.py --DR ./data \
    --save_root ./outputs/${model_dir} \
    --ckpt_path ./outputs/${model_dir}/ckpt/epoch=${model_epoch}.ckpt



# demo
python Tester.py --DR ./data --save_root ./demo --ckpt_path ./outputs/best/ckpt/epoch=best.ckpt

python demo/demo.py --gt_path ./demo/gt.csv --submission_path ./demo/submission.csv