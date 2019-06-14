# Unequal-Training-for-Deep-Face-Recognition-with-Long-Tailed-Noisy-Data.

This is the code of CVPR 2019 paper《Unequal Training for Deep Face Recognition with Long Tailed Noisy Data》.

![](https://github.com/zhongyy/Deep-Difference-Analysis-in-Similar-looking-face-recognition/blob/master/zhou.jpg)

## Usage Instructions

1. Our method need two stage training, therefore our code is also stepwise. I will be happy if my humble code would help you. If there are questions or issues, please let me know. 

2. Our code is adopted from [InsightFace](https://github.com/deepinsight/insightface). I sincerely appreciate for their contributions.

### Prepare the code and the data.

1. Install `MXNet` with GPU support (Python 2.7).

```
pip install mxnet-cu90
```
2. download the code as `unequal_code/`
```
git clone https://github.com/zhongyy/Unequal-Training-for-Deep-Face-Recognition-with-Long-Tailed-Noisy-Data..git
```

3. download the [MF2](https://github.com/deepinsight/insightface) and the [evaluation dataset](https://github.com/deepinsight/insightface), then place them in `unequal_code/MF2_pic9_head/` `unequal_code/MF2_pic9_tail/` and `unequal_code/eval_dataset/` respectively.

### step 1: Pretrain MF2_pic9_head with [ArcFace](https://github.com/deepinsight/insightface).

End it when the acc of validation dataset (lfw,cfp-fp and agedb-30) does not ascend.

```
CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --network r50 --loss-type 4  --margin-m 0.5 --data-dir ./MF2_pic9_head/ --end-epoch 40 --per-batch-size 100 --prefix ../models/r50_arc_pic9/model 2>&1|tee r50_arc_pic9.log
```

### step 2: Train the head data with NRA (finetune from step 1).

1. Once the model_t,0 is saved, end it.

```
CUDA_VISIBLE_DEVICES='0,1' python -u train_relusoft_savemodel.py --network r50 --loss-type 4 --margin-m 0.5 --data-dir ./MF2_pic9_head/ --end-epoch 1 --lr 0.01  --per-batch-size 100 --noise-beta 0.9 --prefix ../models/NRA_r50pic9/model_t --bin-dir ./src/ --pretrained ../models/r50_arc_pic9/model,xx 2>&1|tee NRA_r50pic9_savemodel.log
```

2. End it when the acc of validation dataset(lfw, cfp-fp and agedb-30) does not ascend.

```
CUDA_VISIBLE_DEVICES='0,1' python -u train_relusoft_gs.py --network r50 --loss-type 4 --margin-m 0.5 --data-dir ./bin_dir/ --lr 0.01 --lr-steps 50000,90000 --per-batch-size 100 --prefix ../models/NRA_r50pic9/model --bin-dir ./bin_dir/ --pretrained ../models/NRA_r50pic9/model_t,0 2>&1|tee NRA_r50pic9.log
```

### step 3: 

1. Generate the denoised head data using `./MF2_pic9_head/train.lst` and `0_noiselist.txt` which has been generated in step 2. (We provide our version in [MF2](https://github.com/deepinsight/insightface).)

2. Using the denoised head data (have removed the third type noise) and the tail data to continue the second stage training. It's noting that the training process need finetune manually by increase the `--interweight` gradually. When you change the interweight, you also need change the pretrained model by yourself, because we could not know which is the best model in the last training stage unless we test the model on the target dataset (MF2 test). We always finetune from the best model in the last training stage.  

```
CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' python -u train_debug_soft_gs.py --network r50 --loss-type 4 --data-dir ./MF2_pic9_head_denoise/ --data-dir-interclass ./MF2_pic9_tail/ --end-epoch 100000 --lr 0.001 --interweight 1 --bag-size 3600 --batch-size1 360 --batchsize_id 360 --batch-size2 40  --pretrained /home/zhongyaoyao/insightface/models/NRA_r50pic9/model,xx --prefix ../models/model_all/model 2>&1|tee all_r50.log
```

### Note:

1. We use the last arcface model (best performance) to find the third type noise. Next we drop the fc weight of the last arcface model, then finetune from it using NR loss (adding a reweight term by putting more confidence in the prediction of the training model). 

2. The second stage training process need very careful manual tuning. We provide our training log for reference. 

### Result:

| Method           | m1   | m2   | m3   | LFW   | CFP-FP | AgeDB-30 |
| ---------------- | ---- | ---- | ---- | ----- | ------ | -------- |
| W&F Norm Softmax | 1    | 0    | 0    | 0     | 0      | 0        |


