# Unequal-Training-for-Deep-Face-Recognition-with-Long-Tailed-Noisy-Data.
This is the code of CVPR 2019 paper《Unequal Training for Deep Face Recognition with Long Tailed Noisy Data》.

## Usage Instructions
Our method need two stage training, therefore our code is also stepwise. I will be happy if my humble code would help you. If there are questions or issues, please let me know. 
Our code is adopted from [InsightFace](https://github.com/deepinsight/insightface). I sincerely appreciate for their contributions.

### step 1: Prepare the code and the data.

1. Install `MXNet` with GPU support (Python 2.7).

```
pip install mxnet-cu90
```
2. download the code as `unequal_code/`
```
git clone https://github.com/zhongyy/Unequal-Training-for-Deep-Face-Recognition-with-Long-Tailed-Noisy-Data..git
```

3. download the [MF2](https://github.com/deepinsight/insightface) and the [evaluation dataset](https://github.com/deepinsight/insightface), then place them in `unequal_code/MF2_pic9_head/` `unequal_code/MF2_pic9_tail/` and `unequal_code/eval_dataset/` respectively.

### step 2: Pretrain MF2_pic9_head with [ArcFace](https://github.com/deepinsight/insightface).

```
CUDA_VISIBLE_DEVICES='0,1' python -u train_softmax.py --network r50 --loss-type 4  --margin-m 0.5 --data-dir ./MF2_pic9_head/ --end-epoch 40 --per-batch-size 100 --prefix ../models/r50_arc_pic9/model 2>&1|tee r50_arc_pic9.log
```
End it when the acc of validation dataset (lfw,cfp-fp and agedb-30) does not ascend.


### step 3: 
Train the head data with NRA (finetune from step 2).
1, Once the model_t,0 is saved, end it.
2, run sh NRA.sh, press ctrl+C to end it until the acc of validation dataset(lfw, cfp-fp and agedb-30) does not ascend.

