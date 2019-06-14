# Unequal-Training-for-Deep-Face-Recognition-with-Long-Tailed-Noisy-Data.
This is the code of CVPR 2019 paper《Unequal Training for Deep Face Recognition with Long Tailed Noisy Data》.

## Usage Instructions
Our method need two stage training, therefore our code is also stepwise. Hope my humble code would help you. If there are questions or issues, please let me know. 

### step 1: Prepare the code and the data.

1. Install `MXNet` with GPU support (Python 2.7).

```
pip install mxnet-cu90
```
2. download the code as `unequal_code/`
```
git clone https://github.com/zhongyy/Unequal-Training-for-Deep-Face-Recognition-with-Long-Tailed-Noisy-Data..git
```

3. download the [MF2](https://github.com/deepinsight/insightface) and the [evaluation dataset](https://github.com/deepinsight/insightface), then place them in `unequal_code/MF2_pic9/` `unequal_code/MF2_pic9_tail/` and `unequal_code/eval_dataset/` respectively.


### step 2: 
Pretrain MF2_pic9_head with arcface.

run sh pretrain_mf2pic9.sh, press ctrl+C to end it until the acc of validation dataset(lfw,cfp-fp and agedb-30) does not ascend.

Some of the code is adopted from [InsightFace](https://github.com/deepinsight/insightface). 
