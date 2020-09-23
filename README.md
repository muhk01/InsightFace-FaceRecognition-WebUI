
# InsightFace - Face Recognition

# Requirements
1. MxNet
2. Python3
3. Necessary Library for Building Mxnet

# Starting Webserver 
One webserver started will be served on localnet, all registered face will be viewed.
![Web](https://github.com/muhk01/InsightFace-WebUI/blob/master/2.png)

# Viewing inference Result 
![Inference](https://github.com/muhk01/InsightFace-WebUI/blob/master/1.png)

# Adding Face and Training
Upload face into database to do recognize.
![AddFace](https://github.com/muhk01/InsightFace-WebUI/blob/master/3.png)

### Pretrained Models

You can use `$INSIGHTFACE/src/eval/verification.py` to test all the pre-trained models.

**Please check [Model-Zoo](https://github.com/deepinsight/insightface/wiki/Model-Zoo) for more pretrained models.**

### Verification Results on Combined Margin

A combined margin method was proposed as a function of target logits value and original `θ`:

```
COM(θ) = cos(m_1*θ+m_2) - m_3
```

For training with `m1=1.0, m2=0.3, m3=0.2`, run following command:
```
CUDA_VISIBLE_DEVICES='0,1,2,3' python -u train_softmax.py --network r100 --loss combined --dataset emore
```

Results by using ``MS1M-IBUG(MS1M-V1)``

| Method           | m1   | m2   | m3   | LFW   | CFP-FP | AgeDB-30 |
| ---------------- | ---- | ---- | ---- | ----- | ------ | -------- |
| W&F Norm Softmax | 1    | 0    | 0    | 99.28 | 88.50  | 95.13    |
| SphereFace       | 1.5  | 0    | 0    | 99.76 | 94.17  | 97.30    |
| CosineFace       | 1    | 0    | 0.35 | 99.80 | 94.4   | 97.91    |
| ArcFace          | 1    | 0.5  | 0    | 99.83 | 94.04  | 98.08    |
| Combined Margin  | 1.2  | 0.4  | 0    | 99.80 | 94.08  | 98.05    |
| Combined Margin  | 1.1  | 0    | 0.35 | 99.81 | 94.50  | 98.08    |
| Combined Margin  | 1    | 0.3  | 0.2  | 99.83 | 94.51  | 98.13    |
| Combined Margin  | 0.9  | 0.4  | 0.15 | 99.83 | 94.20  | 98.16    |

## Citation

If you find *InsightFace* useful in your research, please consider to cite the following related papers:

```
@inproceedings{deng2018arcface,
title={ArcFace: Additive Angular Margin Loss for Deep Face Recognition},
author={Deng, Jiankang and Guo, Jia and Niannan, Xue and Zafeiriou, Stefanos},
booktitle={CVPR},
year={2019}
}
```
