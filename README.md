## CUDA: Curriculum of Data Augmentation for Long-tailed Recognition (ICLR'23 Spotlight) 

[OpenReview](https://openreview.net/forum?id=RgUPdudkWlN)  
Sumyeiong Ahn*, Jongwoo Ko*, Se-Young Yun  (KAIST AI)



## Abstract
Class imbalance problems frequently occur in real-world tasks, and conventional deep learning algorithms are well known for performance degradation on imbal- anced training datasets. To mitigate this problem, many approaches have aimed to balance among given classes by re-weighting or re-sampling training samples. These re-balancing methods increase the impact of minority classes and reduce the influence of majority classes on the output of models. However, the extracted repre- sentations may be of poor quality owing to the limited number of minority samples. To handle this restriction, several methods have been developed that increase the representations of minority samples by leveraging the features of the majority samples. Despite extensive recent studies, no deep analysis has been conducted on determination of classes to be augmented and strength of augmentation has been conducted. In this study, we first investigate the correlation between the degree of augmentation and class-wise performance, and find that the proper degree of augmentation must be allocated for each class to mitigate class imbalance problems. Motivated by this finding, we propose a simple and efficient novel curriculum, which is designed to find the appropriate per-class strength of data augmentation, called CUDA: CUrriculum of Data Augmentation for long-tailed recognition. CUDA can simply be integrated into existing long-tailed recognition methods. We present the results of experiments showing that CUDA effectively achieves better general- ization performance compared to the state-of-the-art method on various imbalanced datasets such as CIFAR-100-LT, ImageNet-LT, and iNaturalist 2018.


## Requirements
```
pip install -r requirements.txt
```

## Training
Code for training CIFAR-100 is in the following file: cifar/main.py, code for training ImageNet-LT and iNaturalist 2018 is in the following file: large_scale/train.py

### CIFAR-100-LT
```
cd cifar
python main.py --dataset cifar100 --imb_ratio 100 --num_max 500 --epochs 200 \
               --gpu 0 --out /your/output/directory/here \
               --loss_fn bs --cuda --cutout
```

### ImageNet-LT
```
cd large_scale
python train.py --dataset imgnet --epochs 100 --num_classes 1000 \
                -a resnet50 --root /your/data/directory/here \
                --loss_type LDAM --data_aug CUDA --train_rule DRW \
                --workers 12 --print_freq 100 -b 256 --lr 0.1 --wd 2e-4
```

### iNaturalist 2018 
```
cd large_scale
python train.py --dataset inat --epochs 100 --num_classes 8142 \
                -a resnet50 --root /your/data/directory/here \
                --loss_type CE --data_aug CUDA --train_rule None \
                --workers 24 --print_freq 100 -b 512 --lr 0.1 --wd 2e-4
```

## References
- https://github.com/naver-ai/cmo
- https://github.com/FlamieZhu/Balanced-Contrastive-Learning
- https://github.com/Bazinga699/NCL
- https://github.com/frank-xwang/RIDE-LongTailRecognition

