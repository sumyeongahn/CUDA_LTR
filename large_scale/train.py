# reference code: https://github.com/kaidic/LDAM-DRW/blob/master/cifar_train.py
import math
import os
import random
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

from tqdm import tqdm
from imbalance_data.lt_data import LT_Dataset
from losses import *
from opts import parser
import warnings
from torch.nn import Parameter
import torch.nn.functional as F
from models.util import *
from util.util import *
from util.randaugment import rand_augment_transform
import util.moco_loader as moco_loader
from training_functions import *

best_acc1 = 0

def main():
    args = parser.parse_args()
    args.store_name = '_'.join(
        [args.dataset, args.arch, args.loss_type, args.train_rule, args.data_aug, str(args.imb_factor),
         str(args.rand_number),
         str(args.mixup_prob), args.exp_str])
    prepare_folders(args)
    if args.cos:
        print("use cosine LR")

    if args.seed is not None:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        np.random.seed(args.seed)
        random.seed(args.seed)
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    ngpus_per_node = torch.cuda.device_count()
    main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    global train_cls_num_list
    global cls_num_list_cuda


    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    # create model
    print("=> creating model '{}'".format(args.arch))
    num_classes = 1000 if args.dataset == 'imgnet' else 8142
    if args.loss_type == 'RIDE':
        import models.model as net
        if args.arch == 'resnet50':
            model = net.ResNet50Model(num_classes)
        elif args.arch == 'resnet10':
            model = net.ResNet10Model(num_classes)
        elif args.arch == 'resnext50':
            model = net.ResNeXt50Model(num_classes)
    elif args.loss_type == 'NCL':
        pass
    elif args.loss_type == 'BCL':
        import models.BCLResNet as net
        model = net.BCLModel(num_classes=num_classes, name=args.arch, use_norm=True)
    else:
        if args.arch == 'resnet50':
            import torchvision.models as models
            model = getattr(models, args.arch)(pretrained=False)
        elif args.arch == 'resnet10':
            import models.model as net
            model = net.ResNet10Model(num_classes, num_experts=1)
        elif args.arch == 'resnext50':
            import models.model as net
            model = net.ResNeXt50Model(num_classes, num_experts=1)
        
        if hasattr(model, 'backbone'):
            num_ftrs = model.backbone.fc.in_features
            if args.loss_type == 'LDAM':
                model.backbone.fc = NormedLinear(num_ftrs, num_classes)
            else:
                model.backbone.fc = nn.Linear(num_ftrs, num_classes)
        else:
            num_ftrs = model.fc.in_features
            if args.loss_type == 'LDAM':
                model.fc = NormedLinear(num_ftrs, num_classes)
            else:
                model.fc = nn.Linear(num_ftrs, num_classes)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model = torch.nn.DataParallel(model).cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cuda:0')
            # args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # cudnn.benchmark = True
    if args.dataset == 'imgnet':
        train_config = 'config/ImageNet/ImageNet_LT_train.txt'
        valid_config = 'config/ImageNet/ImageNet_LT_test.txt'
    elif args.dataset == 'inat':
        train_config = 'config/iNaturalist/iNaturalist18_train.txt'
        valid_config = 'config/iNaturalist/iNaturalist18_val.txt'
    
    train_dataset = LT_Dataset(args.root, train_config, args, args.dataset, args.loss_type, 
                               use_randaug=args.use_randaug, split='train', aug_prob=args.aug_prob,
                               upgrade=args.cuda_upgrade, downgrade=args.cuda_downgrade)
    val_dataset = LT_Dataset(args.root, valid_config, args, args.dataset, args.loss_type, split = 'valid')

    num_classes = len(np.unique(train_dataset.targets))
    assert num_classes == 1000 if args.dataset == 'imgnet' else 8142
    args.num_class = num_classes

    cls_num_list = [0] * num_classes
    for label in train_dataset.targets:
        cls_num_list[label] += 1
    print('cls num list:')
    print(cls_num_list)
    args.cls_num_list = cls_num_list
    train_cls_num_list = np.array(cls_num_list)
    
    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    cls_num_list_cuda = torch.from_numpy(np.array(cls_num_list)).float().cuda()
    start_time = time.time()
    print("Training started!")
    log_training = open(os.path.join(args.root_log, args.store_name, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(args.root_log, args.store_name, 'log_test.csv'), 'w')
    with open(os.path.join(args.root_log, args.store_name, 'args.txt'), 'w') as f:
        f.write(str(args))

    if 'CMO' in args.data_aug or args.train_rule == 'CRT':
        cls_weight = 1.0 / (np.array(cls_num_list))
        cls_weight = cls_weight / np.sum(cls_weight) * len(cls_num_list)
        labels = train_loader.dataset.targets
        samples_weight = np.array([cls_weight[t] for t in labels])
        samples_weight = torch.from_numpy(samples_weight)
        samples_weight = samples_weight.double()
        print("samples_weight", samples_weight)
        sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(labels), replacement=True)
        weighted_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=sampler)
    else:
        weighted_trainloader = None
        
    # Prepare index list to update state
    # For iNaturalist2018, it takes 3 minutes.
    # 3 minutes * 100 epoch = 300 mins == 5 hours!
    if args.data_aug == 'CUDA':
        cls_pos_list = []
        for cidx in tqdm(range(len(cls_num_list))):
            class_pos = torch.where(torch.tensor(train_loader.dataset.targets) == cidx)[0]
            cls_pos_list.append(class_pos)
    
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        
        print ('START!!')
        
        if args.train_rule == 'None':
            train_sampler = None
            per_cls_weights = None
        elif args.train_rule == 'CBReweight':
            train_sampler = None
            beta = 0.9999
            effective_num = 1.0 - np.power(beta, cls_num_list)
            per_cls_weights = (1.0 - beta) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'DRW':
            train_sampler = None
            idx = epoch // 80
            betas = [0, 0.9999]
            effective_num = 1.0 - np.power(betas[idx], cls_num_list)
            per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
            per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(args.gpu)
        elif args.train_rule == 'CRT':
            train_sampler = None
            per_cls_weights = None
            train_loader = weighted_trainloader
            # weighted_trainloader.dataset = train_loader.dataset
            # trainloader = copy.deepcopy(weighted_trainloader)
            
            lr = args.lr 
            lr *= 0.5 * (1. + math.cos(math.pi * (epoch + 1) / (args.epochs + 1)))
            optimizer = torch.optim.SGD(model.module.fc.parameters(), lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
        else:
            warnings.warn('Sample rule is not listed')

        if args.loss_type == 'CE':
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'BS':
            criterion = BalancedSoftmaxLoss(cls_num_list=cls_num_list_cuda).cuda(args.gpu)
        elif args.loss_type == 'LDAM':
            criterion = LDAMLoss(cls_num_list=cls_num_list, max_m=0.5, s=30, weight=per_cls_weights).cuda(args.gpu)
        elif args.loss_type == 'RIDE':
            reweight_epoch = 80
            reweight_factor = 0.02 if args.dataset == 'imgnet' else 0.015
            criterion = RIDELoss(cls_num_list=cls_num_list, reweight_epoch=reweight_epoch,
                                 reweight_factor=reweight_factor).to(torch.device('cuda'))
        elif args.loss_type == 'BCL':
            criterion = BCLLoss(cls_num_list_cuda)
        else:
            warnings.warn('Loss type is not listed')
            return

        # train for one epoch
        if args.loss_type == 'RIDE':
            if 'CUDA' in args.data_aug:
                if (epoch +1) % args.test_epoch == 0:
                    update_score_base(train_loader, model, cls_num_list, cls_pos_list, args)
            train_ride(train_loader, model, criterion, optimizer, epoch, args, log_training, weighted_trainloader)
        elif args.loss_type == 'NCL':
            if 'CUDA' in args.data_aug:
                if (epoch +1) % args.test_epoch == 0:
                    update_score_base(train_loader, model, cls_num_list, cls_pos_list, args)
            train_ncl(train_loader, model, criterion, optimizer, epoch, args, log_training, weighted_trainloader)
        elif args.loss_type == 'BCL':
            if 'CUDA' in args.data_aug:
                if (epoch +1) % args.test_epoch == 0:
                    update_score_base(train_loader, model, cls_num_list, cls_pos_list, args)
            train_bcl(train_loader, model, criterion, optimizer, epoch, args, log_training, weighted_trainloader)
        else:
            if 'CUDA' in args.data_aug:
                if (epoch +1) % args.test_epoch == 0:
                    update_score_base(train_loader, model, cls_num_list, cls_pos_list, args)
            train(train_loader, model, criterion, optimizer, epoch, args, log_training, weighted_trainloader)

        # evaluate on validation set
        acc1 = validate(val_loader, model, nn.CrossEntropyLoss(), train_cls_num_list, args, log_testing)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        output_best = 'Best Prec@1: %.3f\n' % (best_acc1)
        print(output_best)
        log_testing.write(output_best + '\n')
        log_testing.flush()

        save_checkpoint(args, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
        }, is_best, epoch + 1)

    end_time = time.time()

    print("It took {} to execute the program".format(hms_string(end_time - start_time)))
    log_testing.write("It took {} to execute the program".format(hms_string(end_time - start_time)) + '\n')
    log_testing.flush()

if __name__ == '__main__':
    main()
