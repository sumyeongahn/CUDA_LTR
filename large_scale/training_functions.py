import time, copy, random
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from util.util import *
from imbalance_data.lt_data import test_loader
from imbalance_data.cmo import *

def train(train_loader, model, criterion, optimizer, epoch, args, log, weighted_trainloader = None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to train mode
    model.train()

    if 'CMO' in args.data_aug and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
        inverse_iter = iter(weighted_trainloader)

    end = time.time()
    for i, data_tuple in enumerate(train_loader):
        input = data_tuple[0]
        target = data_tuple[1]

        if 'CMO' in args.data_aug and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
            try:
                data_tuple_f = next(inverse_iter)
                input2 = data_tuple_f[0]
                target2 = data_tuple_f[1]
            except:
                inverse_iter = iter(weighted_trainloader)
                data_tuple_f = next(inverse_iter)
                input2 = data_tuple_f[0]
                target2 = data_tuple_f[1]
            input2 = input2[:input.size()[0]]
            target2 = target2[:target.size()[0]]
            input2 = input2.cuda(args.gpu, non_blocking=True)
            target2 = target2.cuda(args.gpu, non_blocking=True)


        data_time.update(time.time() - end)
        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        
        r = np.random.rand(1)
        if 'CMO' in args.data_aug and args.start_data_aug < epoch < (args.epochs - args.end_data_aug) and r < args.mixup_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            output = model(input)
            loss = criterion(output, target) * lam + criterion(output, target2) * (1. - lam)

        else:
            output = model(input)
            loss = criterion(output, target).mean()

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

def train_ncl(train_loader, model, criterion, optimizer, epoch, args, log, weighted_trainloader = None):
    pass
    
def train_bcl(train_loader, model, criterion, optimizer, epoch, args, log, weighted_trainloader = None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    if 'CMO' in args.data_aug and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
        inverse_iter = iter(weighted_trainloader)

    end = time.time()
    for i, data_tuple in enumerate(train_loader):
        input = data_tuple[0]
        target = data_tuple[1]

        if 'CMO' in args.data_aug and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
            try:
                data_tuple_f = next(inverse_iter)
                input2 = data_tuple_f[0]
                target2 = data_tuple_f[1]
            except:
                inverse_iter = iter(weighted_trainloader)
                data_tuple_f = next(inverse_iter)
                input2 = data_tuple_f[0]
                target2 = data_tuple_f[1]
            input2 = input2[:input.size()[0]]
            target2 = target2[:target.size()[0]]
            input2 = input2.cuda(args.gpu, non_blocking=True)
            target2 = target2.cuda(args.gpu, non_blocking=True)


        data_time.update(time.time() - end)
        input = torch.cat([input[0], input[1], input[2]], dim=0).cuda(args.gpu, non_blocking=True)
        batch_size = target.shape[0]
        target = target.cuda(args.gpu, non_blocking=True)
        
        r = np.random.rand(1)
        if 'CMO' in args.data_aug and args.start_data_aug < epoch < (args.epochs - args.end_data_aug) and r < args.mixup_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            output = model(input)
            loss = criterion(output, target) * lam + criterion(output, target2) * (1. - lam)

        else:
            feat_mlp, logits, centers = model(input)
            centers = centers[:args.num_class]
            _, f2, f3 = torch.split(feat_mlp, [batch_size, batch_size, batch_size], dim=0)
            features = torch.cat([f2.unsqueeze(1), f3.unsqueeze(1)], dim=1)
            logits, _, __ = torch.split(logits, [batch_size, batch_size, batch_size], dim=0)
            loss = criterion(centers, logits, features, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(logits, target, topk=(1, 5))
        losses.update(loss.item(), batch_size)
        top1.update(acc1[0], batch_size)
        top5.update(acc5[0], batch_size)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()

def train_ride(train_loader, model, criterion, optimizer, epoch, args, log, weighted_trainloader = None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    
    # switch to train mode
    model.train()

    if 'CMO' in args.data_aug and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
        inverse_iter = iter(weighted_trainloader)

    model.module.backbone.returns_feat = True
    criterion._hook_before_epoch(epoch)
    
    end = time.time()
    for i, data_tuple in enumerate(train_loader):
        input = data_tuple[0]
        target = data_tuple[1]
        
        if 'CMO' in args.data_aug and args.start_data_aug < epoch < (args.epochs - args.end_data_aug):
            try:
                data_tuple_f = next(inverse_iter)
                input2 = data_tuple_f[0]
                target2 = data_tuple_f[1]
            except:
                inverse_iter = iter(weighted_trainloader)
                data_tuple_f = next(inverse_iter)
                input2 = data_tuple_f[0]
                target2 = data_tuple_f[1]
            input2 = input2[:input.size()[0]]
            target2 = target2[:target.size()[0]]
            input2 = input2.cuda(args.gpu, non_blocking=True)
            target2 = target2.cuda(args.gpu, non_blocking=True)

        data_time.update(time.time() - end)
        input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        r = np.random.rand(1)
        if 'CMO' in args.data_aug and args.start_data_aug < epoch < (args.epochs - args.end_data_aug) and r < args.mixup_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
            input[:, :, bbx1:bbx2, bby1:bby2] = input2[:, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (input.size()[-1] * input.size()[-2]))
            extra_info = {}
            output = model(input)
            extra_info.update({
                "logits": output["logits"].transpose(0, 1)
            })
            output = output['output']
            loss = criterion(output, target, extra_info) * lam + criterion(output, target2) * (1.-lam)
        else:
            extra_info = {}
            output = model(input)
            extra_info.update({
                "logits": output["logits"].transpose(0, 1)
            })
            output = output['output']
            loss = criterion(output, target, extra_info)
            
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            output = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5, lr=optimizer.param_groups[-1]['lr']))  # TODO
            print(output)
            log.write(output + '\n')
            log.flush()
            
def validate(val_loader, model, criterion, train_cls_num_list, args, log, flag='val'):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    # switch to evaluate mode
    model.eval()
    
    if hasattr(model.module, 'backbone'):
        model.module.backbone.returns_feat = False
    
    all_preds = []
    all_targets = []
    with torch.no_grad():
        end = time.time()
        for i, data_tuple in enumerate(val_loader):
            input = data_tuple[0]
            target = data_tuple[1]
        
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            if args.loss_type == 'BCL':
                _, output, _ = model(input)
            elif args.loss_type == 'NCL':
                pass
            else:
                output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            _, pred = torch.max(output, 1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

            if i % args.print_freq == 0:
                output = ('Test: [{0}/{1}]\t'
                          'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                          'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time, loss=losses,
                    top1=top1, top5=top5))
                print(output)
        cf = confusion_matrix(all_targets, all_preds).astype(float)
        cls_cnt = cf.sum(axis=1)
        cls_hit = np.diag(cf)
        cls_acc = cls_hit / cls_cnt
        output = ('{flag} Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
                  .format(flag=flag, top1=top1, top5=top5, loss=losses))
        out_cls_acc = '%s Class Accuracy: %s' % (
        flag, (np.array2string(cls_acc, separator=',', formatter={'float_kind': lambda x: "%.3f" % x})))
        print(output)

        many_shot = train_cls_num_list > 100
        medium_shot = (train_cls_num_list <= 100) & (train_cls_num_list > 20)
        few_shot = train_cls_num_list <= 20
        clswise_output=f"many avg, med avg, few avg  {float(sum(cls_acc[many_shot]) * 100 / sum(many_shot))} \
        {float(sum(cls_acc[medium_shot]) * 100 / sum(medium_shot))} {float(sum(cls_acc[few_shot]) * 100 / sum(few_shot))}"
        print(clswise_output)

        if log is not None:
            log.write(output + '\n')
            log.write(clswise_output + '\n')
            log.flush()
    return top1.avg

def update_score_base(loader, model, n_samples_per_class, cls_pos_list, args):
    model.eval()
    curr_state = loader.dataset.curr_state
    max_state = torch.max(curr_state).int() + 1
    
    if hasattr(model.module, 'backbone') and hasattr(model.module, 'num_experts'):
        model_name = 'ride'
        model.module.backbone.returns_feat = True
    elif hasattr(model.module, 'head_fc'):
        model_name = 'bcl'
    else:
        model_name = 'vanilla'
    
    with torch.no_grad():
        start_time = time.time()

        n = 10
        pos, state = [], []
        for cidx, class_pos in enumerate(cls_pos_list):
            max_state = loader.dataset.curr_state[class_pos[0]].int() 
            for s in range(max_state+1):
                _pos = random.choices(class_pos.tolist(), k = n * (s+1))
                pos += _pos 
                state += [s] * len(_pos)

        tmp_dataset = test_loader(pos, state, loader.dataset)
        tmp_loader = torch.utils.data.DataLoader(tmp_dataset, batch_size=args.batch_size*4, 
                                                 shuffle=False, num_workers=args.workers,
                                                 pin_memory=True)
        
        for batch_idx, data_tuple in tqdm(enumerate(tmp_loader)):
            data = data_tuple[0].cuda(non_blocking=True)

            label = data_tuple[1]
            idx = data_tuple[2]
            state = data_tuple[3]
            
            if model_name == 'ride':
                outputs = model(data)
                logit = outputs['logits'].cpu()
                outputs = model(data)
                logit = outputs['logits'].cpu().float()
                for cor_idx in range(logit.size(1)):
                    if cor_idx == 0:
                        correct = (logit[:,cor_idx].max(dim=1)[1] == label).int().detach().cpu()
                    else:
                        correct += (logit[:,cor_idx].max(dim=1)[1] == label).int().detach().cpu()
                correct = torch.floor(correct/logit.size(1))
            elif model_name == 'bcl':
                _, logit, _ = model(data)
                logit = logit.cpu()
                correct = (logit.max(dim=1)[1] == label).int().detach().cpu()
            else:
                logit = model(data).cpu()
                correct = (logit.max(dim=1)[1] == label).int().detach().cpu()
            loader.dataset.update_scores(correct, idx, state)
    
    correct_sum_per_class = torch.zeros(len(n_samples_per_class))
    trial_sum_per_class = torch.zeros(len(n_samples_per_class))
#     for cidx in range(len(n_samples_per_class)):
#         class_pos = torch.where(torch.tensor(loader.dataset.targets) == cidx)[0]
    for cidx, class_pos in enumerate(cls_pos_list):
        
        correct_sum_row = torch.sum(loader.dataset.score_tmp[class_pos], dim=0)
        trial_sum_row = torch.sum(loader.dataset.num_test[class_pos], dim=0)


        ratio = correct_sum_row / trial_sum_row 
        idx = loader.dataset.curr_state[class_pos][0].int() + 1
        condition = torch.sum((ratio[:idx] > 0.4)) == idx 
        
        if condition:
            loader.dataset.curr_state[class_pos] += 1
        else:
            loader.dataset.curr_state[class_pos] -= 1
            
        loader.dataset.curr_state = loader.dataset.curr_state.clamp(loader.dataset.min_state, loader.dataset.max_state-1)
        loader.dataset.score_tmp *= 0
        loader.dataset.num_test *= 0
    
    print(f'Estimate time: {hms_string(time.time()-start_time)}')
#     print(f'Max correct: {int(torch.max(loader.dataset.score_tmp))} Max trial: {int(torch.max(loader.dataset.num_test))}')
#     loader.dataset.update()
    model.train()
    
    # Debug
    curr_state = loader.dataset.curr_state
    label = loader.dataset.targets
    print(f'Max state: {int(torch.max(curr_state))} // Min state: {int(torch.min(curr_state))}')
    
    return curr_state, label
