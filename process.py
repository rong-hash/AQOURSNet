import torch
from torch.cuda.amp import autocast, GradScaler
import xgboost as xgb
from tqdm import tqdm
from utils import write_log, prf_score

def train_epoch(model, loader, loss_func, optimizer, args):
    model.train()
    if args.amp: scaler = GradScaler()
    losses, ncorrect, ntot = 0., 0, 0
    for batch in loader:
        batch = batch.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_func(output, batch.y)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        losses += loss.item() * len(batch.y)
        ncorrect += (output.argmax(dim=1) == batch.y).sum().item()
        ntot += len(batch.y)
    torch.cuda.empty_cache()
    return losses / ntot, ncorrect / ntot

def train_epoch_prf(model, loader, loss_func, optimizer, args):
    model.train()
    if args.amp: scaler = GradScaler()
    losses, preds, recls, f1s, ntot = 0., 0., 0., 0., 0
    for batch in loader:
        batch = batch.to(args.device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=args.amp):
            output = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_func(output, batch.y)
            pred, recl, f1 = prf_score(output, batch.y)
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        nsample = len(batch.y)
        losses += loss.item() * nsample
        preds += pred.item() * nsample
        recls += recl.item() * nsample
        f1s += f1.item() * nsample
        ntot += nsample
    torch.cuda.empty_cache()
    return losses / ntot, (preds / ntot, recls / ntot, f1s / ntot)

@torch.no_grad()
def test_epoch(model, loader, args):
    model.eval()
    ncorrect, ntot = 0, 0
    for batch in loader:
        batch = batch.to(args.device)
        output = model(batch.x, batch.edge_index, batch.batch)
        ncorrect += (output.argmax(dim=1) == batch.y).sum().item()
        ntot += len(batch.y)
    torch.cuda.empty_cache()
    return ncorrect / ntot

@torch.no_grad()
def test_epoch_prf(model, loader, args):
    model.eval()
    preds, recls, f1s, ntot = 0., 0., 0., 0
    for batch in loader:
        batch = batch.to(args.device)
        output = model(batch.x, batch.edge_index, batch.batch)
        pred, recl, f1 = prf_score(output, batch.y)
        nsample = len(batch.y)
        preds += pred.item() * nsample
        recls += recl.item() * nsample
        f1s += f1.item() * nsample
        ntot += nsample
    torch.cuda.empty_cache()
    return (preds / ntot, recls / ntot, f1s / ntot)

def process(model, train_loader, test_loader, loss_func, optimizer, args):
    tqdm_meter = tqdm(desc='[Training GAT]')
    write_log(args.dirlog, '\n'+str(args.__dict__)+'\n')
    write_log(args.dirlog, 'Epoch,Loss,TrainAcc,TestAcc\n')
    best_train_acc, best_test_acc = 0., 0.
    for epoch in range(args.nepoch):
        loss, train_acc = train_epoch(model, train_loader, loss_func, optimizer, args)
        test_acc = test_epoch(model, test_loader, args)
        tqdm_meter.set_postfix(
            Epoch='%3d' % (epoch + 1),
            Loss ='%.6f' % loss,
            TrainAcc='%6.2f%%' % (train_acc * 100),
            TestAcc ='%6.2f%%' % (test_acc * 100))
        tqdm_meter.update()
        write_log(args.dirlog, '%03d,%.6f,%6.2f%%,%6.2f%%\n'
                    % (epoch + 1, loss, train_acc, test_acc))
        if test_acc > best_test_acc \
            or (test_acc == best_test_acc and train_acc > best_train_acc):
            best_test_acc, best_train_acc = test_acc, train_acc
            torch.save(model, args.dirmodel)
        torch.cuda.empty_cache()
    tqdm_meter.close()

def process_prf(model, train_loader, test_loader, loss_func, optimizer, args):
    tqdm_meter = tqdm(desc='[Training GAT]')
    write_log(args.dirlog, '\n'+str(args.__dict__)+'\n')
    write_log(args.dirlog, 'Epoch,Loss,TrainPred,TrainRecall,TrainF1,TestPred,TestRecall,TestF1\n')
    best_train_f1, best_test_f1 = 0., 0.
    for epoch in range(args.nepoch):
        loss, train_prf = train_epoch_prf(model, train_loader, loss_func, optimizer, args)
        test_prf = test_epoch_prf(model, test_loader, args)
        tqdm_meter.set_postfix(
            Epoch='%3d' % (epoch + 1),
            Loss ='%.6f' % loss,
            TrainPred='%6.2f%%' % train_prf[0],
            TrainRecl='%6.2f%%' % train_prf[1],
            TrainF1  ='%6.2f%%' % train_prf[2],
            TestPred ='%6.2f%%' % test_prf[0],
            TestRecl ='%6.2f%%' % test_prf[1],
            TestF1   ='%6.2f%%' % test_prf[2])
        tqdm_meter.update()
        write_log(args.dirlog, '%03d,%.6f,%6.2f%%,%6.2f%%,%6.2f%%,%6.2f%%,%6.2f%%,%6.2f%%\n'
                    % (epoch + 1, loss, train_prf[0], train_prf[1], train_prf[2], \
                        test_prf[0], test_prf[2], test_prf[2]))
        if test_prf[2] > best_test_f1 \
            or (test_prf[2] == best_test_f1 and train_prf[2] > best_train_f1):
            best_test_f1, best_train_f1 = test_prf[2], train_prf[2]
            torch.save(model, args.dirmodel)
        torch.cuda.empty_cache()
    tqdm_meter.close()
