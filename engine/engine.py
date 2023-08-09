import torch
import numpy as np
import cv2
from apex import amp
from ignite.engine import Engine
from ignite.engine import Events
from torch.autograd import no_grad
from utils.calc_acc import calc_acc
from torch.nn import functional as F
import os

def create_train_engine(model, optimizer, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())
    # global now_epoch  # 聲明 now_epoch 為全局變數
    # now_epoch = 0
    def _process_func(engine, batch):
        # global now_epoch
        model.train()
        data, labels, cam_ids, img_paths, img_ids= batch
        pair_labels = (labels[::2] != labels[1::2]).long()
        # print(labels)
        epoch = engine.state.epoch
        data = data.to(device, non_blocking=non_blocking)
        labels = labels.to(device, non_blocking=non_blocking)
        pair_labels = pair_labels.to(device, non_blocking=non_blocking)
        cam_ids = cam_ids.to(device, non_blocking=non_blocking)
        optimizer.zero_grad()
        loss, metric = model(data, labels,pair_labels,
                             cam_ids=cam_ids,
                             epoch=epoch)

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()

        return metric

    return Engine(_process_func)


def create_eval_engine(model, non_blocking=False):
    device = torch.device("cuda", torch.cuda.current_device())

    def _process_func(engine, batch):
        model.eval()

        data, labels, cam_ids, img_paths = batch[:4]

        data = data.to(device, non_blocking=non_blocking)

        with no_grad():
            feat = model(data, cam_ids=cam_ids.to(device, non_blocking=non_blocking))

        return feat.data.float().cpu(), labels, cam_ids, np.array(img_paths)

    engine = Engine(_process_func)

    @engine.on(Events.EPOCH_STARTED)
    def clear_data(engine):
        # feat list
        if not hasattr(engine.state, "feat_list"):
            setattr(engine.state, "feat_list", [])
        else:
            engine.state.feat_list.clear()

        # id_list
        if not hasattr(engine.state, "id_list"):
            setattr(engine.state, "id_list", [])
        else:
            engine.state.id_list.clear()

        # cam list
        if not hasattr(engine.state, "cam_list"):
            setattr(engine.state, "cam_list", [])
        else:
            engine.state.cam_list.clear()

        # img path list
        if not hasattr(engine.state, "img_path_list"):
            setattr(engine.state, "img_path_list", [])
        else:
            engine.state.img_path_list.clear()

    @engine.on(Events.ITERATION_COMPLETED)
    def store_data(engine):
        engine.state.feat_list.append(engine.state.output[0])
        engine.state.id_list.append(engine.state.output[1])
        engine.state.cam_list.append(engine.state.output[2])
        engine.state.img_path_list.append(engine.state.output[3])
    return engine
