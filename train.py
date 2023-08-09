import logging
import os
import pprint
import wandb
import torch
import yaml
from apex import amp
from torch import optim
import cv2
from data import get_test_loader
from data import get_test_loader_new
from data import get_train_loader
from engine import get_trainer
from engine import get_trainer_new
from models.baseline import Baseline
import re
from pytorch_grad_cam import GradCAM, \
    HiResCAM, \
    ScoreCAM, \
    GradCAMPlusPlus, \
    AblationCAM, \
    XGradCAM, \
    EigenCAM, \
    EigenGradCAM, \
    LayerCAM, \
    FullGrad, \
    GradCAMElementWise
import gc
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, \
    deprocess_image, \
    preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from WarmUpLR import WarmUpStepLR

def train(cfg): 
    # set logger
    log_dir = os.path.join("logs/", cfg.dataset, cfg.prefix)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(format="%(asctime)s %(message)s",
                        filename=log_dir + "/" + "log.txt",
                        filemode="w")

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info(pprint.pformat(cfg))

    # training data loader
    train_loader = get_train_loader(dataset=cfg.dataset,
                                    root=cfg.data_root,
                                    sample_method=cfg.sample_method,
                                    batch_size=cfg.batch_size,
                                    p_size=cfg.p_size,
                                    k_size=cfg.k_size,
                                    random_flip=cfg.random_flip,
                                    random_crop=cfg.random_crop,
                                    random_erase=cfg.random_erase,
                                    color_jitter=cfg.color_jitter,
                                    padding=cfg.padding,
                                    image_size=cfg.image_size,
                                    num_workers=8)

    # evaluation data loader
    gallery_loader, query_loader = None, None
    if cfg.eval_interval > 0 and cfg.new_test== False:
        gallery_loader, query_loader = get_test_loader(dataset=cfg.dataset,
                                                       root=cfg.data_root,
                                                       batch_size=128,
                                                       image_size=cfg.image_size,
                                                       num_workers=8)

    #------------------------------------------------------- Louis add
    if cfg.eval_interval > 0 and cfg.new_test== True:
        test_loader = get_test_loader_new(dataset=cfg.dataset,
                                        root=cfg.data_root,
                                        batch_size=cfg.batch_size,
                                        p_size=cfg.p_size,
                                        k_size=cfg.k_size,
                                        image_size=cfg.image_size,
                                        num_workers=8)

    # model
    model = Baseline(num_classes=cfg.num_id,
                     pattern_attention=cfg.pattern_attention,
                     modality_attention=cfg.modality_attention,
                     mutual_learning=cfg.mutual_learning,
                     drop_last_stride=cfg.drop_last_stride,
                     triplet=cfg.triplet,
                     k_size=cfg.k_size,
                     center_cluster=cfg.center_cluster,
                     center=cfg.center,
                     margin=cfg.margin,
                     num_parts=cfg.num_parts,
                     weight_KL=cfg.weight_KL,
                     weight_sid=cfg.weight_sid,
                     weight_sep=cfg.weight_sep,
                     update_rate=cfg.update_rate,
                     pair=cfg.pair,
                     classification=cfg.classification)

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}

    print(get_parameter_number(model))

    model.cuda()
    # optimizer
    assert cfg.optimizer in ['adam', 'sgd']
    if cfg.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    else:
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=cfg.wd)

    # convert model for mixed precision training
    model, optimizer = amp.initialize(model, optimizer, enabled=cfg.fp16, opt_level="O1")
    if cfg.center:
        model.center_loss.centers = model.center_loss.centers.float()
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer,
                                                  milestones=cfg.lr_step,
                                                  gamma=0.1)

    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        model.load_state_dict(checkpoint)

    # engine
    checkpoint_dir = os.path.join("checkpoints", cfg.dataset, cfg.prefix)
    if  cfg.new_test== False:
        engine = get_trainer(dataset=cfg.dataset,
                            model=model,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            logger=logger,
                            non_blocking=True,
                            log_period=cfg.log_period,
                            save_dir=checkpoint_dir,
                            prefix=cfg.prefix,
                            eval_interval=cfg.eval_interval,
                            start_eval=cfg.start_eval,
                            gallery_loader=gallery_loader,
                            query_loader=query_loader,
                            rerank=cfg.rerank)
    else:
        engine = get_trainer_new(dataset=cfg.dataset,
                            model=model,
                            optimizer=optimizer,
                            lr_scheduler=lr_scheduler,
                            logger=logger,
                            non_blocking=True,
                            log_period=cfg.log_period,
                            save_dir=checkpoint_dir,
                            prefix=cfg.prefix,
                            eval_interval=cfg.eval_interval,
                            start_eval=cfg.start_eval,
                            test_loader=test_loader,
                            rerank=cfg.rerank)
    # training
    engine.run(train_loader, max_epochs=cfg.num_epoch)
if __name__ == '__main__':
    import argparse
    import random
    import numpy as np
    from configs.default import strategy_cfg
    from configs.default import dataset_cfg

    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="configs/softmax.yml")
    args = parser.parse_args() 

    # load configuration
    customized_cfg = yaml.load(open(args.cfg, "r"), Loader=yaml.SafeLoader)

    cfg = strategy_cfg
    cfg.merge_from_file(args.cfg)

    dataset_cfg = dataset_cfg.get(cfg.dataset)
    for k, v in dataset_cfg.items():
        cfg[k] = v

    if cfg.sample_method == 'identity_uniform':
        cfg.batch_size = cfg.p_size * cfg.k_size

    cfg.freeze()
# set random seed
    if cfg.dataset == 'regdb':
        seed = 1         #RegDB
    else:
        seed = 1         #SYSU
    random.seed(seed)
    np.random.RandomState(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # enable cudnn backend
    os.environ['PYTHONHASHSEED'] = '0'    
    torch.backends.cudnn.enabled = True        
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if cfg.dataset == 'regdb':
        wandb.init(
        # project="test",
        project="RegDB",
        name=f'{cfg.prefix}'
        )
    else:
        wandb.init(
        # project="test",
        project="SYSU",
        name=f'{cfg.prefix}'
        )
    train(cfg)