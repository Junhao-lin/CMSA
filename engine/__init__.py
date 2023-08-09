import logging
import os
import numpy as np
import torch
import scipy.io as sio
import wandb
from ignite.engine import Events
from ignite.handlers import ModelCheckpoint
from ignite.handlers import Timer

from engine.engine import create_eval_engine
from engine.engine import create_train_engine
from engine.metric import AutoKVMetric
from utils.eval_sysu import eval_sysu
from utils.eval_regdb import eval_regdb
from configs.default.dataset import dataset_cfg
from configs.default.strategy import strategy_cfg

def get_trainer(dataset, model, optimizer, lr_scheduler=None, logger=None, writer=None, non_blocking=False, log_period=10,
                save_dir="checkpoints", prefix="model", gallery_loader=None, query_loader=None,
                eval_interval=None, start_eval=None, rerank=False):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.WARN)

    # trainer
    trainer = create_train_engine(model, optimizer, non_blocking)
    
    setattr(trainer, "rerank", rerank)

    # checkpoint handler
    handler = ModelCheckpoint(save_dir, prefix, save_interval=eval_interval, n_saved=3, create_dir=True,
                              save_as_state_dict=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"model": model})

    # metric
    timer = Timer(average=True)

    kv_metric = AutoKVMetric()

    # evaluator
    evaluator = None
    if not type(eval_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if not type(start_eval) == int:
        raise TypeError("The parameter 'start_eval' must be type INT.")
    if eval_interval > 0 and gallery_loader is not None and query_loader is not None:
        evaluator = create_eval_engine(model, non_blocking)

    @trainer.on(Events.STARTED)
    def train_start(engine):
        setattr(engine.state, "best_rank1", 0.0)

    @trainer.on(Events.COMPLETED)
    def train_completed(engine):
        torch.cuda.empty_cache()

        # extract query feature
        evaluator.run(query_loader)

        q_feats = torch.cat(evaluator.state.feat_list, dim=0)
        q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

        # extract gallery feature
        evaluator.run(gallery_loader)

        g_feats = torch.cat(evaluator.state.feat_list, dim=0)
        g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

        # print("best rank1={:.2f}%".format(engine.state.best_rank1))

        if dataset == 'sysu':
            perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                'rand_perm_cam']
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=engine.rerank)
        elif dataset == 'regdb':
            print('infrared to visible')
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)
            print('visible to infrared')
            eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=engine.rerank)
        elif dataset == 'market':
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)


        evaluator.state.feat_list.clear()
        evaluator.state.id_list.clear()
        evaluator.state.cam_list.clear()
        evaluator.state.img_path_list.clear()
        del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

        torch.cuda.empty_cache()

    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started_callback(engine):
    
        epoch = engine.state.epoch
        if model.mutual_learning:
            model.update_rate = min(100 / (epoch + 1), 1.0) * model.update_rate_

        kv_metric.reset()
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_callback(engine):
        epoch = engine.state.epoch

        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % eval_interval == 0:
            logger.info("Model saved at {}/{}_model_{}.pth".format(save_dir, prefix, epoch))

        if evaluator and epoch % eval_interval == 0 and epoch > start_eval:
            torch.cuda.empty_cache()

            # extract query feature
            evaluator.run(query_loader)

            q_feats = torch.cat(evaluator.state.feat_list, dim=0)
            q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
            q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

            # extract gallery feature
            evaluator.run(gallery_loader)

            g_feats = torch.cat(evaluator.state.feat_list, dim=0)
            g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
            g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)

            if dataset == 'sysu':
                perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                    'rand_perm_cam']
                a1_mAP, a1_r1, a1_r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=engine.rerank)
                a10_mAP, a10_r1, a10_r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=engine.rerank)
                i1_mAP, i1_r1, i1_r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=engine.rerank)
                i10_mAP, i10_r1, i10_r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=engine.rerank)
                wandb.log({'a1_mAP': a1_mAP,
                           'a1_r1': a1_r1,
                           'a10_mAP': a10_mAP,
                           'a10_r1': a10_r1,
                           'i1_mAP':i1_mAP,
                           'i1_r1':i1_r1,
                           'i10_mAP':i10_mAP,
                           'i10_r1':i10_r1
                           })
                average_r1 = (a1_r1 + a10_r1 + i1_r1 + i10_r1)/4
                wandb.log({"average r1": average_r1})
            elif dataset == 'regdb':
                print('infrared to visible')
                mAP, r1, r5, r10, r20 = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)
                print('visible to infrared')
                mAP_, r1_, r5_, r10_, r20_ = eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=engine.rerank)
                wandb.log({'i2v mAP': mAP,
                           'i2v r1': r1,
                           'i2v r10': r10,
                           'i2v r20': r20,
                           'v2i mAP': mAP_,
                           'v2i r1': r1_,
                           'v2i r10': r10_,
                           'v2i r20': r20_
                           })
                average_r1 = (r1 + r1_) / 2
                wandb.log({"average r1": average_r1})
            elif dataset == 'market':
                mAP, r1, r5, _, _ = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)
            
            if average_r1 > engine.state.best_rank1:
                print('best epoch:',engine.state.epoch)
                engine.state.best_rank1 = average_r1
                torch.save(model.state_dict(), "{}/model_best.pth".format(save_dir))

            if writer is not None:
                writer.add_scalar('eval/mAP', mAP, epoch)
                writer.add_scalar('eval/r1', r1, epoch)
                writer.add_scalar('eval/r5', r5, epoch)

            evaluator.state.feat_list.clear()
            evaluator.state.id_list.clear()
            evaluator.state.cam_list.clear()
            evaluator.state.img_path_list.clear()
            del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

            torch.cuda.empty_cache()

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_complete_callback(engine):
        timer.step()
        kv_metric.update(engine.state.output)
        epoch = engine.state.epoch
        iteration = engine.state.iteration
        iter_in_epoch = iteration - (epoch - 1) * len(engine.state.dataloader)
        # if iter_in_epoch % log_period == 0 and iter_in_epoch > 0:
        if iter_in_epoch == log_period:
            batch_size = engine.state.batch[0].size(0)
            speed = batch_size / timer.value()
            msg = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (epoch, iter_in_epoch, speed)
            metric_dict = kv_metric.compute()
            wandb.log(metric_dict)
            # log output information
            if logger is not None:
                for k in sorted(metric_dict.keys()):
                    msg += "\t%s: %.4f" % (k, metric_dict[k])
                    if writer is not None:
                        writer.add_scalar('metric/{}'.format(k), metric_dict[k], iteration)
                logger.info(msg)

            kv_metric.reset()
            timer.reset()
    return trainer

def get_trainer_new(dataset, model, optimizer, lr_scheduler=None, logger=None, writer=None, non_blocking=False, log_period=10,
                save_dir="checkpoints", prefix="model", test_loader=None, eval_interval=None, start_eval=None, rerank=False):
    if logger is None:
        logger = logging.getLogger()
        logger.setLevel(logging.WARN)

    # trainer
    trainer = create_train_engine(model, optimizer, non_blocking)
    
    setattr(trainer, "rerank", rerank)

    # checkpoint handler
    handler = ModelCheckpoint(save_dir, prefix, save_interval=eval_interval, n_saved=3, create_dir=True,
                              save_as_state_dict=True, require_empty=False)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, handler, {"model": model})

    # metric
    timer = Timer(average=True)

    kv_metric = AutoKVMetric()

    # evaluator
    evaluator = None
    if not type(eval_interval) == int:
        raise TypeError("The parameter 'validate_interval' must be type INT.")
    if not type(start_eval) == int:
        raise TypeError("The parameter 'start_eval' must be type INT.")
    if eval_interval > 0 and test_loader is not None :
        evaluator = create_eval_engine(model, non_blocking)

    @trainer.on(Events.STARTED)
    def train_start(engine):
        setattr(engine.state, "best_rank1", 0.0)

    @trainer.on(Events.COMPLETED)
    def train_completed(engine):
        torch.cuda.empty_cache()
        evaluator.run(test_loader)
        
        q_feats = torch.cat(evaluator.state.feat_list, dim=0)
        q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)
        q_feats = q_feats[::2]
        q_ids = q_ids[::2]
        q_cams = q_cams[::2]
        q_img_paths = q_img_paths[::2]

        g_feats = torch.cat(evaluator.state.feat_list, dim=0)
        g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
        g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
        g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)
        g_feats = g_feats[1::2]
        g_ids = g_ids[1::2]
        g_cams = g_cams[1::2]
        g_img_paths = g_img_paths[1::2]

        if dataset == 'sysu':
            perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                'rand_perm_cam']
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=engine.rerank)
            eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=engine.rerank)
        elif dataset == 'regdb':
            print('infrared to visible')
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)
            print('visible to infrared')
            eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=engine.rerank)
        elif dataset == 'market':
            eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)


        evaluator.state.feat_list.clear()
        evaluator.state.id_list.clear()
        evaluator.state.cam_list.clear()
        evaluator.state.img_path_list.clear()
        del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

        torch.cuda.empty_cache()

    @trainer.on(Events.EPOCH_STARTED)
    def epoch_started_callback(engine):
    
        epoch = engine.state.epoch
        if model.mutual_learning:
            model.update_rate = min(100 / (epoch + 1), 1.0) * model.update_rate_

        kv_metric.reset()
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def epoch_completed_callback(engine):
        epoch = engine.state.epoch

        if lr_scheduler is not None:
            
            lr_scheduler.step()

        if epoch % eval_interval == 0:
            logger.info("Model saved at {}/{}_model_{}.pth".format(save_dir, prefix, epoch))

        if evaluator and epoch % eval_interval == 0 and epoch > start_eval:
            torch.cuda.empty_cache()
            evaluator.run(test_loader)
            
            q_feats = torch.cat(evaluator.state.feat_list, dim=0)
            q_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            q_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
            q_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)
            q_feats = q_feats[::2]
            q_ids = q_ids[::2]
            q_cams = q_cams[::2]
            q_img_paths = q_img_paths[::2]

            g_feats = torch.cat(evaluator.state.feat_list, dim=0)
            g_ids = torch.cat(evaluator.state.id_list, dim=0).numpy()
            g_cams = torch.cat(evaluator.state.cam_list, dim=0).numpy()
            g_img_paths = np.concatenate(evaluator.state.img_path_list, axis=0)
            g_feats = g_feats[1::2]
            g_ids = g_ids[1::2]
            g_cams = g_cams[1::2]
            g_img_paths = g_img_paths[1::2]
            
            if dataset == 'sysu':
                perm = sio.loadmat(os.path.join(dataset_cfg.sysu.data_root, 'exp', 'rand_perm_cam.mat'))[
                    'rand_perm_cam']
                a1_mAP, a1_r1, a1_r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=1, rerank=engine.rerank)
                a10_mAP, a10_r1, a10_r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='all', num_shots=10, rerank=engine.rerank)
                i1_mAP, i1_r1, i1_r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=1, rerank=engine.rerank)
                i10_mAP, i10_r1, i10_r5, _, _ = eval_sysu(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, perm, mode='indoor', num_shots=10, rerank=engine.rerank)
                wandb.log({'a1_mAP': a1_mAP,
                           'a1_r1': a1_r1,
                           'a10_mAP': a10_mAP,
                           'a10_r1': a10_r1,
                           'i1_mAP':i1_mAP,
                           'i1_r1':i1_r1,
                           'i10_mAP':i10_mAP,
                           'i10_r1':i10_r1
                           })
                r1 = (a1_r1 + a10_r1 + i1_r1 + i10_r1)/4
            elif dataset == 'regdb':
                print('infrared to visible')
                mAP, r1, r5, r10, r20 = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)
                print('visible to infrared')
                mAP_, r1_, r5_, r10_, r20_ = eval_regdb(g_feats, g_ids, g_cams, q_feats, q_ids, q_cams, q_img_paths, rerank=engine.rerank)
                wandb.log({'i2v mAP': mAP,
                           'i2v r1': r1,
                           'i2v r10': r10,
                           'i2v r20': r20,
                           'v2i mAP': mAP_,
                           'v2i r1': r1_,
                           'v2i r10': r10_,
                           'v2i r20': r20_
                           })
                r1 = (r1 + r1_) / 2
            elif dataset == 'market':
                mAP, r1, r5, _, _ = eval_regdb(q_feats, q_ids, q_cams, g_feats, g_ids, g_cams, g_img_paths, rerank=engine.rerank)
            
            if r1 > engine.state.best_rank1:
                print('best epoch:',engine.state.epoch)
                engine.state.best_rank1 = r1
                torch.save(model.state_dict(), "{}/model_best.pth".format(save_dir))

            if writer is not None:
                writer.add_scalar('eval/mAP', mAP, epoch)
                writer.add_scalar('eval/r1', r1, epoch)
                writer.add_scalar('eval/r5', r5, epoch)

            evaluator.state.feat_list.clear()
            evaluator.state.id_list.clear()
            evaluator.state.cam_list.clear()
            evaluator.state.img_path_list.clear()
            del q_feats, q_ids, q_cams, g_feats, g_ids, g_cams

            torch.cuda.empty_cache()

    @trainer.on(Events.ITERATION_COMPLETED)
    def iteration_complete_callback(engine):
        timer.step()
        # print(engine.state.output)
        kv_metric.update(engine.state.output)
        epoch = engine.state.epoch
        iteration = engine.state.iteration
        iter_in_epoch = iteration - (epoch - 1) * len(engine.state.dataloader)
        print("---------------------------------------")
        # if iter_in_epoch % log_period == 0 and iter_in_epoch > 0:
        if iter_in_epoch == log_period:
            batch_size = engine.state.batch[0].size(0)
            speed = batch_size / timer.value()
            msg = "Epoch[%d] Batch [%d]\tSpeed: %.2f samples/sec" % (epoch, iter_in_epoch, speed)
            metric_dict = kv_metric.compute()
            wandb.log(metric_dict)
            # log output information
            if logger is not None:
                for k in sorted(metric_dict.keys()):
                    msg += "\t%s: %.4f" % (k, metric_dict[k])
                    if writer is not None:
                        writer.add_scalar('metric/{}'.format(k), metric_dict[k], iteration)
                logger.info(msg)

            kv_metric.reset()
            timer.reset()
    return trainer
