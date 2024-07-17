from datetime import datetime
import time

import tqdm
import torch
from torch.nn.utils import clip_grad_norm_

import utils
from checkpoint import save_checkpoint
import criteria

def train_model(model, 
                train_loader, val_loader, 
                crit, opt, train_conf, last_epoch, 
                last_best_loss = None, 
                last_best_error_t = None, last_best_error_r = None,
                scheduler = None, trainlog=True, DEVICE='cpu', GRAD_CLIP=1.0,
                MODEL_CONFIG_CL=None, LOAD_MODEL=None):
    
    MODEL_NAME = MODEL_CONFIG_CL.model_name
    MODEL_DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
    if LOAD_MODEL == False:
        BEST_TRAIN_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_train.pth.tar"
        BEST_VAL_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_val.pth.tar"
        BEST_ROT_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_rot.pth.tar"
        BEST_TRANS_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_trans.pth.tar"
    else:
        BEST_TRAIN_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_train.pth.tar"
        BEST_VAL_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_val.pth.tar"
        BEST_ROT_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_rot.pth.tar"
        BEST_TRANS_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_trans.pth.tar"
    
    reg_loss, rot_loss, pcd_loss = crit
    # reg_loss, rot_loss = crit

    loss_treshold = train_conf.loss_treshold
    early_stop_patience = train_conf.early_stop_patience
    treshold_count = 0

    start_time = time.time()
    st_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Start time: ', datetime.now())

    model.train()
    for j in range(train_conf.n_epochs):
        print("===================================")
        print(f'Epoch: {j+1+last_epoch} or {j+1}')
        epoch_start = time.time()

        loss_epoch = 0
        trans_loss_epoch = 0
        rot_loss_epoch = 0
        pcd_loss_epoch = 0

        # process = tqdm(train_loader, unit='batch')

        for _, batch_data in enumerate(train_loader):
        # print(i, batch_data)

            T_gt = [sample["T_gt"].to(DEVICE) for sample in batch_data]
            rgb_img = [sample["img"].to(DEVICE) for sample in batch_data]
            depth_img = [sample["depth_img_error"].to(DEVICE) for sample in batch_data]
            delta_q_gt = [sample["delta_q_gt"].to(DEVICE) for sample in batch_data]
            delta_t_gt = [sample["delta_t_gt"].to(DEVICE) for sample in batch_data]
            pcd_mis = [sample["pcd_mis"].to(DEVICE) for sample in batch_data]
            pcd_gt = [sample["pcd_gt"].to(DEVICE) for sample in batch_data]

            T_gt = torch.stack(T_gt, dim=0)
            rgb_img = torch.stack(rgb_img, dim=0) # correct shape
            depth_img = torch.stack(depth_img, dim=0) # correct shape
            delta_q_gt = torch.stack(delta_q_gt, dim=0)
            delta_t_gt = torch.stack(delta_t_gt, dim=0)
            targets = torch.cat((delta_q_gt, delta_t_gt), 1) # correct shape
            # print('targets shape: ', targets.shape)

            T_mis_batch = torch.tensor([]).to(DEVICE)

            for i in range(targets.shape[0]):
                delta_R_gt  = utils.qua2rot_torch(delta_q_gt[i])
                delta_tr_gt = torch.reshape(delta_t_gt[i],(3,1))
                delta_T_gt  = torch.hstack((delta_R_gt, delta_tr_gt)) 
                delta_T_gt  = torch.vstack((delta_T_gt, torch.Tensor([0., 0., 0., 1.]).to(DEVICE)))

                T_mis = torch.unsqueeze(torch.matmul(delta_T_gt, T_gt[i]), 0)
                T_mis_batch = torch.cat((T_mis_batch, T_mis), 0)

            # print(rgb_img.shape, i)

            opt.zero_grad()

            pcd_pred, _, delta_q_pred, delta_t_pred = model(rgb_img, depth_img, pcd_mis, T_mis_batch)

            # print(rgb_img.shape, depth_img.shape, len(pcd), T_gt.shape, T_mis_batch.shape)
            # print(len(pcd_true), len(pcd_pred), delta_q_pred.shape, delta_t_pred.shape)
            # print(pcd[1].shape, pcd_true[1].shape, pcd_pred[1].shape)
            # print(pcd[2].shape, pcd_true[2].shape, pcd_pred[2].shape)
            # print(pcd[3].shape, pcd_true[3].shape, pcd_pred[3].shape)
            
            # Loss calculation and backprop
            translational_loss = reg_loss(delta_q_gt, delta_t_gt, delta_q_pred, delta_t_pred)
            rotational_loss = rot_loss(delta_q_gt, delta_q_pred)
            pointcloud_loss = pcd_loss(pcd_gt, pcd_pred)
            loss = translational_loss + rotational_loss + pointcloud_loss

            # print(f'L1 = {translational_loss}| L2 = {rotational_loss} | L3 = {pointcloud_loss}')
            # loss = crit(output, targets)
            # # print('loss shape: ', loss.shape)
            loss.backward()

            # gradient clipping
            if GRAD_CLIP is not None:
                clip_grad_norm_(model.parameters(), GRAD_CLIP)

            opt.step()

            loss_epoch += loss.item()
            trans_loss_epoch += translational_loss.item()
            rot_loss_epoch += rotational_loss.item()
            pcd_loss_epoch += pointcloud_loss.item()  
            # print('current loss: ', loss_epoch)

            # experiment.log_metric("batch loss", loss.item()/targets.shape[0])
            # experiment.log_metric("batch trans loss",  translational_loss.item()/targets.shape[0])
            # experiment.log_metric("batch rot loss", rotational_loss.item()/targets.shape[0])
            # experiment.log_metric("batch pcd loss", pointcloud_loss.item()/targets.shape[0])

            # process.set_postfix(loss=loss.item())
            # print('batch_no: ', i)
        train_time = time.time() - epoch_start
        
        # loss_epoch /= len(train_loader)
        trans_loss_epoch /= len(train_loader)
        rot_loss_epoch /= len(train_loader)
        pcd_loss_epoch /= len(train_loader)
        
        val_start = time.time()
        val_loss, error_t, error_r = validate(model, val_loader, crit, DEVICE)
        val_time = time.time() - val_start

        print(f'L1 = {trans_loss_epoch} | L2 = {rot_loss_epoch} | L3 = {pcd_loss_epoch}')
        print(f'Loss Train: {loss_epoch} | Loss Val: {val_loss} | LR: {opt.param_groups[0]["lr"]}')
        print(f'Training time: {train_time} | Validation time: {val_time}')

        if scheduler is not None:
            scheduler.step(val_loss)
        
        # save checkpoint with the best validation score
        if val_loss/len(val_loader) < best_val_loss:
            print("best val loss achieved")
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer:": opt.state_dict(),
                          "epoch": j+last_epoch,
                          "loss": val_loss/len(val_loader),
                          "rot_error": error_r,
                          "trans_error": error_t}
            save_checkpoint(checkpoint, filename=BEST_VAL_CHECKPOINT_DIR)
            best_val_loss = val_loss/len(val_loader)

        # save checkpoint with the best training score
        if loss_epoch/len(train_loader) < best_train_loss:
            print("best train loss achieved")
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer:": opt.state_dict(),
                          "epoch": j+last_epoch,
                          "loss": loss_epoch/len(train_loader),
                          "rot_error": error_r,
                          "trans_error": error_t}
            save_checkpoint(checkpoint, filename=BEST_TRAIN_CHECKPOINT_DIR)
            best_train_loss = loss_epoch/len(train_loader)

        # save checkpoint with the best translational alignment metric
        if error_t < best_error_t:
            print("best translation alignment achieved")
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer:": opt.state_dict(),
                          "epoch": j+last_epoch,
                          "loss": val_loss,
                          "rot_error": error_r,
                          "trans_error": error_t}
            save_checkpoint(checkpoint, filename=BEST_TRANS_CHECKPOINT_DIR)
            best_error_t = error_t

        # save checkpoint with the best rotational alignment metric
        if error_r < best_error_r:
            print("best rotation alignment achieved")
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer:": opt.state_dict(),
                          "epoch": j+last_epoch,
                          "loss": val_loss,
                          "rot_error": error_r,
                          "trans_error": error_t}
            save_checkpoint(checkpoint, filename=BEST_ROT_CHECKPOINT_DIR)
            best_error_r = error_r

        # experiment.log_metric("Training total loss", loss_epoch)
        # experiment.log_metric("Training avg loss", loss_epoch/len(train_loader))
        # experiment.log_metric("Validation total loss", val_loss)
        # experiment.log_metric("Validation avg loss", val_loss/len(val_loader))
        # experiment.log_metric('learning rate', opt.param_groups[0]["lr"])
        
        # if trainlog:
        #     with open(csv_dir, 'a', newline='') as file:
        #         writer = csv.writer(file)
        #         writer.writerow([j+1+last_epoch, j+1,
        #                         trans_loss_epoch, rot_loss_epoch, pcd_loss_epoch,
        #                         loss_epoch, val_loss,
        #                         error_t, error_r,
        #                         opt.param_groups[0]["lr"], 
        #                         train_time, val_time])

        if val_loss <= loss_treshold:
            treshold_count += 1
        
        if treshold_count == early_stop_patience:
            break

    print('Start time: ', st_time, '\nFinished time: ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Overall training time: ', (time.time()-start_time)/3600, ' hours')
    

def validate(model, loader, crit, DEVICE):
    model.eval()
    val_loss = 0
    ex_epoch, ey_epoch, ez_epoch, et_epoch = 0., 0., 0., 0.
    eyaw_epoch, eroll_epoch, epitch_epoch, er_epoch = 0., 0., 0., 0.
    dR_epoch = 0.0
    
    reg_loss, rot_loss, pcd_loss = crit
    # reg_loss, rot_loss = crit

    process = tqdm(loader, unit='batch')

    for _, batch_data in enumerate(process):
        # print(i, batch_data)

        T_gt = [sample["T_gt"].to(DEVICE) for sample in batch_data]
        rgb_img = [sample["img"].to(DEVICE) for sample in batch_data]
        depth_img = [sample["depth_img_error"].to(DEVICE) for sample in batch_data]
        delta_q_gt = [sample["delta_q_gt"].to(DEVICE) for sample in batch_data]
        delta_t_gt = [sample["delta_t_gt"].to(DEVICE) for sample in batch_data]
        pcd_mis = [sample["pcd_mis"].to(DEVICE) for sample in batch_data]
        pcd_gt = [sample["pcd_gt"].to(DEVICE) for sample in batch_data]

        T_gt = torch.stack(T_gt, dim=0)
        rgb_img = torch.stack(rgb_img, dim=0) # correct shape
        depth_img = torch.stack(depth_img, dim=0) # correct shape
        delta_q_gt = torch.stack(delta_q_gt, dim=0)
        delta_t_gt = torch.stack(delta_t_gt, dim=0)
        targets = torch.cat((delta_q_gt, delta_t_gt), 1) # correct shape
        # print('targets shape: ', targets.shape)

        T_mis_batch = torch.tensor([]).to(DEVICE)

        for i in range(targets.shape[0]):
            delta_R_gt  = utils.qua2rot_torch(delta_q_gt[i])
            delta_tr_gt = torch.reshape(delta_t_gt[i],(3,1))
            delta_T_gt  = torch.hstack((delta_R_gt, delta_tr_gt)) 
            delta_T_gt  = torch.vstack((delta_T_gt, torch.Tensor([0., 0., 0., 1.]).to(DEVICE)))

            T_mis = torch.unsqueeze(torch.matmul(delta_T_gt, T_gt[i]), 0)
            T_mis_batch = torch.cat((T_mis_batch, T_mis), 0)
        # print(rgb_img.shape, i)

        pcd_pred, batch_T_pred, delta_q_pred, delta_t_pred = model(rgb_img, depth_img,  pcd_mis, T_mis_batch)
        
        translational_loss = reg_loss(delta_q_gt, delta_t_gt, delta_q_pred, delta_t_pred)
        rotational_loss = rot_loss(delta_q_gt, delta_q_pred)
        pointcloud_loss = pcd_loss(pcd_gt, pcd_pred)
        loss = translational_loss + rotational_loss + pointcloud_loss
        # loss = crit(output, targets)

        val_loss += loss.item()
        # experiment.log_metric('val batch loss', loss.item()/targets.shape[0])

        # print(f'L1 = {translational_loss}| L2 = {rotational_loss} | L3 = {pointcloud_loss}')
        e_x, e_y, e_z, e_t, e_yaw, e_pitch, e_roll, e_r, dR = criteria.test_metrics(batch_T_pred, T_gt)

        ex_epoch += e_x.item()
        ey_epoch += e_y.item()
        ez_epoch += e_z.item()
        et_epoch += e_t.item()
        eyaw_epoch += e_yaw.item()
        eroll_epoch += e_roll.item()
        epitch_epoch += e_pitch.item()
        er_epoch += e_r.item()
        dR_epoch += dR.item()

        # process.set_description('Validation: ')
        # process.set_postfix(loss=loss.item())
    
    ex_epoch /= len(loader)
    ey_epoch /= len(loader)
    ez_epoch /= len(loader)
    et_epoch /= len(loader)
    eyaw_epoch /= len(loader)
    eroll_epoch /= len(loader)
    epitch_epoch /= len(loader)
    er_epoch /= len(loader)
    dR_epoch /= len(loader)

    # val_loss /= len(loader)
    
    print(f'Ex = {ex_epoch}| Ey = {ey_epoch} | Ez = {ez_epoch} | Et = {et_epoch}') 
    print(f'yaw = {eyaw_epoch} | pitch = {epitch_epoch} | roll = {eroll_epoch} | er = {er_epoch} | Dg = {dR_epoch}')

    # experiment.log_metrics({'Ex': ex_epoch, 'Ey': ey_epoch, 'Ez': ez_epoch, 'Et': et_epoch})
    # experiment.log_metrics({'yaw': eyaw_epoch, 'pitch': epitch_epoch, 'roll': eroll_epoch, 'Er': er_epoch, 'Dg': dR_epoch})

    return val_loss, et_epoch, er_epoch