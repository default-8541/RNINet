import logging
import math
import sys
sys.path.append("..")
from utils.util_calculate_psnr_ssim import *
import numpy as np
import torch
import torch.nn.functional as F

def set_parameters_require_grad(model, is_fixed):
    if is_fixed:
        for parameter in model.parameters():
            parameter.requires_grad = False


def warmup_linear(x, warmup=0.1):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def save(model, model_save_path):
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_save_path)
    #torch.save(model.state_dict(), model_save_path)


def get_require_updated_params(model, is_fixed):
    if is_fixed:
        require_update_params = []
        for param in model.parameters():
            if param.requires_grad:
                require_update_params.append(param)
        return require_update_params
    else:
        return model.parameters()


def logger_info(logger_name, log_path='default_logger.log'):
    ''' set up logger
    modified by Kai Zhang (github: https://github.com/cszn)
    '''
    log = logging.getLogger(logger_name)
    if log.hasHandlers():
        print('LogHandlers exist!')
    else:
        print('LogHandlers setup!')
        level = logging.INFO
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d : %(message)s', datefmt='%y-%m-%d %H:%M:%S')
        fh = logging.FileHandler(log_path, mode='a')
        fh.setFormatter(formatter)
        log.setLevel(level)
        log.addHandler(fh)
        # print(len(log.handlers))

        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        log.addHandler(sh)

def log_loss_info(loss,loss_name,logger_name,epoch,iter_times):
    logger = logging.getLogger(logger_name)

    train_info = "avg train loss:{} for epoch/iter {}/{} is {}".format(loss_name, epoch, iter_times,
                                                                    loss)
    logger.info(train_info)
    
    
def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    image=image.transpose(1, 2, 0)
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))



def split_infer(img_gt, model, window_size=8, use_tile=True):
    if not use_tile:
        output = model(img_gt)
    else:
        b, c, h, w = img_gt.size()
        tile = math.ceil(h / (window_size * 2)) * window_size
        tile = min(tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window size"
        tile_overlap = 2 * tile - h
        sf = 1

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
        E = torch.zeros(b, c, h * sf, w * sf).type_as(img_gt)
        W = torch.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_gt[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                out_patch=model(in_patch)
                #out_patch = in_patch
                out_patch_mask = torch.ones_like(out_patch)
                E[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf] = E[...,h_idx * sf:(h_idx + tile) * sf,w_idx * sf:(w_idx + tile) * sf].add(out_patch)
                
                W[..., h_idx * sf:(h_idx + tile) * sf, w_idx * sf:(w_idx + tile) * sf] = W[...,h_idx * sf:(h_idx + tile) * sf,w_idx * sf:(w_idx + tile) * sf].add(out_patch_mask)
                
        output = E.div_(W)

    return output




def test_backbone(model,device_name, testset,iter_time,test_loader,logger_name):
    model.training = False
    model.eval()
    
    device = torch.device(device_name)
    
    avg_psnr, avg_ssim = 0,0
    sample_number = 0
    logger = logging.getLogger(logger_name)
    for i, test_data in enumerate(test_loader):
        img_lq, x_gt,img_name = test_data["L"].to(device), test_data["H"].to(device),test_data["H_path"][0]
        #x_gt = x_gt.flatten(0, 1)
        sample_number += 1
        # img_name = test_data["img_name"]
        #print(img_lq.shape,x_gt.shape)
        
#         #for DRUNet
#         img_lq = torch.cat((img_lq, torch.FloatTensor([15/255.]).repeat(1, 1, img_lq.shape[2], img_lq.shape[3]).to(device)), dim=1)
#         img_lq = img_lq.to(device)
        
        # Pad the input if not_multiple_of 8
        img_multiple_of=8
        height,width = img_lq.shape[2], img_lq.shape[3]
        H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
        padh = H-height if height%img_multiple_of!=0 else 0
        padw = W-width if width%img_multiple_of!=0 else 0
        img_lq = F.pad(img_lq, (0,padw,0,padh), 'reflect')

        #Pad the input if not_multiple_of 16 for CODE
#         img_multiple_of=16
#         height,width = img_lq.shape[2], img_lq.shape[3]
#         H,W = ((height+img_multiple_of)//img_multiple_of)*img_multiple_of, ((width+img_multiple_of)//img_multiple_of)*img_multiple_of
#         padh = H-height if height%img_multiple_of!=0 else 0
#         padw = W-width if width%img_multiple_of!=0 else 0
#         img_lq = F.pad(img_lq, (0,padw,0,padh), 'reflect')

        with torch.no_grad():
            output = model(img_lq)

        output = output[:, :, :height, :width]

        output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        output = (output * 255.0).round().astype(np.uint8)
        x_gt = x_gt.data.squeeze().float().cpu().clamp_(0, 1).numpy()
        x_gt = (x_gt * 255.0).round().astype(np.uint8)
        #print(output.shape,x_gt.shape)

        psnr = calculate_psnr(output, x_gt, crop_border=0)
        ssim = calculate_ssim(output, x_gt, crop_border=0, input_order = 'CHW')

        avg_psnr += psnr
        avg_ssim += ssim
        #avg_psnrb += psnrb
    avg_psnr = round(avg_psnr / sample_number,2)
    avg_ssim = round(avg_ssim / sample_number,4)
    #avg_psnrb = avg_psnrb / sample_number

    epoch_criterion_info="testset {}, iter_time {}, avg psnr/ssim is {}/{}".format(testset,iter_time,avg_psnr,avg_ssim)
    logger.info(epoch_criterion_info)
    
    
    
def train_backbone(model, train_loader, test_loader, args, loss_fn):
    epochs = args["epochs"]
    
    print("train model start, epoch is {}".format(epochs))

    model_path = args["model_path"]
    logger_name = args["train_logger"]
    lr = args["lr"]  # 1e-5
    is_fixed = True
    require_update_params = get_require_updated_params(model, is_fixed)
    optimizer = torch.optim.Adam(require_update_params, lr=lr)
    model.training = True
    model.train()

    device = torch.device(args["device"])
    iter_times = 0
    base_lr = lr

    lr_this_step=base_lr


    for epoch in range(epochs):
        train_loss_this_epoch = 0
        sample_this_epoch = 0

        # warm up
        if args["warm_up"]==True:
            pass
#             g_t = float(epoch + 1) / epochs
#             w_l = warmup_linear(g_t, warmup=0.1)

#             lr_this_step = base_lr * w_l
#             print("lr_this_step is {}".format(lr_this_step))
#             for param_group in optimizer.param_groups:
#                 param_group["lr"] = lr_this_step
            #g_t = float(epoch + 1) / epochs
            #w_l = warmup_linear(g_t, warmup=0.1)
#             if (epoch+1)/args["epochs"]>=w_l_s[w_l_c]:
#                 lr_this_step = lr_this_step * 0.5
#                 w_l_c+=1
#                 print("lr_this_step is {}".format(lr_this_step))
#                 for param_group in optimizer.param_groups:
#                     param_group["lr"] = lr_this_step

        else:
            print("no warm up lr_this_step is {}".format(lr_this_step))


        for i, train_data in enumerate(train_loader):
            optimizer.zero_grad()
            iter_times += 1
            img_lq, x_gt = train_data["L"].to(device), train_data["H"].to(device)
            #x_gt = x_gt.flatten(0, 1)
            batch_size = img_lq.shape[0]

            output = model(img_lq)
            loss = loss_fn(output, x_gt)
 

            loss_this_batch = loss.data.squeeze().float().cpu().numpy()
            train_loss_this_epoch += loss_this_batch * batch_size
            sample_this_epoch += batch_size


            loss.backward()
            optimizer.step()

            if iter_times % args["io_iter"] == 0:
                log_loss_info(train_loss_this_epoch/ sample_this_epoch,"total_loss",logger_name,
                              epoch,iter_times)
            if iter_times%(args["test_iter"])==0 and epoch>=args["test"]:
                log_loss_info(train_loss_this_epoch/ sample_this_epoch,"total_loss",logger_name,
                              epoch,iter_times)
                model_save_path = model_path + str(iter_times) + "_iter_times.pth"
                save(model, model_save_path)
                print("save {}_iter_times.pth model to model backbone".format(iter_times))
                #test_backbone(model,args["device"], args["testset"],iter_times,test_loader,args["test_logger"])
                model.training = True
                model.train()
                print(args["label"])
            if iter_times>args["stop"]:
                return

        



