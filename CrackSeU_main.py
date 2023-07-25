import os
import argparse
import logging
import torch
from torch.utils.data import DataLoader
from torch import autograd, optim
import torch.nn.functional as F
from datetime import datetime
import time
import copy
import cv2

from ptflops import get_model_complexity_info
from Dataset import ConcreteCrackDataset
from utils.loss import dice_loss_function
from utils.metrics import get_IoU,get_Dice,Get_F1_score

def getArgs():
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train/test/", default="train") # Train or Test
    parse.add_argument('--arch', type=str, default='CrackSeU_B_LN_VT', 
                       choices=['CrackSeU_B_LN_VT', 'CrackSeU_B_BN', 'CrackSeU_B_LN_Pytorch', 'CrackSeU_B_LN_He'], help='Model name')
    parse.add_argument('--save_path', type=str, default='./Checkpoints/', help='Model save path')
    parse.add_argument('--dataset', default='Concrete_crack', help='Dataset Name') 
    parse.add_argument("--batch_size", type=int, default=2)
    parse.add_argument("--lr",type=float,default=1e-4)
    parse.add_argument("--log_dir", default='Result_log/log', help="log dir")
    parse.add_argument("--epoch", type=int, default=50)
    parse.add_argument("--test_epoch", type=int, default=0)
    args = parse.parse_args()
    return args


def getLog(args):
    dirname = os.path.join(args.log_dir,args.arch,str(args.batch_size),str(args.dataset),str(args.epoch))
    filename = dirname +'/log.log'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    logging.basicConfig(
            filename=filename,
            level=logging.INFO,
            format='%(asctime)s:%(levelname)s:%(message)s'
        )
    return logging

# ------------- Define the model -------------- #
def getModel(args):
    if 'CrackSeU_B_LN_VT' in args.arch:
        from Nets.CrackSeU_B_LN_VT import CrackSeU
    elif 'CrackSeU_B_BN' in args.arch:
        from Nets.CrackSeU_B_BN import CrackSeU
    elif 'CrackSeU_B_LN_Pytorch' in args.arch:
        from Nets.CrackSeU_B_LN_Pytorch import CrackSeU
    elif 'CrackSeU_B_LN_He' in args.arch:
        from Nets.CrackSeU_B_LN_He import CrackSeU
    
    model = CrackSeU(3, 1).to(device)
    return model


def getModel1(args):
    if 'CrackSeU_B_LN_VT' in args.arch:
        from Nets.CrackSeU_B_LN_VT import CrackSeU_Inference
    elif 'CrackSeU_B_BN' in args.arch:
        from Nets.CrackSeU_B_BN import CrackSeU_Inference
    elif 'CrackSeU_B_LN_Pytorch' in args.arch:    
        from Nets.CrackSeU_B_LN_Pytorch import CrackSeU_Inference
    elif 'CrackSeU_B_LN_He' in args.arch:
        from Nets.CrackSeU_B_LN_He import CrackSeU_Inference
    
    model = CrackSeU_Inference(3, 1)
    return model


def getDataset(args):
    
    # ------------- Choose the dataset. You can use your own dataset. -------------- #
    if args.dataset =='Concrete_crack':  
        Dataset = ConcreteCrackDataset
    
    train_dataset = Dataset(r"train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size)
    test_dataset = Dataset(r"test")
    test_dataloader = DataLoader(test_dataset, batch_size=1)
        
    return train_dataloader, test_dataloader


def val(args, model, best_epoch, epoch, best_dice, val_dataloader):
    
    with torch.no_grad(): 
        model= model.eval()
        
        i=0 
        IoU_total = 0
        Dice_total = 0
        num = len(val_dataloader) 
        print(num)
        
        for image, gt, pic_path, mask_path in val_dataloader:
            
            image = image.to(device)
            predict, _ = model(image)
            predict = torch.squeeze(predict).cpu().numpy()
             
            # ------------- Calculate Dice -------------- #
            gt = gt.squeeze().numpy()
            Dice = get_Dice(gt, predict)
            Dice_total += Dice
            
            if i < num:i+=1
            
        aver_iou = IoU_total / num
        aver_dice = Dice_total / num
        
        print('aver_iou = %f, aver_dice = %f' % (aver_iou, aver_dice))
        logging.info('aver_iou = %f, aver_dice = %f' % (aver_iou, aver_dice))
        if aver_dice > best_dice:
            
            print(f"aver_iou={aver_dice:f} > best_dice={best_dice:f}\n")
            logging.info(f"aver_iou={aver_dice:f} > best_dice={best_dice:f}")
            
            best_dice = aver_dice
            best_epoch = epoch
            
            print('===========>save best model!')
            logging.info('===========>save best model!')
            torch.save(model.state_dict(), save_path + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(epoch) + '.pth')
        
        print(f"best_epoch = {best_epoch}, Best dice = {best_dice:f}\n")
        logging.info(f"best_epoch = {best_epoch}, Best dice = {best_dice:f}\n")
        
        return best_epoch, best_dice, aver_dice

def train(model, train_dataloader, test_dataloader, args):
    
    # ------------- Define loss functions and optimizer -------------- #
    criterion = torch.nn.BCELoss()
    criterion2 = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ------------- Define training hyper-parameters -------------- #
    num_epochs = args.epoch
    best_epoch, best_dice, aver_dice = 0., 0., 0.
    loss_list = []
    dice_list = []
    loss_end, dice_loss_end, loss_2_BCE, loss_2_Dice= 0., 0., 0., 0.
    
    # ------------- Training -------------- #
    for epoch in range(num_epochs):
        model = model.train()
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logging.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(train_dataloader.dataset)
        epoch_loss = 0.
        step = 0
 
        for inputs, labels, _, _ in train_dataloader:

            step += 1
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels_2 = F.interpolate(labels, scale_factor = 0.5, mode='bicubic', align_corners=True).to(device)
            
            optimizer.zero_grad() # zero the parameter gradients
            out, out_2 = model(inputs)
            loss_end = criterion(out, labels)
            dice_loss_end = dice_loss_function(out, labels)
            loss_2_BCE = criterion2(out_2, labels_2)
            #loss_2_Dice = dice_loss_function(out_2, labels)
            
            loss = loss_end + dice_loss_end + 0.1 * loss_2_BCE 

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            if step % 100 == 0:
                print("%s STEP [%d/%d], train_loss: %f, loss_end: %f, dice_loss_end: %f, loss_2_BCE: %f, loss_2_Dice: %f," % (datetime.now(), step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item(), loss_end, dice_loss_end, loss_2_BCE, loss_2_Dice))
                logging.info("%s STEP [%d/%d], train_loss: %f, loss_end: %f, dice_loss_end: %f, loss_2_BCE: %f, loss_2_Dice: %f," % (datetime.now(), step, (dt_size - 1) // train_dataloader.batch_size + 1, loss.item(), loss_end, dice_loss_end, loss_2_BCE, loss_2_Dice))
        
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        logging.info("epoch %d loss:%0.3f" % (epoch, epoch_loss))
        loss_list.append(epoch_loss)
        
        # Validation
        if epoch >= num_epochs * 0.:
            best_epoch, best_dice,aver_dice = val(args, model, best_epoch, epoch, best_dice, test_dataloader)
            dice_list.append(aver_dice)

    print('------------- Congratulations! -------------')

    return model


def test(test_dataloader,save_predict=False):
    
    logging.info('final test........')
    if save_predict ==True:                                     
        dir = os.path.join(r'./Predictions',str(args.arch),str(args.batch_size),str(args.epoch),str(args.dataset)) 
        if not os.path.exists(dir):
            os.makedirs(dir)
        else:
            print('Dir already exist!')

    torch.cuda.synchronize()
    start_time=time.perf_counter()
    print('Start time is %s Seconds' % start_time)
    
    #------------------ Load the model ------------------#
    model.load_state_dict(torch.load(save_path + str(args.arch) + '_' + str(args.batch_size) + '_' + str(args.dataset) + '_' + str(args.test_epoch) + '.pth', map_location='cpu'))  # 载入训练好的模型
    model.eval()

    # ------------- Inference -------------- #
    with torch.no_grad():
        i=0   # i-th image
        IoU_total = 0.
        Dice_total = 0.
        F1_score_total = 0.
        num = len(test_dataloader)  # total number
        print("Total number = ", num)
        
        for image, gt, pic_path, mask_path in test_dataloader:
            
            image = image.to(device)
            predict, _ = model(image)
            predict = torch.squeeze(predict).cpu().numpy() 
            
            # ------------- Save the prediction -------------- #
            cv2.imwrite(os.path.join(dir, mask_path[0].split('\\')[-1]),predict*255.0)
            
            # ------------- Calculate Dice -------------- #
            gt = gt.squeeze().numpy() # gt: numpy
            Dice = get_Dice(gt, predict)
            Dice_total += Dice
                   
            # ------------- Calculate IoU -------------- #
            gt_IoU = copy.deepcopy(gt)
            predict_IoU = copy.deepcopy(predict)
            IoU = get_IoU(gt_IoU, predict_IoU)
            IoU_total += IoU  
            
            # ------------- F1-Score -------------- #
            F1_score = Get_F1_score(gt, predict)
            F1_score_total += F1_score

            if i < num:
                i += 1 # Next image

        torch.cuda.synchronize()
        end_time=time.perf_counter()
        print('End time is %s Seconds' % end_time)
        Run_time = end_time - start_time
        FPS = num / Run_time
        print('Evaluation speed: %s FPS' % FPS)

        # ------------- Print the IoU and Dice -------------- #
        print('mi IoU=%f' % (IoU_total / num))
        print('mi Dice=%f' % (Dice_total / num))
        print('F1 score=%f' % (F1_score_total / num))

        # ------------- Calculate the computational complexity -------------- #
        model1 = getModel1(args)
        
        # ------------- https://github.com/sovrasov/flops-counter.pytorch ------------- #
        macs, params = get_model_complexity_info(model1, (3, 512, 512), as_strings = False, print_per_layer_stat = False, verbose = False)
        print("# ------------- ptflops -------------- #")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        print(f"macs = {macs/1e9}G")
        print(f"params = {params/1e6}M")


if __name__ =="__main__":
    
    # ------------- Preliminary -------------- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    args = getArgs()
    logging = getLog(args)
    print('*' * 30)
    print('models:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    logging.info('\n=======\nmodels:%s,\nepoch:%s,\nbatch size:%s\ndataset:%s\n========' % \
          (args.arch, args.epoch, args.batch_size,args.dataset))
    print('*' * 30)

    save_path = args.save_path + args.arch + '/'
    if not os.path.exists(save_path):
            os.makedirs(save_path)
            
    # ------------- Get the model -------------- #
    model = getModel(args)
    train_dataloader, test_dataloader = getDataset(args)
    if 'train' in args.action:
        train(model, train_dataloader, test_dataloader, args)
    if 'test' in args.action:
        test(test_dataloader, save_predict=True)
