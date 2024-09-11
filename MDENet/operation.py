import cv2
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import metrics,metrics_v2
import glob
from path import *
from utils import save_pre_result
import torch.nn.functional as F
from imageio import imwrite

filename = glob.glob(test_src_t1 + '/*.png')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1]

def Down_16(x):
    return F.interpolate(x,scale_factor=0.0625, mode="bilinear")
def Down_8(x):
    return F.interpolate(x,scale_factor=0.125, mode="bilinear")
def Down_4(x):
    return F.interpolate(x,scale_factor=0.25, mode="bilinear")
def Down_2(x):
    return F.interpolate(x,scale_factor=0.5, mode="bilinear")

def imgshow(img, showpath, index):
    img = img[0,:,:,:]
    img_final = img.detach().cpu().numpy()
    img_final = np.round(((img_final - img_final.min())/ (img_final.max() - img_final.min()))*255)
    img = img_final.transpose(1,2,0)
    img = img.astype('uint8')
    # if img.shape[2] == 1:
    #     img = img.reshape([img.shape[0], img.shape[1]])
    # else :
    #     img = img[:,:,0]
    indexd = format(index, '05d')
    file_name = str(indexd) + '.png'
    path_out = showpath + file_name          
    imwrite(path_out, img)
    # return img

def feature_show(feature, showpath, index):
    feature_map = feature[0]
    # num_features = feature_map.size(0)
    # # fig, axarr = plt.subplots(4, num_features // 4, figsize=(15, 15))
    # # for idx in range(num_features):
    # #     row = idx // 4
    # #     col = idx % 4
    # #     axarr[row, col].imshow(feature_map[idx, :, :].detach().cpu().numpy())
    # #     axarr[row, col].axis('off')
    # # plt.show()

    # fig, axarr = plt.subplots(1, num_features, figsize=(15, 15))
    # for idx in range(num_features):
    #     axarr[idx].imshow(feature_map[idx, :, :].detach().cpu().numpy())
    #     axarr[idx].axis('off')
    # plt.show()
    cam = feature_map[0]
    # _, cam = torch.max(feature_map, 0)
    cam = cam.detach().cpu().numpy()
    # cam = cv2.resize(predicted, (224, 224))
    cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # 归一化
    cam = 1-cam
    cam = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam_on_image = heatmap# + np.transpose(np.float32(image), (1, 2, 0))
    cam_on_image = cam_on_image / np.max(cam_on_image)

    indexd = format(index, '05d')
    file_name = str(indexd) + '.png'
    path_out = showpath + file_name          
    imwrite(path_out, np.uint8(255 * cam_on_image))
    # plt.imshow()
    # plt.axis('off')
    # plt.show()

def train(net, dataloader_train, total_step, criterion_ce, optimizer):
    print('Training...')
    model = net.train()
    num = 0
    epoch_loss = 0
    cm_total = np.zeros((2, 2))

    for x1, x2, y in dataloader_train:
        inputs_t1 = x1.to(device)
        inputs_t2 = x2.to(device)
        labels = y.to(device)

        optimizer.zero_grad()
        pre = model(inputs_t1, inputs_t2)
        loss = criterion_ce(pre, labels)   # out_ch=1
        #loss3 = criterion_ce(out3, Down_2(labels)) 
        #loss4 = criterion_ce(out4, Down_4(labels)) 
        #loss5 = criterion_ce(out5, Down_8(labels)) 
        loss = loss#+loss3+loss4+loss5
        # loss = criterion_ce(pre, torch.squeeze(labels.long(), dim=1))   # out_ch=2
        loss.backward()
        epoch_loss += loss.item()
        optimizer.step()

        # pre = torch.max(pre, 1)[1]  # out_ch=2
        cm = metrics.ConfusionMatrix(2, pre, labels)
        cm_total += cm
        precision, recall, f1, iou, kc = metrics.get_score(cm)

        num += 1

        print('%d/%d, loss:%f, Pre:%f, Rec:%f, F1:%f, IoU:%f, KC:%f' % (num, total_step, loss.item(), precision[1], recall[1], f1[1], iou[1], kc))
    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)

    return epoch_loss, precision_total['precision_1'], recall_total['recall_1'], f1_total['f1_1'], iou_total['iou_1'], kc_total


def validate(net, dataloader_val, epoch):
    print('Validating...')
    model = net.eval()
    num = 0
    cm_total = np.zeros((2, 2))

    for x1, x2, y in tqdm(dataloader_val):
        inputs_t1 = x1.to(device)
        inputs_t2 = x2.to(device)
        labels = y.to(device)
        pre = model(inputs_t1, inputs_t2)
        # pre = torch.max(pre, 1)[1]  # out_ch=2
        cm = metrics.ConfusionMatrix(2, pre, labels)
        cm_total += cm
        num += 1
    precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    return precision_total['precision_1'], recall_total['recall_1'], f1_total['f1_1'], iou_total['iou_1'], kc_total


def predict(net, dataloader_test):
    print('Testing...')
    model = net.eval()
    num = 0
    cm_total = np.zeros((2, 2))
    for x1, x2, y in tqdm(dataloader_test):
        inputs_t1 = x1.to(device)
        inputs_t2 = x2.to(device)
        labels = y.to(device)
        pre = model(inputs_t1, inputs_t2)
        cm = metrics.ConfusionMatrix(2, pre, labels)
        cm_total += cm
        save_pre_result(pre, 'test', num, save_path=test_predict)
        num += 1
    # precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    # return precision_total, recall_total, f1_total, iou_total, kc_total
    precision_total, recall_total, f1_total, iou_total, kc_total,oa = metrics_v2.get_score_sum(cm_total)
    return precision_total, recall_total, f1_total, iou_total, kc_total,oa

def predict2(net, dataloader_test):  # compute runing time
    print('Testing...')
    model = net.eval()
    num = 0
    cm_total = np.zeros((2, 2))
    for x1, x2, y in tqdm(dataloader_test):
        inputs_t1 = x1.to(device)
        inputs_t2 = x2.to(device)
        labels = y.to(device)
        pre = model(inputs_t1, inputs_t2)
        # cm = metrics.ConfusionMatrix(2, pre, labels)
        # cm_total += cm
        # save_pre_result(pre, 'test', num, save_path=test_predict)
        num += 1
    # precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    # return precision_total, recall_total, f1_total, iou_total, kc_total
    # precision_total, recall_total, f1_total, iou_total, kc_total,oa = metrics_v2.get_score_sum(cm_total)
    # return precision_total, recall_total, f1_total, iou_total, kc_total,oa

def predict3(net, dataloader_test): # show feature
    print('Testing...')
    model = net.eval()
    num = 0
    cm_total = np.zeros((2, 2))
    for x1, x2, y in tqdm(dataloader_test):
        inputs_t1 = x1.to(device)
        inputs_t2 = x2.to(device)
        labels = y.to(device)
        pre,x1,x2,x3,x4 = model(inputs_t1, inputs_t2)
        cm = metrics.ConfusionMatrix(2, pre, labels)
        cm_total += cm
        # save_pre_result(pre, 'test', num, save_path='/home/tanlishan/overall/CDss-Net-main/features_show/LEVIR/our5_c_3f/pre')
        # save_pre_result(labels, 'gt', num, save_path='/home/tanlishan/overall/CDss-Net-main/features_show/LEVIR/our5_c_3f/gt')
        feature_show(x1,'/home/tanlishan/overall/CDss-Net-main/features_show/LEVIR/our5_c_v2/fc1_1/',num)
        feature_show(x2,'/home/tanlishan/overall/CDss-Net-main/features_show/LEVIR/our5_c_v2/fc1_2/',num)
        # feature_show(x3,'/home/tanlishan/overall/CDss-Net-main/features_show/LEVIR/our5_c_v2/x3/',num)
        # feature_show(x4,'/home/tanlishan/overall/CDss-Net-main/features_show/LEVIR/our5_c_v2/x4/',num)
        # imgshow(inputs_t1,'/home/tanlishan/overall/CDss-Net-main/features_show/LEVIR/our5_c_3f/t1/',num)
        # imgshow(inputs_t2,'/home/tanlishan/overall/CDss-Net-main/features_show/LEVIR/our5_c_3f/t2/',num)
        num += 1
    # precision_total, recall_total, f1_total, iou_total, kc_total = metrics.get_score_sum(cm_total)
    # return precision_total, recall_total, f1_total, iou_total, kc_total
    # precision_total, recall_total, f1_total, iou_total, kc_total,oa = metrics_v2.get_score_sum(cm_total)
    # return precision_total, recall_total, f1_total, iou_total, kc_total,oa