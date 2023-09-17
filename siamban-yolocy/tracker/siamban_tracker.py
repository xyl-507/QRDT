from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch.nn.functional as F
import torch
from PIL import Image
from siamban.core.config import cfg
from siamban.tracker.base_tracker import SiameseTracker
from siamban.utils.bbox import corner2center
from siamban.tracker.classifier.base_classifier import BaseClassifier
from toolkit.utils.statistics import iou
class SiamBANTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamBANTracker, self).__init__()
        self.score_size = (cfg.TRACK.INSTANCE_SIZE - cfg.TRACK.EXEMPLAR_SIZE) // \
            cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE                                  #255-127//8+1+8
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self.lost_count = 0

        self.model = model
        self.model.eval()

        if cfg.TRACK.USE_CLASSIFIER:
            self.classifier = BaseClassifier(self.model)

    def generate_points(self, stride, size):
        ori = - (size // 2) * stride
        x, y = np.meshgrid([ori + stride * dx for dx in np.arange(0, size)],
                           [ori + stride * dy for dy in np.arange(0, size)])
        points = np.zeros((size * size, 2), dtype=np.float32)
        points[:, 0], points[:, 1] = x.astype(np.float32).flatten(), y.astype(np.float32).flatten()

        return points

    def _convert_bbox(self, delta, point):
        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1)
        delta = delta.detach().cpu().numpy()

        delta[0, :] = point[:, 0] - delta[0, :]
        delta[1, :] = point[:, 1] - delta[1, :]
        delta[2, :] = point[:, 0] + delta[2, :]
        delta[3, :] = point[:, 1] + delta[3, :]
        delta[0, :], delta[1, :], delta[2, :], delta[3, :] = corner2center(delta)
        return delta

    def _convert_score(self, score):
        if self.cls_out_channels == 1:
            score = score.permute(1, 2, 3, 0).contiguous().view(-1)
            score = score.sigmoid().detach().cpu().numpy()
        else:
            score = score.permute(1, 2, 3, 0).contiguous().view(self.cls_out_channels, -1).permute(1, 0)
            score = score.softmax(1).detach()[:, 1].cpu().numpy()
        return score

    def _bbox_clip(self, cx, cy, width, height, boundary):
        cx = max(0, min(cx, boundary[1]))
        cy = max(0, min(cy, boundary[0]))
        width = max(10, min(width, boundary[1]))
        height = max(10, min(height, boundary[0]))
        return cx, cy, width, height

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox
        """
        self.frame_num = 1
        self.temp_max = 0


        self.center_pos = np.array([bbox[0]+(bbox[2]-1)/2,
                                    bbox[1]+(bbox[3]-1)/2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        self.z0_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.z_crop = self.z0_crop
        with torch.no_grad():
            self.model.template(self.z0_crop)

        if cfg.TRACK.USE_CLASSIFIER:                                                #模板更新代码
            if cfg.TRACK.TEMPLATE_UPDATE:
                with torch.no_grad():
                    self.model.template_short_term(self.z_crop)

            s_xx = s_z * (cfg.TRACK.INSTANCE_SIZE * 2 / cfg.TRACK.EXEMPLAR_SIZE)        #s_z*(255*2/127)  x是搜索分支帧，z是模板分支帧
            x_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.INSTANCE_SIZE * 2,  #根据目标的中心点，以255*2为通道数量得到裁剪后的x
                round(s_xx), self.channel_average)

            self.classifier.initialize(x_crop.type(torch.FloatTensor), bbox)

    def track(self, img):

        self.frame_num += 1
        self.curr_frame = img
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)


        with torch.no_grad():
            outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))


        def normalize(score):                                                       #规范化得分值socre=（所有的分-最小得分）/（最大得分-最小得分）
            score = (score - np.min(score)) / (np.max(score) - np.min(score))
            return score


        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0]*scale_z, self.size[1]*scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0]/self.size[1]) /
                     (pred_bbox[2, :]/pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score

        if cfg.TRACK.USE_CLASSIFIER:  # 模板更新代码
            flag, s = self.classifier.track()
            if flag == 'not_found':
                self.lost_count += 1
            else:
                self.lost_count = 0

            # confidence = Image.fromarray(np.uint8(s.detach().cpu().numpy()) ) # 创建图像内存

            confidence = Image.fromarray(s.detach().cpu().numpy())
            confidence = np.array(
                confidence.resize((self.score_size, self.score_size))).flatten() # 将得到的二维图像存储拉伸为一个一维数组

            pscore = pscore.reshape( 1,-1) * (1 - cfg.TRACK.COEE_CLASS) + \
                     normalize(confidence) * cfg.TRACK.COEE_CLASS  # 预测分数=pscore reshape为5维*（1-0.8）+/规范化后的条件分数*0.8
            pscore = pscore.flatten()  # 得到预测得分并把预测得分展为一维数组

            if cfg.TRACK.TEMPLATE_UPDATE:  # 若是模板更新指标为正的话
                score_st = self._convert_score(outputs['cls_st'])  # 首次预测得分赋值为通过softmax函数优化后的得分
                pred_bbox_st = self._convert_bbox(outputs['loc_st'], self.points)  # 根据loc以及anchor的除得出首次预测框
                s_c_st = change(sz(pred_bbox_st[2, :], pred_bbox_st[3, :]) /  # 而后根据首次预测框
                                (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
                r_c_st = change((self.size[0] / self.size[1]) /
                                (pred_bbox_st[2, :] / pred_bbox_st[3, :]))  # 得出
                penalty_st = np.exp(-(r_c_st * s_c_st - 1) * cfg.TRACK.PENALTY_K)
                pscore_st = penalty_st * score_st
                pscore_st = pscore_st.reshape(1, -1) * (1 - cfg.TRACK.COEE_CLASS) + \
                            normalize(confidence) * cfg.TRACK.COEE_CLASS
                pscore_st = pscore_st.flatten()


        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
            self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR

        # cx = bbox[0] + self.center_pos[0]
        # cy = bbox[1] + self.center_pos[1]
        if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.SHORT_TERM_DRIFT and self.lost_count >= 8:        #若
            cx, cy = bbox[0] / 4 + self.center_pos[0], bbox[1] / 4 + self.center_pos[1]             #cx，cy赋值为bbox前两个维度的值/4加上对用维度位置的中心点值
        else:
            cx, cy = bbox[0] + self.center_pos[0], bbox[1] + self.center_pos[1]                     #否则cx，cy赋值为未除4之后的值
        # smooth bbox
        width = self.size[0] * (1 - lr) + bbox[2] * lr
        height = self.size[1] * (1 - lr) + bbox[3] * lr

        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])


        if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.TEMPLATE_UPDATE:                                  #若是使用分类器及模板更新指标都为true
            pscore_st = pscore_st * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                     self.window * cfg.TRACK.WINDOW_INFLUENCE                                        #则将首次预测得分置为首次预测得分*（1-0.44）+\window*0.44
            best_idx_st = np.argmax(pscore_st)                                                       #best——idx——st置为首次预测得分中最大得分所在位置的下标
            bbox_st = pred_bbox_st[:, best_idx_st] / scale_z                                        #框置为置为首次预测得分中最大得分所在位置的下标之前的首次预测框/模板正方形边长
            lr_st = penalty_st[best_idx_st] * score_st[best_idx_st] * cfg.TRACK.LR                  #learning rate
            if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.SHORT_TERM_DRIFT and self.lost_count >= 8:
                cx_st, cy_st = bbox_st[0] / 4 + self.center_pos[0], bbox_st[1] / 4 + self.center_pos[1]
            else:
                cx_st, cy_st = bbox_st[0] + self.center_pos[0], bbox_st[1] + self.center_pos[1]
            width_st = self.size[0] * (1 - lr_st) + bbox_st[2] * lr_st
            height_st = self.size[1] * (1 - lr_st) + bbox_st[3] * lr_st
            cx_st, cy_st, width_st, height_st = self._bbox_clip(cx_st, cy_st, width_st, height_st, img.shape[:2])   #与上方预测狂一样只是此次置的是首次预测
            if iou((cx_st, cy_st, width_st, height_st), (cx, cy, width, height), wh=True) >= cfg.TRACK.TAU_REGRESSION \
                and score_st[best_idx_st] - score[best_idx] >= cfg.TRACK.TAU_CLASSIFICATION:
                cx, cy, width, height, score, best_idx = cx_st, cy_st, width_st, height_st, score_st, best_idx_st  #判断其iou值，iou大于一定阈值并且score要比前面的模板大很多之时替换其模板
                                                                                                    #为论文中模板更新公式7的代码


        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        best_score = score[best_idx]


        if cfg.TRACK.USE_CLASSIFIER:
            self.classifier.update(bbox, scale_z, flag)

            if cfg.TRACK.TEMPLATE_UPDATE:                                                           #为论文中模板更新部分公式6的代码
                if torch.max(s).item() >= cfg.TRACK.TARGET_UPDATE_THRESHOLD and flag != 'hard_negative':
                    if torch.max(s).item() > self.temp_max:
                        self.temp_max = torch.max(s).item()
                        self.channel_average = np.mean(img, axis=(0, 1))
                        self.z_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE, s_z, self.channel_average)

                if (self.frame_num - 1) % cfg.TRACK.TARGET_UPDATE_SKIPPING == 0:
                    self.temp_max = 0
                    with torch.no_grad():
                        self.model.template_short_term(self.z_crop)


        # return {
        #         'bbox': bbox,
        #         'best_score': best_score
        #        }
        if cfg.TRACK.USE_CLASSIFIER:
            return {
                    'bbox': bbox,
                    'best_score': best_score,
                    'flag': flag
                   }
        else:
            return {
                    'bbox': bbox,
                    'best_score': best_score
                   }
