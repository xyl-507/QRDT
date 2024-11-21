from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch  # DROL, XYL 20220927
from PIL import Image  # DROL, XYL 20220927
from toolkit.utils.statistics import iou  # DROL, XYL 20220927

from siamban.core.config import cfg
from siamban.tracker.base_tracker import SiameseTracker
from siamban.utils.bbox import corner2center
from siamban.tracker.classifier.base_classifier import BaseClassifier
from siamban.tracker.classifier.libs.plotting import show_tensor_np
import cv2


class SiamBANTracker(SiameseTracker):
    def __init__(self, model):
        super(SiamBANTracker, self).__init__()
        self.longterm_state = False
        if self.longterm_state:
            self.instance_size = cfg.TRACK.LOST_INSTANCE_SIZE
        else:
            self.instance_size = cfg.TRACK.INSTANCE_SIZE
        self.score_size = (self.instance_size - cfg.TRACK.EXEMPLAR_SIZE) // \
                          cfg.POINT.STRIDE + 1 + cfg.TRACK.BASE_SIZE
        hanning = np.hanning(self.score_size)
        window = np.outer(hanning, hanning)
        self.cls_out_channels = cfg.BAN.KWARGS.cls_out_channels
        self.window = window.flatten()
        self.points = self.generate_points(cfg.POINT.STRIDE, self.score_size)
        self.model = model
        self.model.eval()
        self.lost_count = 0
        self.update_count = 0

# -------------------------------------KalmanFilter  from CFME
        # code: https://github.com/SY-Xuan/CFME
        # paper: Object Tracking in Satellite Videos by Improved Correlation Filters With Motion Estimations
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                    np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                   np.float32) * 0.001  # 0.001
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1
        self.predict = [0, 0]
        # whether kalman filter can be used
        self.iskalman_work = False
        self.last_bbox = []
        self.kalman_work_num = 0
# -------------------------------------KalmanFilter

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
        self.frame_num = 1  # DROL, XYL 20220927
        self.temp_max = 0  # DROL, XYL 20220927
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]])

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        # z_crop = self.get_subwindow(img, self.center_pos,
        #                             cfg.TRACK.EXEMPLAR_SIZE,
        #                             s_z, self.channel_average)
        # self.model.template(z_crop)
        # ----------------------------------------------------------------------------------------------------------------------
        self.z0_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE,  # DROL, XYL 20220927
                                          s_z, self.channel_average)
        self.z_crop = self.z0_crop  # DROL, XYL 20220927

        with torch.no_grad():  # DROL, XYL 20220927
            self.model.template(self.z0_crop)  # DROL, XYL 20220927

        if cfg.TRACK.TEMPLATE_UPDATE:
            with torch.no_grad():
                self.model.template_short_term(self.z_crop)

    # ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------KalmanFilter
        self.kalman.correct(np.array([[np.float32(bbox[0])], [np.float32(bbox[1])]]))
# -------------------------------------KalmanFilter
    def track(self, img):
    # def track(self, img,i):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        """
        self.frame_num += 1  # DROL, XYL 20220927
        self.curr_frame = img  # DROL, XYL 20220927
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (self.instance_size / cfg.TRACK.EXEMPLAR_SIZE)
        x_crop = self.get_subwindow(img, self.center_pos,
                                    self.instance_size,
                                    round(s_x), self.channel_average)
        with torch.no_grad():  # DROL, XYL 20220927
            outputs = self.model.track(x_crop)  # DROL, XYL 20220927
        # outputs = self.model.track(x_crop)

        score = self._convert_score(outputs['cls'])
        pred_bbox = self._convert_bbox(outputs['loc'], self.points)

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            return np.sqrt((w + pad) * (h + pad))

        def normalize(score):  # DROL, XYL 20220927
            score = (score - np.min(score)) / (np.max(score) - np.min(score))
            return score

        # scale penalty
        s_c = change(sz(pred_bbox[2, :], pred_bbox[3, :]) /
                     (sz(self.size[0] * scale_z, self.size[1] * scale_z)))

        # aspect ratio penalty
        r_c = change((self.size[0] / self.size[1]) /
                     (pred_bbox[2, :] / pred_bbox[3, :]))
        penalty = np.exp(-(r_c * s_c - 1) * cfg.TRACK.PENALTY_K)
        pscore = penalty * score
        # ----------------------------------------------------------------------------------------------------------------------
        if cfg.TRACK.TEMPLATE_UPDATE:
            score_st = self._convert_score(outputs['cls_st'])
            pred_bbox_st = self._convert_bbox(outputs['loc_st'], self.points)
            s_c_st = change(sz(pred_bbox_st[2, :], pred_bbox_st[3, :]) /
                            (sz(self.size[0] * scale_z, self.size[1] * scale_z)))
            r_c_st = change((self.size[0] / self.size[1]) /
                            (pred_bbox_st[2, :] / pred_bbox_st[3, :]))
            penalty_st = np.exp(-(r_c_st * s_c_st - 1) * cfg.TRACK.PENALTY_K)
            pscore_st = penalty_st * score_st
            # pscore_st = pscore_st.reshape(1, -1) * (1 - cfg.TRACK.COEE_CLASS) + \
            #                 normalize(confidence) * cfg.TRACK.COEE_CLASS
            # pscore_st = pscore_st.reshape(1, -1) * (1 - cfg.TRACK.COEE_CLASS) + \
            #             normalize(confidence) * cfg.TRACK.COEE_CLASS
            # pscore_st = pscore_st.flatten()
        # window penalty
        pscore = pscore * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                 self.window * cfg.TRACK.WINDOW_INFLUENCE
        best_idx = np.argmax(pscore)
        bbox = pred_bbox[:, best_idx] / scale_z
        lr = penalty[best_idx] * score[best_idx] * cfg.TRACK.LR
        # cx = bbox[0] + self.center_pos[0]
        # cy = bbox[1] + self.center_pos[1]
        #
        # # smooth bbox
        # width = self.size[0] * (1 - lr) + bbox[2] * lr
        # height = self.size[1] * (1 - lr) + bbox[3] * lr
# ------------------------------------------------------------------ LT
        best_score = score[best_idx]
        if best_score >= cfg.TRACK.CONFIDENCE_LOW:  # 0.85
            cx = bbox[0] + self.center_pos[0]
            cy = bbox[1] + self.center_pos[1]
            # smooth bbox
            width = self.size[0] * (1 - lr) + bbox[2] * lr
            height = self.size[1] * (1 - lr) + bbox[3] * lr
        else:
            cx = self.center_pos[0]
            cy = self.center_pos[1]
            width = self.size[0]
            height = self.size[1]
        if best_score < cfg.TRACK.CONFIDENCE_LOW:  # 0.85
            self.longterm_state = True
        elif best_score > cfg.TRACK.CONFIDENCE_HIGH:  # 0.998
            self.longterm_state = False
# ------------------------------------------------------------------ LT
        # clip boundary
        cx, cy, width, height = self._bbox_clip(cx, cy, width,
                                                height, img.shape[:2])
        # ----------------------------------------------------------------------------------------------------------------------
        if cfg.TRACK.TEMPLATE_UPDATE:  # DROL, XYL 20220927
            pscore_st = pscore_st * (1 - cfg.TRACK.WINDOW_INFLUENCE) + \
                        self.window * cfg.TRACK.WINDOW_INFLUENCE
            best_idx_st = np.argmax(pscore_st)
            bbox_st = pred_bbox_st[:, best_idx_st] / scale_z
            lr_st = penalty_st[best_idx_st] * score_st[best_idx_st] * cfg.TRACK.LR
            if cfg.TRACK.USE_CLASSIFIER and cfg.TRACK.SHORT_TERM_DRIFT and self.lost_count >= 8:
                cx_st, cy_st = bbox_st[0] / 4 + self.center_pos[0], bbox_st[1] / 4 + self.center_pos[1]
            else:
                cx_st, cy_st = bbox_st[0] + self.center_pos[0], bbox_st[1] + self.center_pos[1]
            width_st = self.size[0] * (1 - lr_st) + bbox_st[2] * lr_st
            height_st = self.size[1] * (1 - lr_st) + bbox_st[3] * lr_st
            # cx_st, cy_st, width_st, height_st = self._bbox_clip(cx_st, cy_st, width_st, height_st, img.shape[:2])
            if iou((cx_st, cy_st, width_st, height_st), (cx, cy, width, height), wh=True) >= cfg.TRACK.TAU_REGRESSION \
                    and score_st[best_idx_st] - score[best_idx] >= cfg.TRACK.TAU_CLASSIFICATION:
                cx, cy, width, height, score, best_idx = cx_st, cy_st, width_st, height_st, score_st, best_idx_st  # 公式7，决定使用初始模板or模板库

            iou_1 = iou((cx_st, cy_st, width_st, height_st), (cx, cy, width, height), wh=True)
            # score_1 = score_st[best_idx_st] - score[best_idx]
            score_1 = score_st[best_idx_st]
            if iou_1 >= cfg.TRACK.TAU_REGRESSION and score_1 >= cfg.TRACK.TAU_CLASSIFICATION:
                # print('template_upadta_nums: {0}, iou: {1}, score: {2}'.format(self.update_count,
                #         iou_1, score_1))
                # self.update_count += 1
                cx, cy, width, height = cx_st, cy_st, width_st, height_st  # 公式7，决定使用初始模板or模板库
        # ----------------------------------------------------------------------------------------------------------------------
        # clip boundary
        # cx, cy, width, height = self._bbox_clip(cx, cy, width,
        #                                         height, img.shape[:2])
        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])

        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]
        # best_score = score[best_idx]
# --------------------------------------------------------------------------KalmanFilter
#         show_tensor_np(pscore, 1, title='Classification Max score = {:.4f}'.format(
#             best_score))
        if self.iskalman_work and best_score < cfg.TRACK.CONFIDENCE_LOW:
            self.predict = self.kalman.predict()
            bbox[0] = np.float32(self.predict[0][0])
            bbox[1] = np.float32(self.predict[1][0])  # array --> float64
            # print('Kalman Filter:', self.update_count)
            # self.update_count += 1
            self.iskalman_work = False
        else:
            if (abs(self.predict[0] - bbox[0]) < 2) and (abs(self.predict[1] - bbox[1]) < 2):
                self.kalman_work_num += 1
                if self.kalman_work_num == 3:  # 3
                    self.iskalman_work = True
                    self.kalman_work_num = 0
                # self.iskalman_work = True
            else:
                self.kalman_work_num = 0
            self.kalman.correct(np.array([[np.float32(bbox[0])], [np.float32(bbox[1])]]))
            self.predict = self.kalman.predict()
            # print('no Kalman Filter', self.update_count)
        # self.update_count += 1
# --------------------------------------------------------------------------KalmanFilter
        # return {
        #         'bbox': bbox,
        #         'best_score': best_score
        #        }
        # ----------------------------------------------------------------------------------------------------------------------
        #         if cfg.TRACK.USE_CLASSIFIER:  # DROL, XYL 20220927
        #             self.classifier.update(bbox, scale_z, flag)

        if cfg.TRACK.TEMPLATE_UPDATE:
            if np.max(pscore_st).item() >= cfg.TRACK.TARGET_UPDATE_THRESHOLD:  # 大于更新阈值的先保存下来，找到5帧中的最大值
                if np.max(pscore_st).item() > self.temp_max:
                    self.temp_max = np.max(pscore_st).item()
                    self.channel_average = np.mean(img, axis=(0, 1))
                    self.z_crop = self.get_subwindow(img, self.center_pos, cfg.TRACK.EXEMPLAR_SIZE, s_z,
                                                     self.channel_average)
            if cfg.TRACK.VISUALIZE_CLASS:
                show_tensor_np(pscore_st, 2, title='Classification Max score of Template update = {:.4f}, score = {:.4f}'.format(
                    np.max(pscore_st).item(),np.max(pscore).item()), i=i)
            if (self.frame_num - 1) % cfg.TRACK.TARGET_UPDATE_SKIPPING == 0:  # TARGET_UPDATE_SKIPPING = 5，表示5帧为周期更新模板
                self.temp_max = 0
                with torch.no_grad():
                    self.model.template_short_term(self.z_crop)
                    # self.model.template_short_term(self.z_crop * (1 - cfg.TRACK.COEE_CLASS)
                    #                                + self.z0_crop * cfg.TRACK.COEE_CLASS)

            return {
                'bbox': bbox,
                'best_score': best_score
            }
        else:
            return {
                'bbox': bbox,
                'best_score': best_score
               }
