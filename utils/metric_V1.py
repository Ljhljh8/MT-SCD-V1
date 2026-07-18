import math
import os.path
from scipy.stats import hmean
import numpy as np

import itertools
import numpy as np
import pandas as pd
import matplotlib
# import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from utils.colormap import heatmap, annotate_heatmap


def cal_kappa(hist):
    if hist.sum() == 0:
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)
    return kappa


class IOUandSek:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def _fast_hist(self, label_pred, label_true):

        mask = (label_pred >= 0) & (label_pred < self.num_classes)

        # 妫€鏌� mask 鍚庣殑闀垮害
        true_len = len(label_true[mask])
        pred_len = len(label_pred[mask])
        label_l = label_true[mask].astype(int)
        label_p = label_pred[mask]
        # 妫€鏌ユ槸鍚︾浉绛変笖闈為浂
        assert true_len == pred_len and true_len > 0, "Label lengths after masking do not match or are zero"

        # 妫€鏌ユ渶澶х储寮曞€�
        max_index = max(self.num_classes * label_true[mask].astype(int) + label_pred[mask])
        max_index_tensor = self.num_classes * label_true[mask].astype(int) + label_pred[mask]
        # print(f"Max index: {max_index}, Expected max: {self.num_classes ** 2 - 1}")

        # 纭繚 num_classes 鏄綘鏈熸湜鐨勫€�
        # print(f"Number of classes: {self.num_classes}")

        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) +
            label_pred[mask], minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            # print((lp.flatten()).shape, (lt.flatten()).shape)
            # print((lt.flatten()).shape)
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def reset(self):

        """閲嶇疆缁熻鏁版嵁"""

        self.hist = np.zeros((self.num_classes, self.num_classes))

    def color_map_WUSU(self, path):
        ax = plt.plot()
        # y = ['Road', 'Low building', 'High building', 'ArableLand', 'unknown', 'Woodland', 'Grassland', 'water', 'lake', 'structure', 'excavation', 'bare']
        # x = ['Road', 'Low building', 'High building', 'ArableLand', 'unknown', 'Woodland', 'Grassland', 'water', 'lake', 'structure', 'excavation', 'bare']
        y = ['nochange', 'Road', 'Low building', 'High building', 'ArableLand', 'unknown', 'Woodland', 'Grassland',
             'water', 'lake', 'structure', 'excavation', 'bare']
        x = ['nochange', 'Road', 'Low building', 'High building', 'ArableLand', 'unknown', 'Woodland', 'Grassland',
             'water', 'lake', 'structure', 'excavation', 'bare']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                        cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                         textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_WUSU.png')
        plt.savefig(save_path, transparent=True, dpi=800)

    def color_map_SECOND(self, path):
        ax = plt.plot()
        # y = ['No change', 'Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        # x = ['No change', 'Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        y = ['Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        x = ['Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        # confusion = np.array(self.hist, dtype=int)
        # confusion[0][0] = 0
        confusion = np.array(self.hist[1:, 1:], dtype=int)
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                        cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                         textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_SECOND.png')
        plt.savefig(save_path, transparent=True, dpi=800)

    def color_map_FZSCD(self, path):
        ax = plt.plot()
        y = ['No change', 'Bare', 'Building', 'Vegetation', 'Water', 'Road', 'others']
        x = ['No change', 'Bare', 'Building', 'Vegetation', 'Water', 'Road', 'others']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                        cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                         textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_SECOND.png')
        plt.savefig(save_path, transparent=True, dpi=800)

    def color_map_DynamicEarth(self, path):
        ax = plt.plot()
        # y = ['No change', 'Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        # x = ['No change', 'Water', 'Ground', 'Low vegetation', 'Tree', 'Building', 'Playground']
        y = ['nochange', 'bg', 'impervious surface', 'agriculture', 'forest & other vegetation', 'wetlands', 'soil',
             'water', 'snow & ice']
        x = ['nochange', 'bg', 'impervious surface', 'agriculture', 'forest & other vegetation', 'wetlands', 'soil',
             'water', 'snow & ice']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                        cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                         textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_DynamicEarth.png')
        plt.savefig(save_path, transparent=True, dpi=800)

    def color_map_Landsat_SCD(self, path):
        ax = plt.plot()
        y = ['No change', 'Farmland', 'Desert', 'Building', 'Water']
        x = ['No change', 'Farmland', 'Desert', 'Building', 'Water']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                        cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", threshold=20,
                         textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = os.path.join(path, 'Confusion_Matrix_LandSat_SCD.png')
        plt.savefig(save_path, transparent=True, dpi=800)

    def color_map_HRSCD(self):
        ax = plt.plot()
        y = ['No change', 'Artificial surfaces', 'Agricultural area', 'Forests', 'Wetlands', 'Water']
        x = ['No change', 'Artificial surfaces', 'Agricultural area', 'Forests', 'Wetlands', 'Water']
        confusion = np.array(self.hist, dtype=int)
        confusion[0][0] = 0
        im, _ = heatmap(confusion, y, x, ax=ax, vmin=0,
                        cmap="magma_r", cbarlabel="transition countings")
        annotate_heatmap(im, valfmt="{x:d}", size=6, threshold=20,
                         textcolors=("red", "green"), fontsize=6)
        plt.tight_layout()
        save_path = '/disk527/Datadisk/b527_cfz/SenseEarth2020-ChangeDetection/utils/method_1.png'
        plt.savefig(save_path, transparent=True, dpi=800)

    def evaluate(self):
        hist = self.hist
        TN, FP, FN, TP = hist[0][0], hist[1][0], hist[0][1], hist[1][1]
        pr = TP / (TP + FP)  # precision
        re = TP / (TP + FN)  # recall
        F1 = 2 * pr * re / (pr + re)
        return F1

    def evaluate_SECOND(self):
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()

        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        miou = np.mean(iou)

        hist = self.hist.copy()
        OA = (np.diag(hist).sum()) / (hist.sum())
        hist[0][0] = 0
        kappa = cal_kappa(hist)
        sek = kappa * math.exp(iou[1] - 1)

        score = 0.3 * miou + 0.7 * sek

        pixel_sum = self.hist.sum()
        # self.hist rows are ground truth; columns are predictions.
        gt_change_sum = self.hist[1:, :].sum()
        pred_change_sum = self.hist[:, 1:].sum()
        change_ratio = gt_change_sum / pixel_sum if pixel_sum > 0 else 0.0
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = (
            SC_TP / pred_change_sum if pred_change_sum > 0 else 0.0
        )
        SC_Recall = SC_TP / gt_change_sum if gt_change_sum > 0 else 0.0

        Fscd = (
            stats.hmean([SC_Precision, SC_Recall])
            if SC_Precision > 0 and SC_Recall > 0
            else 0.0
        )

        return change_ratio, score, miou, sek, Fscd, OA, SC_Precision, SC_Recall

    def evaluate_classification(self):
        """
        璁＄畻鍦熷湴瑕嗙洊鍒嗙被鐨勮瘎浠锋寚鏍�

        杩斿洖:
            miou (float): 骞冲潎浜ゅ苟姣� (mIoU)
            oa (float): 鎬讳綋绮惧害 (Overall Accuracy)
            f1 (float): 瀹忓钩鍧嘑1鍒嗘暟
            precision (float): 瀹忓钩鍧囩簿纭巼
            recall (float): 瀹忓钩鍧囧彫鍥炵巼
        """
        # 璁＄畻鏍囧噯鍒嗙被鎸囨爣
        hist = self.hist.copy()  # 浣跨敤瀹屾暣鐨勬贩娣嗙煩闃�

        # 1. 鎬讳綋绮惧害 (OA)
        oa = np.diag(hist).sum() / hist.sum()

        # 2. 璁＄畻姣忎釜绫诲埆鐨処oU
        iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        miou = np.nanmean(iou)  # 骞冲潎浜ゅ苟姣� (mIoU)

        # 3. 璁＄畻姣忎釜绫诲埆鐨勭簿纭巼鍜屽彫鍥炵巼
        precision_per_class = np.diag(hist) / (hist.sum(0) + 1e-10)
        recall_per_class = np.diag(hist) / (hist.sum(1) + 1e-10)
        f1_per_class = 2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class + 1e-10)

        # 4. 璁＄畻瀹忓钩鍧囨寚鏍�
        precision = np.nanmean(precision_per_class)  # 瀹忓钩鍧囩簿纭巼
        recall = np.nanmean(recall_per_class)  # 瀹忓钩鍧囧彫鍥炵巼
        f1 = np.nanmean(f1_per_class)  # 瀹忓钩鍧嘑1鍒嗘暟

        return miou, oa, f1, precision, recall

    def evaluate_WUSU(self):
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()

        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        miou = np.mean(iou)

        hist = self.hist.copy()
        OA = (np.diag(hist).sum()) / (hist.sum())
        hist[0][0] = 0
        kappa = cal_kappa(hist)

        pixel_sum = self.hist.sum()
        change_pred_sum = pixel_sum - self.hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - self.hist.sum(0)[0].sum()
        change_ratio = change_label_sum / pixel_sum
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = SC_TP / change_pred_sum
        if change_pred_sum == 0:
            SC_Precision = 0
        SC_Recall = SC_TP / change_label_sum
        if change_label_sum == 0:
            SC_Recall = 0

        Fscd = stats.hmean([SC_Precision, SC_Recall])

        return miou, Fscd, OA

    def evaluate_BCD1(self):
        """
        璁＄畻鍙屾椂鐩稿缓绛戠墿鍙樺寲妫€娴嬬殑绮惧害璇勪环鎸囨爣銆�
        杩斿洖: Recall, Precision, OA, F1, IoU, KC
        """
        # 纭繚娣锋穯鐭╅樀鏄�2x2鐨�
        if self.hist.shape != (2, 2):
            # 濡傛灉涓嶆槸2x2锛屽皾璇曟彁鍙栧墠2x2閮ㄥ垎
            if self.hist.shape[0] >= 2 and self.hist.shape[1] >= 2:
                hist = self.hist[:2, :2]
            else:
                # 鏃犳硶璁＄畻锛岃繑鍥�0
                return 0, 0, 0, 0, 0, 0
        else:
            hist = self.hist

        # 鐩存帴浣跨敤娣锋穯鐭╅樀鐨勫€�
        TN = hist[0][0]
        FP = hist[0][1]
        FN = hist[1][0]
        TP = hist[1][1]

        # 璁＄畻鍙洖鐜� (Recall)
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # 璁＄畻绮剧‘搴� (Precision)
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        # 璁＄畻鎬讳綋绮惧害 (OA)
        OA = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0

        # 璁＄畻 F1 鍒嗘暟
        F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) > 0 else 0

        # 璁＄畻 IoU (Intersection over Union)
        IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

        # 璁＄畻 Kappa 绯绘暟 (KC)
        total = TP + FP + FN + TN
        p_o = OA  # 瑙傚療涓€鑷存€�
        p_e = ((TP + FP) * (TP + FN) + (FN + TN) * (FP + TN)) / (total ** 2) if total > 0 else 0
        KC = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0

        return Recall, Precision, OA, F1, IoU, KC

    def evaluate_BCD(self):
        """
        璁＄畻鍙屾椂鐩稿缓绛戠墿鍙樺寲妫€娴嬬殑绮惧害璇勪环鎸囨爣銆�
        杩斿洖: Recall, Precision, OA, F1, IoU, KC
        """
        # 鏋勫缓娣锋穯鐭╅樀
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]  # TN (True Negative)
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]  # FP (False Positive)
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]  # FN (False Negative)
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()  # TP (True Positive)

        # 璁＄畻鍙洖鐜� (Recall)
        TP = confusion_matrix[1][1]
        FN = confusion_matrix[1][0]
        Recall = TP / (TP + FN) if (TP + FN) > 0 else 0

        # 璁＄畻绮剧‘搴� (Precision)
        FP = confusion_matrix[0][1]
        Precision = TP / (TP + FP) if (TP + FP) > 0 else 0

        # 璁＄畻鎬讳綋绮惧害 (OA)
        OA = (TP + confusion_matrix[0][0]) / confusion_matrix.sum()

        # 璁＄畻 F1 鍒嗘暟
        F1 = hmean([Precision, Recall]) if (Precision > 0 and Recall > 0) else 0

        # 璁＄畻 IoU (Intersection over Union)
        IoU = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

        # 璁＄畻 Kappa 绯绘暟 (KC)
        total = confusion_matrix.sum()
        p_o = OA  # 瑙傚療涓€鑷存€�
        p_e = ((TP + FP) * (TP + FN) + (FN + confusion_matrix[0][0]) * (FP + confusion_matrix[0][0])) / (
                total ** 2)  # 闅忔満涓€鑷存€�
        KC = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else 0

        return Recall, Precision, OA, F1, IoU, KC

    def evaluate_inference(self):
        ''' BCD '''
        confusion_matrix = np.zeros((2, 2))
        confusion_matrix[0][0] = self.hist[0][0]
        confusion_matrix[0][1] = self.hist.sum(1)[0] - self.hist[0][0]
        confusion_matrix[1][0] = self.hist.sum(0)[0] - self.hist[0][0]
        confusion_matrix[1][1] = self.hist[1:, 1:].sum()

        iou = np.diag(confusion_matrix) / (confusion_matrix.sum(0) +
                                           confusion_matrix.sum(1) - np.diag(confusion_matrix))
        miou = np.mean(iou)
        TN, FP, FN, TP = confusion_matrix[0][0], confusion_matrix[1][0], confusion_matrix[0][1], confusion_matrix[1][1]
        pr = TP / (TP + FP)  # precision
        re = TP / (TP + FN)  # recall
        F1 = 2 * pr * re / (pr + re)

        ''' SCD '''
        hist = self.hist.copy()
        oa = (np.diag(hist).sum()) / (hist.sum())
        hist[0][0] = 0
        kappa = cal_kappa(hist)
        sek = kappa * math.exp(iou[1] - 1)

        score = 0.3 * miou + 0.7 * sek

        pixel_sum = self.hist.sum()
        change_pred_sum = pixel_sum - self.hist.sum(1)[0].sum()
        change_label_sum = pixel_sum - self.hist.sum(0)[0].sum()
        change_ratio = change_label_sum / pixel_sum
        SC_TP = np.diag(hist[1:, 1:]).sum()
        SC_Precision = SC_TP / change_pred_sum
        if change_pred_sum == 0:
            SC_Precision = 0
        SC_Recall = SC_TP / change_label_sum
        if change_label_sum == 0:
            SC_Recall = 0
        Fscd = stats.hmean([SC_Precision, SC_Recall])
        # acc = np.diag(hist).sum() / (hist.sum() + 1e-10)
        # return change_ratio, score, miou, sek, Fscd, oa, iou[1], F1, kappa, pr, re

        return change_ratio, oa, miou, sek, Fscd, score, SC_Precision, SC_Recall

    def miou(self):
        confusion_matrix = self.hist[1:, 1:]
        iou = np.diag(confusion_matrix) / (
                    confusion_matrix.sum(0) + confusion_matrix.sum(1) - np.diag(confusion_matrix))
        return iou, np.mean(iou)