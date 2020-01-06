# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes, ignore_index=-1):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class) & (label_true != self.ignore_index)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask], minlength=n_class ** 2
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten(), self.n_classes)

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "Overall Acc: \t": acc,
                "Mean Acc : \t": acc_cls,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

class runningUncertaintyScore(object):
    def __init__(self, n_classes, ignore_index=-1, name="", scale_uncertainty=True):
        self.n_classes = n_classes
        self.ignore_index = ignore_index
        self.scale_uncertainty = scale_uncertainty
        self.name = name
        self.ece = 0
        #TODO add other metrics
        self.count  = 0

    def get_ause(self, label_true, label_pred, uncertainty, softmax_output):
        """
        get ause based on brier score
        """
        raise NotImplementedError("not implemented yet")

    def _calc_acc(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return (y_true == y_pred).sum()/float(y_pred.shape[0])

    def get_caliberation_errors(self, label_true, label_pred, conf, n_bins=15):
        """
        get expected caliberation error for confidence
        """
        # if self.scale_uncertainty :
        #     #min max scale uncertainty 
        #     uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())
        bins = np.linspace(0,1.01, n_bins+1)
        inds = np.digitize(conf, bins)
        calib_error = 0
        
        for i in range(1, n_bins+1):
            subset = (inds == i)
            if not np.any(subset):
                continue
            true_sub = label_true[subset]
            pred_sub = label_pred[subset]
            uncertainty_sub = conf[subset]
            acc = self._calc_acc(true_sub, pred_sub)
            calib_error += np.abs(acc - conf.mean()) * float(uncertainty_sub.shape[0]) 
            #TODO plot acc vs uncertainty curve 
        ece = calib_error/float(conf.shape[0])
        return ece

    def update(self, label_trues, label_preds, softmax_outputs, uncertainties):
        for lt, lp, uc, sm in zip(label_trues, label_preds, uncertainties, softmax_outputs):
            lt = lt.flatten()
            lp = lp.flatten()
            conf = np.max(sm, axis=2)
            conf = conf.flatten()

            uc = uc.flatten()
            idx = (lt != self.ignore_index) & (lt < self.n_classes)
            lt  = lt[idx]
            lp = lp[idx]
            uc = uc[idx]
            conf = conf[idx]

            self.ece = (self.ece * self.count + self.get_caliberation_errors(lt, lp, conf))/float(self.count + 1)
            self.count+=1

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        ece = self.ece

        return (
            {
                "Overall ECE using max class score in the softmax:" + self.name + ": \t": ece,
            }
        )

    def reset(self):
        self.ece = 0


class averageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
