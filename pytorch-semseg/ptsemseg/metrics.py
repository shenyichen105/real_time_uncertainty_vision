# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np
from sklearn.metrics import roc_auc_score


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
        #self.ause = 0
        #self.auc_miss = 0
        #TODO add other metrics
        self.count  = 0

        self.brier_score = np.array([])
        self.uncertainty = np.array([]) 
        self.label_true = np.array([])
        self.label_pred = np.array([])
        self.conf = np.array([])

    def _calculate_brier_score(self, label_true, softmax_output):
        """
        calculate brier score
        """
        one_hot = np.zeros(softmax_output.shape)
        one_hot[np.arange(one_hot.shape[0]), label_true] = 1
        dist = (one_hot - softmax_output)
        brier_score = np.mean(dist**2, axis=-1)
        return brier_score

    def get_ause(self, label_true, label_pred, uncertainty, softmax_output):
        """
        get ause based on brier score
        """
        brier_score = self._calculate_brier_score(label_true, softmax_output)
        #normalize
        ause = self._calculate_ause(uncertainty,brier_score)
        return ause
    
    def _calculate_ause(self, uncertainty, brier_score):
        """
        get ause based on brier score
        """
        #normalize
        brier_score /= np.max(brier_score)
        ind_oracle = np.argsort(brier_score)
        ind_uncertainty = np.argsort(uncertainty)
        
        avg_brier_oracle = np.cumsum(brier_score[ind_oracle])/(np.arange(len(brier_score)) + 1)
        avg_brier_w_uncertainty = np.cumsum(brier_score[ind_uncertainty])/(np.arange(len(brier_score)) + 1)
        ause = np.trapz((avg_brier_w_uncertainty - avg_brier_oracle), x=np.linspace(0, 1, num=len(brier_score)))
        return ause

    def _calculate_auc_misdetection(self, label_true, label_pred, uncertainty):
        misdetect = (label_true !=label_pred) 
        auc = roc_auc_score(misdetect, uncertainty)
        return auc

    def _calc_acc(self, y_true, y_pred):
        y_true = y_true.flatten()
        y_pred = y_pred.flatten()
        return (y_true == y_pred).sum()/float(y_pred.shape[0])

    def get_caliberation_errors(self, label_true, label_pred, conf, n_bins=15):
        """
        get expected caliberation error for confidence
        """
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

            n_classes = sm.shape[2]
            sm = sm.reshape(-1, n_classes)

            uc = uc.flatten()
            idx = (lt != self.ignore_index) & (lt < self.n_classes)
            lt  = lt[idx]
            lp = lp[idx]
            uc = uc[idx]
            conf = conf[idx]
            sm = sm[idx]
            #self.ece = (self.ece * self.count + self.get_caliberation_errors(lt, lp, conf))/float(self.count + 1)
            #self.ause = (self.ause * self.count + self.get_ause(lt, lp, uc, sm))/float(self.count + 1)
            #self.auc_miss = (self.auc_miss * self.count + self._calculate_auc_misdetection(lt, lp, uc))/float(self.count + 1)
            self.count+=1

            self.brier_score = np.concatenate([self.brier_score, self._calculate_brier_score(lt, sm)])
            self.uncertainty = np.concatenate([self.uncertainty, uc])
            self.label_true = np.concatenate([self.label_true, lt])
            self.label_pred = np.concatenate([self.label_pred, lp])
            self.conf = np.concatenate([self.conf, conf])

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        #ece = self.ece
        #ause = self.ause
        #auc_miss = self.auc_miss
        ece = self.get_caliberation_errors(self.label_true, self.label_pred, self.conf)
        ause = self._calculate_ause(self.uncertainty, self.brier_score)
        auc_miss = self._calculate_auc_misdetection(self.label_true, self.label_pred, self.uncertainty)
        return (
            {
                "Overall ECE using max class score in the softmax:" + self.name + ": \t": ece,
                "mean AUSE using brier score as oracle:" + self.name + ": \t": ause,
                "mean AUC_misdetect:" + self.name + ": \t": auc_miss,
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
