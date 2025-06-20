import tensorflow as tf

from ..registry import LOSSES
from .binary_logistic_regression_loss_tf import binary_logistic_regression_loss


@LOSSES.register_module()
class BMNLoss(tf.keras.Model):
    """BMN Loss.

    From paper https://arxiv.org/abs/1907.09702,
    code https://github.com/JJBOY/BMN-Boundary-Matching-Network.
    It will calculate loss for BMN Model. This loss is a weighted sum of

        1) temporal evaluation loss based on confidence score of start and
        end positions.
        2) proposal evaluation regression loss based on confidence scores of
        candidate proposals.
        3) proposal evaluation classification loss based on classification
        results of candidate proposals.
    """

    def tem_loss(self, pred_start, pred_end, gt_start, gt_end):
        """Calculate Temporal Evaluation Module Loss.

        This function calculate the binary_logistic_regression_loss for start
        and end respectively and returns the sum of their losses.

        Args:
            pred_start (tf.Tensor): Predicted start score by BMN model.
            pred_end (tf.Tensor): Predicted end score by BMN model.
            gt_start (tf.Tensor): Groundtruth confidence score for start.
            gt_end (tf.Tensor): Groundtruth confidence score for end.

        Returns:
            tf.Tensor: Returned binary logistic loss.
        """
        loss_start = binary_logistic_regression_loss(pred_start, gt_start)
        loss_end = binary_logistic_regression_loss(pred_end, gt_end)
        loss = loss_start + loss_end
        return loss

    def pem_reg_loss(self,
                     pred_score,
                     gt_iou_map,
                     mask,
                     high_temporal_iou_threshold=0.7,
                     low_temporal_iou_threshold=0.3):
        """Calculate Proposal Evaluation Module Regression Loss.

        Args:
            pred_score (tf.Tensor): Predicted temporal_iou score by BMN.
            gt_iou_map (tf.Tensor): Groundtruth temporal_iou score.
            mask (tf.Tensor): Boundary-Matching mask.
            high_temporal_iou_threshold (float): Higher threshold of
                temporal_iou. Default: 0.7.
            low_temporal_iou_threshold (float): Higher threshold of
                temporal_iou. Default: 0.3.

        Returns:
            tf.Tensor: Proposal evalutaion regression loss.
        """
        u_hmask = tf.cast(gt_iou_map > high_temporal_iou_threshold, tf.float32)
        u_mmask = tf.cast((gt_iou_map <= high_temporal_iou_threshold) &
                          (gt_iou_map > low_temporal_iou_threshold), tf.float32)
        u_lmask = tf.cast((gt_iou_map <= low_temporal_iou_threshold) &
                          (gt_iou_map > 0.), tf.float32)
        u_lmask = u_lmask * mask

        num_h = tf.reduce_sum(u_hmask)
        num_m = tf.reduce_sum(u_mmask)
        num_l = tf.reduce_sum(u_lmask)

        r_m = num_h / num_m
        u_smmask = tf.random.uniform(shape=tf.shape(gt_iou_map))
        u_smmask = u_mmask * u_smmask
        u_smmask = tf.cast(u_smmask > (1. - r_m), tf.float32)

        r_l = num_h / num_l
        u_slmask = tf.random.uniform(shape=tf.shape(gt_iou_map))
        u_slmask = u_lmask * u_slmask
        u_slmask = tf.cast(u_slmask > (1. - r_l), tf.float32)

        weights = u_hmask + u_smmask + u_slmask

        loss = tf.keras.losses.mean_squared_error(gt_iou_map * weights, pred_score * weights)
        loss = 0.5 * tf.reduce_sum(
            loss * tf.ones_like(weights)) / tf.reduce_sum(weights)

        return loss

    def pem_cls_loss(self,
                     pred_score,
                     gt_iou_map,
                     mask,
                     threshold=0.9,
                     ratio_range=(1.05, 21),
                     eps=1e-5):
        """Calculate Proposal Evaluation Module Classification Loss.

        Args:
            pred_score (tf.Tensor): Predicted temporal_iou score by BMN.
            gt_iou_map (tf.Tensor): Groundtruth temporal_iou score.
            mask (tf.Tensor): Boundary-Matching mask.
            threshold (float): Threshold of temporal_iou for positive
                instances. Default: 0.9.
            ratio_range (tuple): Lower bound and upper bound for ratio.
                Default: (1.05, 21)
            eps (float): Epsilon for small value. Default: 1e-5

        Returns:
            tf.Tensor: Proposal evalutaion classification loss.
        """
        pmask = tf.cast(gt_iou_map > threshold, tf.float32)
        nmask = tf.cast(gt_iou_map <= threshold, tf.float32)
        nmask = nmask * mask

        num_positive = tf.maximum(tf.reduce_sum(pmask), 1.0)
        num_entries = num_positive + tf.reduce_sum(nmask)
        ratio = num_entries / num_positive
        ratio = tf.clip_by_value(ratio, ratio_range[0], ratio_range[1])

        coef_0 = 0.5 * ratio / (ratio - 1)
        coef_1 = 0.5 * ratio

        loss_pos = coef_1 * tf.math.log(pred_score + eps) * pmask
        loss_neg = coef_0 * tf.math.log(1.0 - pred_score + eps) * nmask
        loss = -1 * tf.reduce_sum(loss_pos + loss_neg) / num_entries
        return loss

    def call(self,
             pred_bm,
             pred_start,
             pred_end,
             gt_iou_map,
             gt_start,
             gt_end,
             bm_mask,
             weight_tem=1.0,
             weight_pem_reg=10.0,
             weight_pem_cls=1.0):
        """Calculate Boundary Matching Network Loss.

        Args:
            pred_bm (tf.Tensor): Predicted confidence score for boundary
                matching map.
            pred_start (tf.Tensor): Predicted confidence score for start.
            pred_end (tf.Tensor): Predicted confidence score for end.
            gt_iou_map (tf.Tensor): Groundtruth score for boundary matching
                map.
            gt_start (tf.Tensor): Groundtruth temporal_iou score for start.
            gt_end (tf.Tensor): Groundtruth temporal_iou score for end.
            bm_mask (tf.Tensor): Boundary-Matching mask.
            weight_tem (float): Weight for tem loss. Default: 1.0.
            weight_pem_reg (float): Weight for pem regression loss.
                Default: 10.0.
            weight_pem_cls (float): Weight for pem classification loss.
                Default: 1.0.

        Returns:
            tuple([tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]):
                (loss, tem_loss, pem_reg_loss, pem_cls_loss). Loss is the bmn
                loss, tem_loss is the temporal evaluation loss, pem_reg_loss is
                the proposal evaluation regression loss, pem_cls_loss is the
                proposal evaluation classification loss.
        """
        pred_bm_reg = pred_bm[:, 0]
        pred_bm_cls = pred_bm[:, 1]
        gt_iou_map = gt_iou_map * bm_mask

        pem_reg_loss = self.pem_reg_loss(pred_bm_reg, gt_iou_map, bm_mask)
        pem_cls_loss = self.pem_cls_loss(pred_bm_cls, gt_iou_map, bm_mask)
        tem_loss = self.tem_loss(pred_start, pred_end, gt_start, gt_end)
        loss = (
            weight_tem * tem_loss + weight_pem_reg * pem_reg_loss +
            weight_pem_cls * pem_cls_loss)
        return loss, tem_loss, pem_reg_loss, pem_cls_loss 