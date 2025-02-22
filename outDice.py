import os
import numpy as np
import nibabel as nib
import scipy.spatial.distance as distance
from sklearn.metrics import jaccard_score


def load_nii(file_path):
    return nib.load(file_path).get_fdata()


def dice_coefficient(gt, pred):
    pred = (pred > 0.5).astype(np.uint8)
    intersection = np.sum(gt * pred)
    return (2. * intersection) / (np.sum(gt) + np.sum(pred) + 1e-7)


def iou_score(gt, pred):
    pred = (pred > 0.5).astype(np.uint8)
    return jaccard_score(gt.flatten(), pred.flatten(), average='binary')


def recall(gt, pred):
    pred = (pred > 0.5).astype(np.uint8)
    tp = np.sum(gt * pred)
    fn = np.sum(gt * (1 - pred))
    return tp / (tp + fn + 1e-7)


def precision(gt, pred):
    pred = (pred > 0.5).astype(np.uint8)
    tp = np.sum(gt * pred)
    fp = np.sum((1 - gt) * pred)
    return tp / (tp + fp + 1e-7)


def hausdorff_distance(gt, pred):
    pred = (pred > 0.5).astype(np.uint8)
    gt_points = np.array(np.where(gt > 0)).T
    pred_points = np.array(np.where(pred > 0)).T

    if len(gt_points) == 0 or len(pred_points) == 0:
        return np.inf

    hd1 = np.max([np.min(distance.cdist([p], pred_points)) for p in gt_points])
    hd2 = np.max([np.min(distance.cdist([p], gt_points)) for p in pred_points])
    return max(hd1, hd2)

def evaluate_metrics(gt_dir, pred_dir):
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith('.nii.gz')])
    pred_files = sorted([f for f in os.listdir(pred_dir) if f.endswith('.nii.gz')])

    # assert gt_files == pred_files, "Mismatch between ground truth and prediction files"

    results = []
    dice_scores, iou_scores, recall_scores, precision_scores, hd_scores = [], [], [], [], []

    for file in gt_files:
        gt_path = os.path.join(gt_dir, file)
        pred_path = os.path.join(pred_dir, file)

        gt = load_nii(gt_path).astype(np.uint8)
        pred = load_nii(pred_path.replace('.nii', '_0000.nii')).astype(np.uint8)
        pred = np.squeeze(pred)

        dice = dice_coefficient(gt, pred)
        iou = iou_score(gt, pred)
        rec = recall(gt, pred)
        pre = precision(gt, pred)
        hd = hausdorff_distance(gt, pred)

        dice_scores.append(dice)
        iou_scores.append(iou)
        recall_scores.append(rec)
        precision_scores.append(pre)
        hd_scores.append(hd)

        results.append((file, dice, iou, rec, pre, hd))
        print(f"{file}: Dice={dice:.4f}, IoU={iou:.4f}, Recall={rec:.4f}, Precision={pre:.4f}, HD={hd:.2f}")

    avg_dice = np.mean(dice_scores)
    avg_iou = np.mean(iou_scores)
    avg_recall = np.mean(recall_scores)
    avg_precision = np.mean(precision_scores)
    avg_hd = np.mean(hd_scores)

    print(
        f"Average: Dice={avg_dice:.4f}, IoU={avg_iou:.4f}, Recall={avg_recall:.4f}, Precision={avg_precision:.4f}, HD={avg_hd:.2f}")

    return results


if __name__ == "__main__":
    gt_dir = "mydata/masks"
    pred_dir = "my_results"
    evaluate_metrics(gt_dir, pred_dir)