
import csv
import os
import time
import torch
import argparse
import numpy as np
import nibabel as nib
from torch.utils.data import DataLoader
from utils.metric import calculate_metrics_with_debug
from utils.test_data_loader import BraTSDataset, post_process_prediction

from models.Networks import (base,DENet_l, DENet_s, DENet, base_PGAM, base_PGAM1, base_PGAM2, base_AEMF, base_AEMF1, base_AEMF2, base_AEMF3,
                             base_DME, base_DME1, base_DME2, base_DME3, base_all2, base_all3, base_all4)

class TestTimeAugmentation:
    def __init__(self, device):
        self.device = device

    def augment(self, t1, t1ce, t2, flair):
        # List of flip combinations
        flip_combinations = [
            [],  # no flip
            [2],  # flip H
            [3],  # flip W
            [4],  # flip D
            [2, 3],  # flip H, W
            [2, 4],  # flip H, D
            [3, 4],  # flip W, D
            [2, 3, 4]  # flip H, W, D
        ]

        results = []
        for dims in flip_combinations:
            if dims:
                aug_t1 = torch.flip(t1, dims=dims)
                aug_t1ce = torch.flip(t1ce, dims=dims)
                aug_t2 = torch.flip(t2, dims=dims)
                aug_flair = torch.flip(flair, dims=dims)
            else:
                aug_t1, aug_t1ce, aug_t2, aug_flair = t1, t1ce, t2, flair
            results.append((aug_t1, aug_t1ce, aug_t2, aug_flair))
        return results, flip_combinations

    def reverse_augment(self, predictions, flip_combinations):
        reversed_preds = []
        for pred, dims in zip(predictions, flip_combinations):
            if dims:
                reversed_preds.append(torch.flip(pred, dims=dims))
            else:
                reversed_preds.append(pred)
        return reversed_preds


Checkpoint_path = "/data/qazisami/models/my_model_2/checkpoints/brats2020/base_light_FAM/train_only/base_light_FAM_400_0.1224.pkl"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="brats2020", help='dataset name')
    parser.add_argument('--model', type=str, default='DENet', help='network for training')
    parser.add_argument('--cp', type=str, default=Checkpoint_path, help='model checkpoint for loading model')
    parser.add_argument('--gpu', type=str, default='2', help='GPU ID')
    parser.add_argument('--post_process', action="store_true", help='whether to use post processing')
    parser.add_argument('--save_pred', action="store_true", help='save predictions')
    parser.add_argument('--labels', action="store_true", help='calculate metrics')
    parser.add_argument('--tta', action="store_true", help='use test time augmentation')
    parser.add_argument('--custom_crop', type=str, default=None, help='Path to custom crop coordinates file')
    return parser.parse_args()

def main():

    INF_VALUE = 373.13
    patch_size = 128
    suppress_thr = 300
    connected_comp = 10
    affine_array = np.array([[-1, 0, 0, 0], [0, -1, 0, 239], [0, 0, 1, 0], [0, 0, 0, 1]])

    args = parse_args()
    model_dict = {
        'base': base,
        'DENet': DENet,
        'DENet_l': DENet_l,
        'DENet_s': DENet_s,
        'base_PGAM': base_PGAM,
        'base_PGAM1': base_PGAM1,
        'base_PGAM2': base_PGAM2,
        'base_AEMF': base_AEMF,
        'base_AEMF1': base_AEMF1,
        'base_AEMF2': base_AEMF2,
        'base_AEMF3': base_AEMF3,
        'base_all2': base_all2,
        'base_all3': base_all3,
        'base_all4': base_all4,
        'base_DME': base_DME,
        'base_DME1': base_DME1,
        'base_DME2': base_DME2,
        'base_DME3': base_DME3,
    }
    assert args.model in model_dict.keys(), 'Model name is wrong!'

    if args.dataset == 'brats2019':
        data_root = "/data/qazisami/dataset/BraTS2019/test"
    elif args.dataset == 'brats2020':
        data_root = "/data/qazisami/dataset/BraTS2020/MICCAI_BraTS2020_ValidationData"
    elif args.dataset == 'brats2021':
        data_root = "/data/qazisami/dataset/BraTS2023-GLI/BraTS2023-GLI-ValidationData"
    elif args.dataset == 'brats2023':
        data_root = ("/data/qazisami/dataset/BraTS2023-MEN/BraTS_MEN_Train/test")
    else:
        raise Exception('[error]please provide correct dataset name!')


    print('-' * 100)
    print(f'{args.dataset} Challenge Testing!')
    print(f'Model: {args.model}')
    print(f'Labels - {args.labels} \nPost Processing - {args.post_process} \nTest Time Aug - {args.tta} \nSave Prediction - {args.save_pred} \nGPU ID - {args.gpu} \nPatch Size - {patch_size}')
    print(f'Checkpoint - {args.cp} \nCustom Crops - {args.custom_crop is not None}')

    # Setup device and create save directory if needed
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    use_tta = TestTimeAugmentation(device) if args.tta else None

    if args.save_pred:
        pred_savedir = f'./predictions/{args.dataset}/{args.model}/sub'
        os.makedirs(pred_savedir, exist_ok=True)

    # Initialize model
    model = model_dict[args.model](patch_size, patch_size, patch_size,
                                   in_channels=1, num_classes=4).to(device)
    checkpoint = torch.load(args.cp, map_location=device)
    model.load_state_dict(checkpoint['net'], strict=False)
    model.eval()

    # Initialize dataset and dataloader
    if args.labels:
        dataset = BraTSDataset(data_root, train_dataset=args.dataset, labels=args.labels)
    else:
        dataset = BraTSDataset(data_root, train_dataset=args.dataset, labels=False, smart_crop=True,
                               custom_crops_file=args.custom_crop)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    # Initialize metrics
    metrics = {
        'et': {'dice': [], 'hd95': [], 'sensitivity': [], 'specificity': [],
               'valid_hd': 0, 'penalty_hd': 0, 'total_cases': 0},
        'tc': {'dice': [], 'hd95': [], 'sensitivity': [], 'specificity': [],
               'valid_hd': 0, 'penalty_hd': 0, 'total_cases': 0},
        'wt': {'dice': [], 'hd95': [], 'sensitivity': [], 'specificity': [],
               'valid_hd': 0, 'penalty_hd': 0, 'total_cases': 0}
    }

    # Add this line to create a list for storing case results
    case_results = []

    print(f"\nStarting evaluation on {len(dataset)} cases...\n")
    start_time = time.time()
    with torch.no_grad():
        if args.labels:
            # Evaluation loop with labels
            for batch_idx, (t1, t1ce, t2, flair, true_mask, filename, crop_coords) in enumerate(dataloader):
                print(f"[{batch_idx + 1}/{len(dataset)}]  {filename[0]}:")
                t1, t1ce, t2, flair = t1.to(device), t1ce.to(device), t2.to(device), flair.to(device)
                true_mask = true_mask.to(device)

                if args.tta:
                    # TTA processing
                    aug_inputs, flip_combinations = use_tta.augment(t1, t1ce, t2, flair)
                    predictions = []
                    for aug_t1, aug_t1ce, aug_t2, aug_flair in aug_inputs:
                        pred = model(aug_t1, aug_t1ce, aug_t2, aug_flair)
                        predictions.append(pred)
                    reversed_preds = use_tta.reverse_augment(predictions, flip_combinations)
                    outputs = torch.stack(reversed_preds).mean(dim=0)
                else:
                    outputs = model(t1, t1ce, t2, flair)

                pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()
                true_mask = true_mask[0].cpu().numpy()

                # Post-processing and metrics calculation
                if args.post_process:
                    pred_mask = post_process_prediction(
                        pred_mask,
                        et_threshold=suppress_thr,
                        min_component_size=connected_comp,
                        apply_connected_components=True
                    )

                # Calculate metrics
                # ET metrics
                et_pred = (pred_mask == 3).astype(np.uint8)
                et_true = (true_mask == 3).astype(np.uint8)
                et_metrics = calculate_metrics_with_debug(et_pred, et_true, "ET", inf_value=INF_VALUE)

                # TC metrics
                tc_pred = ((pred_mask == 1) | (pred_mask == 3)).astype(np.uint8)
                tc_true = ((true_mask == 1) | (true_mask == 3)).astype(np.uint8)
                tc_metrics = calculate_metrics_with_debug(tc_pred, tc_true, "TC", inf_value=INF_VALUE)

                # WT metrics
                wt_pred = ((pred_mask == 1) | (pred_mask == 2) | (pred_mask == 3)).astype(np.uint8)
                wt_true = ((true_mask == 1) | (true_mask == 2) | (true_mask == 3)).astype(np.uint8)
                wt_metrics = calculate_metrics_with_debug(wt_pred, wt_true, "WT", inf_value=INF_VALUE)

                # Update metrics
                for region, metrics_val in zip(['et', 'tc', 'wt'], [et_metrics, tc_metrics, wt_metrics]):
                    if metrics_val[0] is not None:
                        metrics[region]['dice'].append(metrics_val[0])
                        metrics[region]['hd95'].append(metrics_val[1])
                        if metrics_val[1] == INF_VALUE:
                            metrics[region]['penalty_hd'] += 1
                        else:
                            metrics[region]['valid_hd'] += 1
                        metrics[region]['sensitivity'].append(metrics_val[2])
                        metrics[region]['specificity'].append(metrics_val[3])
                        metrics[region]['total_cases'] += 1

                # Print current case metrics
                avg_dice = (et_metrics[0] + tc_metrics[0] + wt_metrics[0]) / 3


                et_hd = "INF" if et_metrics[1] == INF_VALUE else f"{et_metrics[1]:.2f}"
                tc_hd = "INF" if tc_metrics[1] == INF_VALUE else f"{tc_metrics[1]:.2f}"
                wt_hd = "INF" if wt_metrics[1] == INF_VALUE else f"{wt_metrics[1]:.2f}"

                if all(x != "INF" for x in [et_hd, tc_hd, wt_hd]):
                    avg_hd = f"{(float(et_hd) + float(tc_hd) + float(wt_hd)) / 3:.2f}"
                else:
                    avg_hd = "INF"

                # Store case results
                case_results.append({
                    'case': filename[0],
                    'dice_et': et_metrics[0] * 100,
                    'dice_tc': tc_metrics[0] * 100,
                    'dice_wt': wt_metrics[0] * 100,
                    'dice_avg': avg_dice * 100,
                    'hd95_et': et_hd,
                    'hd95_tc': tc_hd,
                    'hd95_wt': wt_hd,
                    'hd95_avg': avg_hd
                })

                print(f"Dice  ET: {et_metrics[0] * 100:.2f}%   TC: {tc_metrics[0] * 100:.2f}%   "
                      f"WT: {wt_metrics[0] * 100:.2f}%   AVG: {avg_dice * 100:.2f}%")
                print(f"HD95  ET: {et_hd}     TC: {tc_hd}     WT: {wt_hd}     AVG: {avg_hd}")

                if args.save_pred:
                    save_mask = pred_mask.copy()
                    save_mask[save_mask == 3] = 4  # Convert label 3 to 4 for BraTS format

                    # Create array with correct dimensions (240, 240, 160)
                    full_size_pred = np.zeros((240, 240, 160), dtype=np.float32)

                    # Use crop coordinates for placing prediction
                    x_start, y_start, z_start = crop_coords[0].item(), crop_coords[1].item(), crop_coords[2].item()

                    # Ensure we don't exceed the 160 slice limit
                    z_end = min(z_start + 128, 160)
                    z_length = z_end - z_start

                    # Place prediction back in full size array
                    full_size_pred[x_start:x_start + 128,
                    y_start:y_start + 128,
                    z_start:z_end] = save_mask[:, :, :z_length]

                    # Create and save NIfTI image
                    nifti_img = nib.Nifti1Image(full_size_pred, affine=affine_array)
                    save_path = os.path.join(pred_savedir, f"{filename[0]}.nii.gz")
                    nib.save(nifti_img, save_path)
                    print(f"Saved prediction with shape: {full_size_pred.shape}")

        else:
            # Evaluation loop without labels
            for batch_idx, (t1, t1ce, t2, flair, filename, crop_coords) in enumerate(dataloader):
                print(f"\n[{batch_idx + 1}/{len(dataset)}]  {filename[0]}:")
                t1, t1ce, t2, flair = t1.to(device), t1ce.to(device), t2.to(device), flair.to(device)

                if args.tta:
                    # TTA processing
                    aug_inputs, flip_combinations = use_tta.augment(t1, t1ce, t2, flair)
                    predictions = []
                    for aug_t1, aug_t1ce, aug_t2, aug_flair in aug_inputs:
                        pred = model(aug_t1, aug_t1ce, aug_t2, aug_flair)
                        predictions.append(pred)
                    reversed_preds = use_tta.reverse_augment(predictions, flip_combinations)
                    outputs = torch.stack(reversed_preds).mean(dim=0)
                else:
                    outputs = model(t1, t1ce, t2, flair)

                pred_mask = torch.argmax(outputs, dim=1)[0].cpu().numpy()

                # Post-processing
                if args.post_process:
                    pred_mask = post_process_prediction(
                        pred_mask,
                        et_threshold=suppress_thr,
                        min_component_size=connected_comp,
                        apply_connected_components=True
                    )

                if args.save_pred:
                    save_mask = pred_mask.copy()
                    save_mask[save_mask == 3] = 4  # Convert label 3 to 4 for BraTS format

                    # Create array with correct dimensions (240, 240, 155)
                    full_size_pred = np.zeros((240, 240, 155), dtype=np.float32)

                    # Use crop coordinates for placing prediction
                    x_start, y_start, z_start = crop_coords[0].item(), crop_coords[1].item(), crop_coords[2].item()

                    # Ensure we don't exceed the 155 slice limit
                    z_end = min(z_start + 128, 155)
                    z_length = z_end - z_start

                    # Place prediction back in full size array
                    full_size_pred[x_start:x_start + 128,
                    y_start:y_start + 128,
                    z_start:z_end] = save_mask[:, :, :z_length]

                    # Create and save NIfTI image
                    nifti_img = nib.Nifti1Image(full_size_pred, affine=affine_array)
                    save_path = os.path.join(pred_savedir, f"{filename[0]}.nii.gz")
                    nib.save(nifti_img, save_path)
                    print(f"Saved prediction with shape: {full_size_pred.shape}")


    if args.labels and case_results:
        # Create results directory
        results_dir = f'./results/{args.dataset}/{args.model}'
        os.makedirs(results_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        csv_filename = os.path.join(results_dir, f'case_metrics_{timestamp}.csv')

        # Write results to CSV in simplified format
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                ['Case', 'DSC_ET', 'DSC_TC', 'DSC_WT', 'DSC_AVG', 'HD95_ET', 'HD95_TC', 'HD95_WT', 'HD95_AVG'])

            for result in case_results:
                writer.writerow([
                    result['case'],
                    f"{result['dice_et']:.2f}",
                    f"{result['dice_tc']:.2f}",
                    f"{result['dice_wt']:.2f}",
                    f"{result['dice_avg']:.2f}",
                    result['hd95_et'],
                    result['hd95_tc'],
                    result['hd95_wt'],
                    result['hd95_avg']
                ])

        print(f"\nCase metrics saved to {csv_filename}")

    if args.labels:
        # Calculate and print final results
        print("\n")
        print("-" * 70)
        for region in ['et', 'tc', 'wt']:
            results = {
                'dice': np.nanmean(metrics[region]['dice']) if metrics[region]['dice'] else 0.0,
                'hd95': np.nanmean(metrics[region]['hd95']) if metrics[region]['hd95'] else INF_VALUE,
                'sensitivity': np.nanmean(metrics[region]['sensitivity']) if metrics[region]['sensitivity'] else 0.0,
                'specificity': np.nanmean(metrics[region]['specificity']) if metrics[region]['specificity'] else 0.0
            }

            print(f"{region.upper()} Metrics")
            print(
                f"Dice: {results['dice'] * 100:.2f}%   HD95: {results['hd95']:.2f}   Sensitivity: {results['sensitivity'] * 100:.2f}%   Specificity: {results['specificity'] * 100:.2f}%")
            print(
                f"Valid HD95: {metrics[region]['valid_hd']}  Penalty HD95 ({INF_VALUE}): {metrics[region]['penalty_hd']}  Total cases: {metrics[region]['total_cases']}")
            #print("-" * 70)

    total_time = (time.time() - start_time) / 60
    print(f"Testing completed. Total time: {total_time:.2f} minutes")
    print(f'Checkpoint - {args.cp}')

if __name__ == "__main__":
    main()