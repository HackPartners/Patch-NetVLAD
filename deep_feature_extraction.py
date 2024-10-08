import json
import math
import numpy as np
import cv2
import torch
import os 
import pandas as pd
from match_two import match_two
from haversine import haversine
import glob
import time


violation_dict = {}

show_image = False

filter_by_distance = True
filter_by_violation_type = True


distance_threshold = 500  # in meters
score_threshold = 0.2

scan_data = {"00089": {"dates": [], "order": []}}
scan_data = {
    "00089": {
        "dates": ["2312", "2401", "2402", "2403"],
        "order": [1, 2, 3, 4]
    },
    "00112": {
        "dates": ["2312", "2401", "2402", "2403"],
        "order": [1, 2, 3, 4]
    },
    "00668": {
        "dates": ["2312", "2401", "2402", "2403"],
        "order": [1, 2, 3, 4]
    },
    "00977": {
        "dates": ["2311", "2312", "2402", "2403"],
        "order": [1, 2, 3, 4]
    },
    "01148": {
        "dates": ["2308", "2401", "2402", "2403"],
        "order": [1, 2, 3, 4]
    },
    "01299": {
        "dates": ["2306", "2308", "2311", "2402"],
        "order": [1, 2, 3, 4]
    },
    "01321": {
        "dates": ["2311", "2312", "2401"],
        "order": [1, 2, 3]
    },
    "03096": {
        "dates": ["2310", "2401", "2402", "2403"],
        "order": [1, 2, 3, 4]
    },
    "03179": {
        "dates": ["2311", "2401", "2402", "2403"],
        "order": [2, 3, 1, 4]
    },
    "03546": {
        "dates": ["2305", "2308", "2311", "2402"],
        "order": [1, 2, 3, 4]
    },
    }


def get_patch_model():
    """ Load Patch-NetVLAD model, device, and configuration. """
    from match_two import get_backend, get_model, PATCHNETVLAD_ROOT_DIR
    import configparser
    from os.path import join, isfile
    
    config_path = join(PATCHNETVLAD_ROOT_DIR, 'configs/speed.ini')
    assert os.path.isfile(config_path)
    
    config = configparser.ConfigParser()
    config.read(config_path)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get model and encoder
    encoder_dim, encoder = get_backend()
    resume_ckpt = config['global_params']['resumePath'] + config['global_params']['num_pcs'] + '.pth.tar'
    
    if not isfile(resume_ckpt):
        from download_models import download_all_models
        download_all_models(ask_for_permission=True)
    
    checkpoint = torch.load(resume_ckpt, map_location=device)
    assert checkpoint['state_dict']['WPCA.0.bias'].shape[0] == int(config['global_params']['num_pcs'])
    config['global_params']['num_clusters'] = str(checkpoint['state_dict']['pool.centroids'].shape[0])
    model = get_model(encoder, encoder_dim, config['global_params'], append_pca_layer=True)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(device)
    
    return device, model, config

def preprocess_image(path, target_size=(640, 480)):
    img = cv2.imread(path, -1)
    return img


def do_matches(device, model, config, base_scan, initial_scan, target_scan, save_data=False):
    """Match images between two sets using Patch-NetVLAD."""
    
    # Paths to images
    track_folder = f"./image_dataset/{base_scan}"
    initial_scan = f"/{initial_scan}"
    target_scan = f"/{target_scan}"
    
    target_images = sorted(glob.glob(f"{track_folder}{target_scan}/*.png"))
    base_images = sorted(glob.glob(f"{track_folder}{initial_scan}/*.png"))
    
    matches = {}
    
    for i, base_image_path in enumerate(base_images):
        scores = []
        base_image = preprocess_image(base_image_path)
        base_violation = violation_dict[initial_scan.strip('/')][os.path.basename(base_image_path).strip('.png')]
        

        for j, target_image_path in enumerate(target_images):
            target_image = preprocess_image(target_image_path)
            target_violation = violation_dict[target_scan.strip('/')][os.path.basename(target_image_path).strip('.png')]
            
      
            violation_separation = haversine(
                (base_violation["lat"], base_violation["long"]),
                (target_violation["lat"], target_violation["long"]), unit='m'
            )
            
            if base_violation["rule"] != target_violation["rule"] and filter_by_violation_type:
                scores.append(0)
            elif violation_separation > distance_threshold and filter_by_distance:
                scores.append(0)
            else:
                # Use Patch-NetVLAD to get similarity score
                score = match_two(model, device, config, base_image, target_image, plot_save_path=None)
                scores.append(score)
        
        best_match_index = np.argmax(scores)
        if scores[best_match_index] < score_threshold:
            matches[f"{os.path.basename(base_image_path).split('.')[0]}"] = {"match": "none", "score": float(scores[best_match_index])}
        else:
            matches[f"{os.path.basename(base_image_path).split('.')[0]}"] = {"match": f"{os.path.basename(target_images[best_match_index]).split('.')[0]}", "score": round(float(scores[best_match_index]), 4)}

    # save matches
    if save_data:
        with open(f"./output/{initial_scan.strip('/')}_to_{target_scan.strip('/')}.json", 'w') as f:
            json.dump(matches, f, indent=4)

    return matches

def measure_success(my_match_data, gt_data_df, order_a, order_b, load_data=False, path=None):
    if load_data:
        with open(path, 'r') as f:
            my_match_data = json.load(f)

    tp = 0  
    fp = 0  
    tn = 0 
    fn = 0  

    success_scores = []
    fail_scores = []

    for index, row in gt_data_df.iterrows():
        if math.isnan(row[f"scan_{order_a}_violation_id"]):
            continue  # Skip if there's no violation in order_a

        expected = row.get(f"scan_{order_b}_violation_id", None)

        try:
            my_match = my_match_data[str(int(row[f"scan_{order_a}_violation_id"]))]["match"]

            if expected is None or math.isnan(expected):  # Handle true negatives
                if my_match == "none":
                    tn += 1
                    print(f"TN: Base violation {row[f'scan_{order_a}_violation_id']} has no match, prediction also 'none'")
                else:
                    fp += 1  # A match was predicted but should not have been
                    print(f"FP: Base violation {row[f'scan_{order_a}_violation_id']} predicted match {my_match}, but should be 'none'")
            else:
                expected_str = str(int(expected))

                if my_match == "none":
                    fn += 1  # A match should have been found but wasn't
                    print(f"FN: Base violation {row[f'scan_{order_a}_violation_id']} has expected match {expected_str}, but no match found")
                elif my_match == expected_str:
                    tp += 1  # Correct match
                    print(f"TP: Base violation {row[f'scan_{order_a}_violation_id']} matched correctly to {expected_str}")
                    success_scores.append(my_match_data[str(int(row[f"scan_{order_a}_violation_id"]))]["score"])
                else:
                    fp += 1  # Incorrect match
                    print(f"FP: Base violation {row[f'scan_{order_a}_violation_id']} matched to {my_match}, but expected {expected_str}")
                    fail_scores.append(my_match_data[str(int(row[f"scan_{order_a}_violation_id"]))]["score"])

        except Exception as e:
            print(f"Exception: {e}")

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total != 0 else 0
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0

    print(f"Success: {tp}, Fail: {fp}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"Mean Success Score: {np.mean(success_scores):.4f}, Mean Fail Score: {np.mean(fail_scores):.4f}")

    return {
        "n": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "success_scores": success_scores,
        "fail_scores": fail_scores,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall
    }

def populate_violation_dictionary(base_folder):
    global violation_dict

    csv_files = glob.glob(f"./image_dataset/{base_folder}/*.csv")

    for csv_file in csv_files:
        # load csv

        if csv_file.split('/')[-1] == "matches.csv":
            continue

        this_csv_data = pd.read_csv(csv_file)
        this_scan_name = csv_file.split('/')[-1].split('.')[0]
        this_scan_name = this_scan_name.split('_')[0][-4:]

        for index, row in this_csv_data.iterrows():
            if index == 0:
                # we are just using the date to identify the scan.
                violation_dict[this_scan_name] = {}

            violation_dict[this_scan_name][str(row["id"])] = {
                "id": row["id"],
                "lat": row["lat"],
                "long": row["long"],
                "rule": row["rule_id"],
                "path": row["image_path"]
            }


if __name__ == "__main__":

    start = time.time()
    device, model, config = get_patch_model()

    true_positives = 0
    false_positives = 0
    true_nagatives = 0
    false_negatives = 0
    total = 0
    all_success_scores = []
    all_fail_scores = []

    # main()
    # "dates": ["2311", "2401", "2402", "2403"],
    #         "order": [2, 3, 1, 4]}
    for base_scan in scan_data.keys():

        print(f"running folder: {base_scan}")

        populate_violation_dictionary(base_scan)
        gt_data_df = pd.read_csv(f"./image_dataset/{base_scan}/matches.csv")

        for initial_scan, initial_scan_order, \
            target_scan, target_scan_order in \
                zip(scan_data[base_scan]["dates"][0:-1], scan_data[base_scan]["order"][0:-1], \
                    scan_data[base_scan]["dates"][1:], scan_data[base_scan]["order"][1:]):

            print(f"\t{initial_scan} -> {target_scan}")

            #matches = do_matches(base_scan, initial_scan, target_scan, save_data=False, show_image=False)
            matches = do_matches(device, model, config, base_scan, initial_scan, target_scan, save_data=True)
            # target_size=(244, 244)
            # cosine: Success: 22, Fail: 28, accuracy: 0.44
            # Mean Success Score: 0.4517866237597032, Mean Fail Score: 0.4405456068260329

            # target_size=(500, 500)
            # cosine: Success: 22, Fail: 28, accuracy: 0.44
            # Mean Success Score: 0.4023240574381568, Mean Fail Score: 0.3588664872305734

            # return {"n": tp+fp, "tp": tp, "fp": fp, "success_scores": success_scores, "fail_scores": fail_scores}
            metrics = measure_success(matches, gt_data_df, initial_scan_order, target_scan_order)

            true_positives += metrics["tp"]
            false_positives += metrics["fp"]
            total += metrics["n"]
            all_success_scores += metrics["success_scores"]
            all_fail_scores += metrics["fail_scores"]
            true_nagatives += metrics["tn"]
            false_negatives += metrics["fn"]

    print(f"true_positives: {true_positives}")
    print(f"false_positives: {false_positives}")
    print(f"total: {total}")
    print(f"accuracy: {(true_positives + true_nagatives)/total}")
    print(f"precision: {true_positives / (true_positives + false_positives)}")
    print(f"recall: {true_positives / (true_positives + false_negatives)}")
    print(f"success_scores: {np.mean(all_success_scores)}")
    print(f"fail_scores: {np.mean(all_fail_scores)}")

    end = time.time()

    print(f"Time elapsed: {end - start}")
