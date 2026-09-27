import numpy as np

def compute_macro_f05(ground_truth_dict, predictions_dict):
    """
    Computes the competition Macro F_0.5 score.
    
    ground_truth_dict: {s1_id: set([matched_ids])}
    predictions_dict:  {s1_id: set([predicted_ids])}
    """
    scores = []
    
    for s1_id, y_true in ground_truth_dict.items():
        y_pred = predictions_dict.get(s1_id, set())
        
        # 1. Singleton: true match set is empty
        if len(y_true) == 0:
            scores.append(1.0 if len(y_pred) == 0 else 0.0)
            continue
            
        # 2. Missed match: predicted set is empty
        if len(y_pred) == 0:
            scores.append(0.0)
            continue
            
        # 3. Precision & Recall calculation
        tp = len(y_true & y_pred)
        if tp == 0:
            scores.append(0.0)
            continue
            
        precision = tp / len(y_pred)
        recall = tp / len(y_true)
        
        # F_0.5 formula (beta = 0.5 -> beta^2 = 0.25)
        # (1 + 0.25) * P * R / (0.25 * P + R)
        denom = (0.25 * precision) + recall
        f05 = (1.25 * precision * recall) / denom if denom > 0 else 0.0
        scores.append(f05)
        
    return float(np.mean(scores))