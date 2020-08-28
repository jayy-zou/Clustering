import numpy as np
from sklearn.metrics import adjusted_mutual_info_score

def adjusted_mutual_info(predicted_labels, predicted_targets):
    return adjusted_mutual_info_score(predicted_labels, predicted_targets)
