from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score
import numpy as np
import cv2


def score_fnc(y_true, y_prob_raw, metric='auc'):
	if metric == 'auc':
		auc = roc_auc_score(y_true, y_prob_raw)
		return auc
	elif metric == 'recall':
		recall = recall_score(y_true, y_prob_raw > 0.5)
		return recall
	elif metric == 'f1':
		f1 = f1_score(y_true, y_prob_raw > 0.5)
		return f1
	elif metric == 'accuracy':
		accuracy = accuracy_score(y_true, y_prob_raw > 0.5)
		return accuracy
	else:
		raise ValueError(f'Metric not supported: {metric}')
	
def BootStrap(y_true, y_prob_raw, n_bootstraps, metric = 'auc'):
	"""
	Calculates the difference in scores for each bootstrap iteration
	returns sorted_scores, confidence_lower, confidence_upper
	"""
	
	n_bootstraps = n_bootstraps
	rng_seed = 42  # control reproducibility
	bootstrapped_scores = []


	rng = np.random.RandomState(rng_seed)
	
	for i in range(n_bootstraps):
		# bootstrap by sampling with replacement on the prediction indices
		indices = rng.randint(0, len(y_prob_raw), len(y_prob_raw))

		if len(np.unique(y_prob_raw[indices])) < 2:
			continue
		else:
			score = score_fnc(y_true[indices], y_prob_raw[indices], metric=metric)
			bootstrapped_scores.append(score)
		

	sorted_scores = np.array(bootstrapped_scores)
	sorted_scores.sort()
	if len(sorted_scores)==0:
		return 0., 0.
	confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
	confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]
	return sorted_scores, confidence_lower, confidence_upper

def BootStrap_Compare(y_true, y_prob_raw1, y_prob_raw2, n_bootstraps, metric='auc' ):
	"""
	Calculates the difference in scores for each bootstrap iteration
	Returns three tuples of (sorted_scores, confidence_lower, confidence_upper)
	"""
	# initialization by bootstraping
	n_bootstraps = n_bootstraps
	rng_seed = 42  # control reproducibility
	bootstrapped_scores1 = []
	bootstrapped_scores2 = []
	
	
	rng = np.random.RandomState(rng_seed)
	
	for i in range(n_bootstraps):
		indices = rng.randint(0, len(y_prob_raw1), len(y_prob_raw1))
		
		if len(np.unique(y_prob_raw1[indices])) < 2:
			continue
		else:
			score1 = score_fnc(y_true[indices], y_prob_raw1[indices], metric=metric)
			score2 = score_fnc(y_true[indices], y_prob_raw2[indices], metric=metric)
			
			bootstrapped_scores1.append(score1)
			bootstrapped_scores2.append(score2)
		
	bootstrapped_scores_diff = np.array(bootstrapped_scores1) - np.array(bootstrapped_scores2)
	sorted_scores1 = np.array(bootstrapped_scores1)
	sorted_scores2 = np.array(bootstrapped_scores2)
	sorted_scores_diff = np.array(bootstrapped_scores_diff)
	
	confidence_lower1 = sorted_scores1[int(0.025 * len(sorted_scores1))]
	confidence_upper1 = sorted_scores1[int(0.975 * len(sorted_scores1))]
	
	confidence_lower2 = sorted_scores2[int(0.025 * len(sorted_scores2))]
	confidence_upper2 = sorted_scores2[int(0.975 * len(sorted_scores2))]
	
	confidence_lower_diff = sorted_scores_diff[int(0.025 * len(sorted_scores_diff))]
	confidence_upper_diff = sorted_scores_diff[int(0.975 * len(sorted_scores_diff))]
	
	return (sorted_scores1, confidence_lower1, confidence_upper1), (sorted_scores2, confidence_lower2, confidence_upper2), (sorted_scores_diff, confidence_lower_diff, confidence_upper_diff)

def Jaccard_Index(mask1, mask2):
	if mask1.shape != mask2.shape:
		raise ValueError("Shape mismatch: mask1 and mask2 must have the same shape.")
	intersection = np.logical_and(mask1, mask2)
	union = np.logical_or(mask1, mask2)
	return np.sum(intersection) / np.sum(union)

def Dice_Coefficient(prediction_mask, truth_mask):
	if prediction_mask.shape != truth_mask.shape:
		raise ValueError("Shape mismatch: mask1 and mask2 must have the same shape.")
	
	intersection = np.logical_and(prediction_mask, truth_mask)
	
	return  (2 * np.sum(intersection) / (np.sum(prediction_mask/255) + np.sum(truth_mask/255)))