import numpy as np

def apk(actual, predicted, k):
    """
    Average precision at k
    """

    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)



y_true = [1,2,3,4,5]
y_predicted = [3,4,1,7,5]

print apk(y_true, y_predicted, 2)
print apk(y_true, y_predicted, 3)
print apk(y_true, y_predicted, 4)
