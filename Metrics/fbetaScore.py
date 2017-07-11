"""
Fbeta Score

The F-beta score is the weighted harmonic mean of precision and recall, 
reaching its optimal value at 1 and its worst value at 0.

"""


def fbetaScore(y_true, y_pred, beta):
	pass


from sklearn.metrics import fbeta_score
y_true = [0, 1, 2, 0, 1, 2]
y_pred = [0, 2, 1, 0, 0, 1]	


print fbeta_score(y_true, y_pred, average='macro', beta=0.5)
