import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics

# setup the two dimensional inputs
precisions = np.linspace(0.01, 1, 100)
recalls = np.linspace(0.01, 1, 100)
points = np.array(np.meshgrid(precisions, recalls)).T.reshape(-1, 2)

# the objective value here is just the average
objective_values = (points[:,0] + points[:,1])/2

img = objective_values.reshape((100,100)) # rows are precisions and cols are recalls

f = plt.figure()
plt.imshow(img, origin='lower')
plt.xlabel('Recall', fontsize=28)
plt.ylabel('Precision', fontsize=28)
plt.xticks(np.linspace(0, 100, 6))
plt.yticks(np.linspace(0, 100, 6))
plt.colorbar()
plt.title('Average of Precision and Recall', fontsize=14)

plt.savefig('tmp/block10_precision_recall_average.png', dpi=120, bbox_inches='tight')


objective_values = (2 * points[:,0] * points[:,1]) / (points[:,0] + points[:,1])

img = objective_values.reshape((100,100)) # rows are precisions and cols are recalls
f = plt.figure()
plt.imshow(img, origin='lower')
plt.xlabel('Recall', fontsize=28)
plt.ylabel('Precision', fontsize=28)
plt.xticks(np.linspace(0, 100, 6))
plt.yticks(np.linspace(0, 100, 6))
plt.colorbar()
plt.title('F-Score', fontsize=14)
plt.savefig('tmp/block10_precision_recall_f1.png', dpi=120, bbox_inches='tight')


#
# Accuracy @ different thresholds
#
y_actual = np.array([0, 1, 1, 0, 1, 1, 0]) # N
y_pred = np.array([0.06, 0.92, 0.86, 0.03, 0.40, 0.70, 0.23]) # N
thresholds = np.linspace(0, 1, 9)
hard_pred = y_pred[:, None] > thresholds[None, :] # Nx1 > 1xT = NxT
acc = np.mean(hard_pred == y_actual[:, None], axis=0) # T 

f, ax = plt.subplots(1,1,figsize=(5,2.5))
ax.plot(thresholds, acc, linewidth=2, marker='o', markersize=5)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.set_xlabel('Threshold', fontsize=18)
ax.set_ylabel('Accuracy', fontsize=18)
ax.grid(True, linestyle='--')
f.savefig('tmp/block10_accuracy_vs_threshold.svg', bbox_inches='tight')

#
# ROC-Demo
#
y_actual = np.array([0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0]) # N
y_pred = np.array([0.06, 0.92, 0.86, 0.03, 0.40, 0.70, 0.23, 0.4, 0.2, 0.8, 0.9, 0.65, 0.75, 0.4]) # N

fpr, tpr, _ = sklearn.metrics.roc_curve(y_actual, y_pred)

f, ax = plt.subplots(1,1,figsize=(5,5))
ax.step(fpr, tpr, linewidth=2, linestyle='--', marker='o', markersize=5)
ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.set_xlabel('FPR', fontsize=18)
ax.set_ylabel('TPR', fontsize=18)
ax.plot(fpr, fpr, linestyle='dashed')
ax.fill_between(fpr, tpr, alpha=0.2, step='pre')
ax.grid(True, linestyle='--')

roc = sklearn.metrics.roc_auc_score(y_actual, y_pred)
ax.text(0.5, 0.2, "AUC-ROC: %0.2f" % roc, fontsize=16)
f.savefig('tmp/block10_roc.svg', bbox_inches='tight')

#
# AUC-ROC
#

f, ax = plt.subplots(1,1,figsize=(5,5))

precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_actual, y_pred)
ax.step(recall, precision, linewidth=2, linestyle='-', marker='o', markersize=5)
ax.text(0.3, 0.9, "AUC-PR: %0.2f" % (sklearn.metrics.average_precision_score(y_actual, y_pred)), fontsize=16)

ypred_null = np.mean(y_actual) * np.ones_like(y_actual)
precision, recall, _ = sklearn.metrics.precision_recall_curve(y_actual, ypred_null)
ax.step(recall, precision, linewidth=2, linestyle='--', marker='o', markersize=5, where='post')
ax.text(0.3, 0.55, "AUC-PR (Null): %0.2f" % (sklearn.metrics.average_precision_score(y_actual, ypred_null)), fontsize=16)

ax.xaxis.set_tick_params(labelsize=16)
ax.yaxis.set_tick_params(labelsize=16)
ax.set_xlabel('Recall', fontsize=18)
ax.set_ylabel('Precision', fontsize=18)
ax.grid(True, linestyle='--')

ax.set_ylim([0, 1])
ax.set_xlim([0, 1])
f.savefig('tmp/block10_pr.svg', bbox_inches='tight')