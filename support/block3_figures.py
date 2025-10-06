import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

df = pd.read_json("data/cars.json")
cols = ['Cylinders', 'Displacement', 'Horsepower', 'Weight_in_lbs', 'Acceleration']

#
# plot loss surface for both b0 and b1
#

x = df['Horsepower']
y = df['Miles_per_Gallon']
ix_excluded = np.isnan(x) | np.isnan(y)

# exclude examples with no horsepower or mpg
x = x[~ix_excluded]
y = y[~ix_excluded]

x = x.to_numpy()
y = y.to_numpy()

z = (x - np.mean(x)) / np.std(x, ddof=1)


b0s = np.linspace(-50, 50, 100) # R0
b1s = np.linspace(-15, 0, 100) # R1

B0, B1 = np.meshgrid(b0s, b1s) # R0 x R1
Yhat = B0[:,:,None] + B1[:,:, None] * z[None,None, :] # R0 x R1 x 1 * 1x1xN = R0xR1xN
mse = np.mean(np.square(Yhat - y[None,None, :]), axis=2) # R0xR1

fig, axes = plt.subplots(1, 3, figsize=(15,5))

N = 100

etas = [0.001, 0.1, 1]

for j, eta in enumerate(etas):

    beta0, beta1 = [-40, -10]

    ax = axes[j]
    ax.pcolor(B0, B1, mse)
    ax.set_ylabel('$b_1$', fontsize=16)
    ax.set_xlabel('$b_0$', fontsize=16)

    steps = []
    steps_mse = []
    for i in range(N):
        
        # compute model predictions
        yhat = z * beta1 + beta0
        
        # compute gradient at those predictions
        beta0_grad = np.mean(2 * (yhat - y))
        beta1_grad = np.mean(2 * (yhat - y) * z)
        
        # update
        beta0 = beta0 - eta * beta0_grad
        beta1 = beta1 - eta * beta1_grad
        
        # track progress
        steps.append([beta0, beta1])
        yhat = z * beta1 + beta0
        steps_mse.append(np.mean(np.square(yhat - y)))
        print(steps_mse[-1])
    steps = np.array(steps)
    print("Best mse: %0.2f" % steps_mse[-1])
    print(steps[-1])
    ax.plot(steps[:,0], steps[:,1], color='magenta', linewidth=2)
    ix_min_b1, ix_min_b0 = np.unravel_index(mse.argmin(), mse.shape)
    min_b0 = b0s[ix_min_b0]
    min_b1 = b1s[ix_min_b1]

    ax.plot(min_b0, min_b1, markersize=10, marker='o', color='orange')
    ax.set_xlim([b0s[0], b0s[-1]])
    ax.set_ylim([b1s[0], b1s[-1]])
    ax.set_title(r'$\eta = %0.3f$' % eta)
fig.savefig('tmp/cars_loss_2d_gd.png', bbox_inches='tight')
