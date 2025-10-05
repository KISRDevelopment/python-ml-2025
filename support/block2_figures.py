import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt 

df = pd.read_json("data/cars.json")
cols = ['Cylinders', 'Displacement', 'Horsepower', 'Weight_in_lbs', 'Acceleration']

#
# plot all predictors
#
fig, axes = plt.subplots(1, 5, figsize=(25, 4), sharey=True)
axes = axes.flatten()
for i in range(len(cols)):
    ax = axes[i]
    col = cols[i]
    sns.scatterplot(ax=ax, 
                    data=df, 
                    y='Miles_per_Gallon', 
                    x=col,
                    hue='Origin')
    ax.xaxis.set_tick_params(labelsize=14)  
    ax.yaxis.set_tick_params(labelsize=14)  
    ax.set_ylabel('Miles per Gallon', fontsize=16)
    ax.set_xlabel(col, fontsize=16)
    ax.grid(True, 'both')
#fig.delaxes(axes[-1])
fig.savefig('tmp/cars.svg', bbox_inches='tight')

#
# plot just HP vs MPG
#
fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharey=True)
sns.scatterplot(ax=ax, 
                data=df, 
                y='Miles_per_Gallon', 
                x='Horsepower',
                edgecolor='black')
ax.xaxis.set_tick_params(labelsize=14)  
ax.yaxis.set_tick_params(labelsize=14)  
ax.set_ylabel('Miles per Gallon', fontsize=16)
ax.set_xlabel('Horsepower', fontsize=16)

ax.set_ylim([np.min(df['Miles_per_Gallon']), np.max(df['Miles_per_Gallon'])])
ax.grid(True, 'both')
fig.savefig('tmp/cars_univariate.svg', bbox_inches='tight')

#
# plot with some lines
#
fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharey=True)
sns.scatterplot(ax=ax, 
                data=df, 
                y='Miles_per_Gallon', 
                x='Horsepower',
                edgecolor='black')
ax.xaxis.set_tick_params(labelsize=14)  
ax.yaxis.set_tick_params(labelsize=14)  
ax.set_ylabel('Miles per Gallon', fontsize=16)
ax.set_xlabel('Horsepower', fontsize=16)
ax.grid(True, 'both')

params = [(30, 0), (40, -0.15), (10, 0.03)]
colors = ['red', 'magenta', 'purple']
max_hp = np.max(df['Horsepower'])
min_hp = np.min(df['Horsepower'])
xs = np.linspace(min_hp, max_hp, 100)
i = 0
for b0, b1 in params:
    ax.plot(xs, b0 + b1*xs, linestyle='--', linewidth=2, color=colors[i])
    i += 1
ax.set_ylim([np.min(df['Miles_per_Gallon']), np.max(df['Miles_per_Gallon'])])

fig.savefig('tmp/cars_univariate_lines.svg', bbox_inches='tight')

#
# plot HP vs MPG with some MSE lines
#
fig, ax = plt.subplots(1, 1, figsize=(5, 4), sharey=True)
sns.scatterplot(ax=ax, 
                data=df, 
                y='Miles_per_Gallon', 
                x='Horsepower',
                edgecolor='black')
ax.xaxis.set_tick_params(labelsize=14)  
ax.yaxis.set_tick_params(labelsize=14)  
ax.set_ylabel('Miles per Gallon', fontsize=16)
ax.set_xlabel('Horsepower', fontsize=16)

x = df['Horsepower']
y = df['Miles_per_Gallon']
b0, b1 = 40, -0.15
yhat = b0 + b1 * x

idx = np.random.choice(x.shape[0], replace=False, size=10)
for i in idx:
    ax.plot([x[i], x[i]], [y[i], yhat[i]], color='magenta', linewidth=2)

ax.plot(xs, b0 + b1*xs, linestyle='--', linewidth=2, color='purple')
ax.set_ylim([np.min(df['Miles_per_Gallon']), np.max(df['Miles_per_Gallon'])])
ax.grid(True, 'both')
fig.savefig('tmp/cars_univariate_diff.svg', bbox_inches='tight')

#
# plot loss surface for b0
#
b0s = np.linspace(0, 100, 1000) 
b1 = -0.15 # R
Yhat = b0s[:, None] + b1 * x.to_numpy()[None, :] # Rx1 * 1xN = RxN
mse = np.nanmean(np.square(Yhat - y.to_numpy()[None, :]), axis=1) # R 
ix_min = np.argmin(mse)
min_b0 = b0s[ix_min]
min_mse = mse[ix_min]

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.plot(b0s, mse, linestyle='--')
ax.plot(min_b0, min_mse, markersize=10, marker='o')
ax.xaxis.set_tick_params(labelsize=14)  
ax.yaxis.set_tick_params(labelsize=14)  
ax.set_ylabel('MSE', fontsize=16)
ax.set_xlabel('$b_0$', fontsize=16)
fig.savefig('tmp/cars_loss.svg', bbox_inches='tight')

#
# plot loss surface for both b0 and b1
#
x = x.to_numpy()
y = y.to_numpy()
b0s = np.linspace(-50, 50, 100) # R0
b1s = np.linspace(-1, 1, 100) # R1
B0, B1 = np.meshgrid(b0s, b1s) # R0 x R1
Yhat = B0[:,:,None] + B1[:,:, None] * x[None,None, :] # R0 x R1 x 1 * 1x1xN = R0xR1xN
mse = np.nanmean(np.square(Yhat - y[None,None, :]), axis=2) # R0xR1

fig, ax = plt.subplots(1, 1, figsize=(5,5))
ax.pcolor(B0, B1, mse)
ax.set_ylabel('$b_1$', fontsize=16)
ax.set_xlabel('$b_0$', fontsize=16)

ix_min_b1, ix_min_b0 = np.unravel_index(mse.argmin(), mse.shape)
min_b0 = b0s[ix_min_b0]
min_b1 = b1s[ix_min_b1]

ax.plot(min_b0, min_b1, markersize=10, marker='o', color='orange')
fig.savefig('tmp/cars_loss_2d.png', bbox_inches='tight')
