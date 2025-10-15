import matplotlib.pyplot as plt 
import numpy as np 

fig, ax = plt.subplots(1, 1, figsize=(5,5))

xs = np.linspace(-5, 5, 100)
ys = 1/(1+np.exp(-xs))

ax.plot(xs, ys, linestyle='--', linewidth=5)
ax.xaxis.set_tick_params(labelsize=14)  
ax.yaxis.set_tick_params(labelsize=14)  
ax.set_ylabel('$P(y=1)$', fontsize=16)
ax.set_xlabel('$x$', fontsize=16)
ax.grid(True, linestyle='--')
fig.savefig('tmp/sigmoid.svg', bbox_inches='tight')


fig, ax = plt.subplots(1, 1, figsize=(5,5))

p = np.linspace(0,1,100)
ys0 = -np.log(1-p)
ys1 = -np.log(p)

ax.plot(p, ys0, linestyle='-', linewidth=5, label='$y_i = 0$')
ax.plot(p, ys1, linestyle='-', linewidth=5, label='$y_i = 1$')

ax.xaxis.set_tick_params(labelsize=14)  
ax.yaxis.set_tick_params(labelsize=14)  
ax.set_ylabel('Cross Entropy', fontsize=16)
ax.set_xlabel('$p_i$', fontsize=16)
ax.grid(True, linestyle='--')
ax.legend(fontsize=14, frameon=False)
fig.savefig('tmp/xentropy.svg', bbox_inches='tight')
