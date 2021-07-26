import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'black', 'tab:orange', 'lime', 'tab:brown', 'fuchsia', 'tab:gray', 'yellow', 'aqua', 'tab:blue', 'indigo', 'navy', 'lightcoral', 'darkolivegreen']
# colors = cm.rainbow(np.linspace(0, 1, 15))    
labels = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18']

X = np.load("output/features.npy") # (7680, 304) --> (7680*19, 16)

allX = None

for i in range(19):
    Xnew = X[:,(16*i):16*(i+1)]
    # print(len(Xnew[0,:]))
    # allX.append(Xnew, axis=0)
    if i == 0:
        allX = Xnew
    else:
        allX = np.vstack([allX, Xnew])

# X = np.array([[0, 0, 0, 0], [0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [6, 6, 6, 6]])
y = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18])
#y = ['a', 'b', 'c', 'd']
X_embedded = TSNE(n_components=3).fit_transform(allX)

print(X_embedded[:,0])
print(X_embedded.shape)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i, color in enumerate(colors):
    left = i*7680
    right = (i+1)*7680
    ax.scatter(X_embedded[left:right,0], X_embedded[left:right,1], X_embedded[left:right,2], zdir='z', c= color, label = labels[i])


plt.show()

X_embedded = TSNE(n_components=2).fit_transform(allX)

print(X_embedded[:,0])
print(X_embedded)

fig = plt.figure()
ax = fig.add_subplot(111)
for i, color in enumerate(colors):
    left = i*7680
    right = (i+1)*7680
    ax.scatter(X_embedded[left:right,0], X_embedded[left:right,1], c= color, label = labels[i])


plt.show()