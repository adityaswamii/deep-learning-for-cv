import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from ipywidgets import interact, IntSlider, FloatSlider, Dropdown
from scipy import stats


# k-Nearest Neighbor Classifier

def l2_distance(a, b, axis = 1):
    return np.sqrt(np.sum((a - b) ** 2, axis=axis))

def l1_distance(a, b, axis = 1):
    return np.sum(np.abs(a - b), axis=axis)

class KNearestNeighbor:
    def __init__(self, k=5, dist='l2'):
        self.k = k
        self.dist = dist

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        if self.dist == 'l1': # use L1 distance
            distances = l1_distance(X[:, None, :], self.Xtr[None, :, :], axis=2)
        else:
            distances = l2_distance(X[:, None, :], self.Xtr[None, :, :], axis=2)
        idx = np.argsort(distances, axis=1)[:, :self.k] # Indices of k nearest neighbors
        nearest_labels = self.ytr[idx] # Labels of k nearest neighbors
        preds = np.array([stats.mode(row, keepdims=True).mode[0] for row in nearest_labels])
        return preds
    
    def neighbours(self, X):
        if self.dist == 'l1': # use L1 distance
            distances = l1_distance(X[:, None, :], self.Xtr[None, :, :], axis=2)
        else:
            distances = l2_distance(X[:, None, :], self.Xtr[None, :, :], axis=2)
        idx = np.argsort(distances, axis=1)[:, :self.k] # Indices of k nearest neighbors
        nearest_labels = idx # Indices of k nearest neighbors
        return nearest_labels





# 2D Decision Boundary Visualization

def make_2d_data(n_samples=200, noise=0.2):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=0)
    return X, y

def plot_knn_decision_boundary(samples=200, noise=0.2, k=3, dist='l2'):
    X, y = make_2d_data(n_samples=samples, noise=noise)
    knn = KNearestNeighbor(k=k, dist=dist)
    knn.train(X, y)

    # Create a meshgrid
    h = .02
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Predict over grid
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = knn.predict(grid_points)
    Z = Z.reshape(xx.shape)

    # Plot
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4, picker=True)

    sc = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm,
                    edgecolors='k', picker=True)

    # empty plot for highlighting neighbours
    neigh_plot, = ax.plot([], [], 'o', 
                          markerfacecolor='none', 
                          markeredgecolor='yellow', 
                          markersize=12, linewidth=2)

    clicked_plot, = ax.plot([], [], 'x', color='black', markersize=2)

    ax.set_title(f'2D dataset — k={k}, distance={dist}, noise={noise}')
    ax.set_xlabel('X1')
    ax.set_ylabel('X2')

    def on_pick(event):
        idx = event.ind[0]          # clicked point index
        clicked_point = X[idx]      # coordinates

        # find k nearest neighbours **using your existing KNN**
        neighbor_idx = knn.neighbours(clicked_point.reshape(1, -1))
        neighbor_idx = neighbor_idx.flatten()

        # update neighbour highlight
        neigh_plot.set_data(X[neighbor_idx, 0], X[neighbor_idx, 1])

        # update clicked point marker
        clicked_plot.set_data([clicked_point[0]], [clicked_point[1]])

        fig.canvas.draw_idle()

        print("Clicked:", idx, "Neighbours:", neighbor_idx)

    fig.canvas.mpl_connect('pick_event', on_pick)
    # fig.canvas.mpl_connect('click_event', on_pick) 

    plt.show()
    plt.close()

interact(
    plot_knn_decision_boundary,
    samples=IntSlider(min=100, max=1000, step=100, value=200),
    noise=FloatSlider(min=0, max=1, step=0.05, value=0.2),
    k=IntSlider(min=1, max=15, step=1, value=3),
    dist=Dropdown(options=['l2', 'l1'], value='l2')
)
