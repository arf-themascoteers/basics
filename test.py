import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from matplotlib import cm

x = np.array([-1,0,1])
y = np.array([-0.8,1,3.3])

def plot_original_points():
    plt.scatter(x,y)
    for i in range(3):
        plt.annotate(xy=(x[i], y[i]), text=f"   {x[i]},  {y[i]}")

m = 0.1
c = 2
y_pred = m * x + c
plt.plot(x, y_pred)

def get_loss(y, y_pred):
    return np.square(y_pred - y).sum()

def get_y_pred(m,c):
    y_preds = np.zeros((m.shape[0], m.shape[1], len(x)))
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            y_preds[i][j] = m[i][j] * x + c[i][j]
    return y_preds

def plot_loss():
    m_values = np.linspace(2-3,2+3, 500)
    c_values = np.linspace(1-3,1+3, 500)
    meshgrid_X, meshgrid_Y = np.meshgrid(m_values, c_values)
    print("c")
    print(c_values)
    print("meshgrid")
    print(meshgrid_Y)
    y_preds = get_y_pred(meshgrid_X, meshgrid_Y)
    losses = np.zeros((meshgrid_X.shape[0], meshgrid_X.shape[1]))
    for i in range(meshgrid_X.shape[0]):
        for j in range(meshgrid_X.shape[1]):
            losses[i][j] = get_loss(y, y_preds[i][j])
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(meshgrid_X, meshgrid_Y, losses, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel("m")
    ax.set_ylabel("c")
    ax.set_zlabel("loss")
    ax.xaxis.label.set_color('red')
    ax.yaxis.label.set_color('red')
    ax.zaxis.label.set_color('red')
    plt.show()
    return ax

plot_loss()