#!/usr/bin/env python
# coding: utf-8




import os
import numpy as np
import matplotlib.pyplot as plt




plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'





# Show an image and save it in the plot directory.
def ImageShow(image, fname):
    plt.title(fname)
    plt.imshow(image)
    plt.savefig(fname)
    plt.close()
    return




def CostsPlot(costs:list, learning_rate:int, name:str, out_dir:str):
    # Let's also plot the cost function and the gradients Plot learning curve (with costs)
    plt.title(name)
    plt.plot(costs)

    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))

    plt.savefig(os.path.join(out_dir, name))
    plt.show()
    plt.close()
    return


def LearningRatesCostsPlot(models:dict, learning_rates:np.array, name, PLOTS_DIR):
    plt.title(name)
    for i in learning_rates:
        plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

    plt.ylabel('cost')
    plt.xlabel('iterations (hundreds)')
    legend = plt.legend(loc='upper center', shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')

    plt.savefig(os.path.join(PLOTS_DIR, name))
    plt.show()
    plt.close()
    return




#####################################################
#                                                   #
#                   Plots                           #
#                                                   #
#####################################################
# Visualize 2d grid the dataset using matplotlib.
"""
X_0 - points horizontal coodinate
X_1 - points vertical coordinate.
Y_c - points Dot color
scatter - size.
"""
def plot_dataset(X_h, X_v, Y_c, scatter, path:str, title:str):
    plt.title(title)
    plt.scatter(X_h, X_v, c=Y_c, s=scatter, cmap=plt.cm.Spectral)

    plt.savefig(os.path.join(path,title+'.jpg'))
    plt.show()
    plt.close()
    return

def plot_decision_boundary(model, X, y, path:str, title:str):
    plt.title(title)

    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01

    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y[0, :], cmap=plt.cm.Spectral)

    plt.savefig(os.path.join(path,title+'.jpg'))
    plt.show()
    plt.close()
    return




def print_mislabeled_images(classes, X, y, p):
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    plt.rcParams['figure.figsize'] = (40.0, 40.0)  # set default size of plots
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]

        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:, index].reshape(64, 64, 3), interpolation='nearest')
        plt.axis('off')
        plt.title( "Prediction: " + classes[int(p[0, index])].decode("utf-8") + " \n Class: " + classes[y[0, index]].decode("utf-8"))
    return
