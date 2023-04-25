import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from IPython.display import clear_output


def plot_dataset(samples, mean, covariance):
    ellipse_colors = np.array([u'#1f78c4', u'#ff801e', u'#2ca13c', u'#d62838', u'#9468cd', u'#8c575b', u'#e378d2', u'#7f808f', u'#bcbe32', u'#17bfdf'])
    n_std = 2

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    for mu, sigma, color in zip(mean, covariance, ellipse_colors):
        lambda_, v = np.linalg.eig(sigma)
        lambda_ = np.sqrt(lambda_)
        ellipse = Ellipse((mu[0], mu[1]), width=lambda_[0] * 2 * n_std, height=lambda_[1] * 2 * n_std,
                        angle=np.rad2deg(np.arctan2(*v[:,0][::-1])), edgecolor=color, lw=2)
        ellipse.set_facecolor('none')
        ax.add_patch(ellipse)

        plt.scatter(samples[0], samples[1], s=0.5, c='black')
    plt.axis('equal')
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_clusters(cluster, samples, mean, covariance, iter):

    # Plot
    ellipse_colors = np.array([u'#1f78c4', u'#ff801e', u'#2ca13c', u'#d62838', u'#9468cd', u'#8c575b', u'#e378d2', u'#7f808f', u'#bcbe32', u'#17bfdf'])
    n_std = 2

    point_colors = np.array([u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd', u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf'])
    sample_colors = point_colors[cluster]
    
    # clear_output(wait = True)
    clear_output(wait=True)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    plt.scatter(samples[0], samples[1], s=0.5, c=sample_colors)
    for mu, sigma, color in zip(mean, covariance, ellipse_colors):
        lambda_, v = np.linalg.eig(sigma)
        lambda_ = np.sqrt(lambda_)
        ellipse = Ellipse((mu[0], mu[1]), width=lambda_[0] * 2 * n_std, height=lambda_[1] * 2 * n_std,
                        angle=np.rad2deg(np.arctan2(*v[:,0][::-1])), edgecolor=color, lw=2)
        ellipse.set_facecolor('none')
        ax.add_patch(ellipse)
    title = 'Iteration: ' + str(iter + 1)
    plt.title(title)
    plt.axis('equal')
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_classification_init(samples_c1, mean_c1, covariance_c1, samples_c2, mean_c2, covariance_c2):
    n_std = 2

    # Show initial clusters for both classes
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    for mu_c1, sigma_c1, mu_c2, sigma_c2 in zip(mean_c1, covariance_c1, mean_c2, covariance_c2):
        lambda_c1, v_c1 = np.linalg.eig(sigma_c1)
        lambda_c1 = np.sqrt(lambda_c1)
        ellipse_c1 = Ellipse((mu_c1[0], mu_c1[1]), width=lambda_c1[0] * 2 * n_std, height=lambda_c1[1] * 2 * n_std,
                        angle=np.rad2deg(np.arctan2(*v_c1[:,0][::-1])), edgecolor='r', lw=2)
        ellipse_c1.set_facecolor('none')
        ax.add_patch(ellipse_c1)

        plt.scatter(samples_c1[0], samples_c1[1], s=0.5, c='b')

        lambda_c2, v_c2 = np.linalg.eig(sigma_c2)
        lambda_c2 = np.sqrt(lambda_c2)
        ellipse_c2 = Ellipse((mu_c2[0], mu_c2[1]), width=lambda_c2[0] * 2 * n_std, height=lambda_c2[1] * 2 * n_std,
                        angle=np.rad2deg(np.arctan2(*v_c2[:,0][::-1])), edgecolor='b', lw=2)
        ellipse_c2.set_facecolor('none')
        ax.add_patch(ellipse_c2)

        plt.scatter(samples_c2[0], samples_c2[1], s=0.5, c='r')
    plt.axis('equal')
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_classification_clusters(mean_c1, covariance_c1, samples_c1, mean_c2, covariance_c2, samples_c2, iter):
    n_std = 2

    # Plot
    clear_output(wait=True)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    
    plt.scatter(samples_c1[0], samples_c1[1], s=0.5, c='r')
    plt.scatter(samples_c2[0], samples_c2[1], s=0.5, c='b')
    for mu_c1, sigma_c1, mu_c2, sigma_c2 in zip(mean_c1, covariance_c1, mean_c2, covariance_c2):
        # Class 1
        lambda_c1, v_c1 = np.linalg.eig(sigma_c1)
        lambda_c1 = np.sqrt(lambda_c1)
        ellipse_c1 = Ellipse((mu_c1[0], mu_c1[1]), width=lambda_c1[0] * 2 * n_std, height=lambda_c1[1] * 2 * n_std,
                        angle=np.rad2deg(np.arctan2(*v_c1[:,0][::-1])), edgecolor='r', lw=2)
        ellipse_c1.set_facecolor('none')
        ax.add_patch(ellipse_c1)

        # Class 2
        lambda_c2, v_c2 = np.linalg.eig(sigma_c2)
        lambda_c2 = np.sqrt(lambda_c2)
        ellipse_c2 = Ellipse((mu_c2[0], mu_c2[1]), width=lambda_c2[0] * 2 * n_std, height=lambda_c2[1] * 2 * n_std,
                        angle=np.rad2deg(np.arctan2(*v_c2[:,0][::-1])), edgecolor='b', lw=2)
        ellipse_c2.set_facecolor('none')
        ax.add_patch(ellipse_c2)

    title = 'Iteration: ' + str(iter + 1)
    plt.title(title)
    plt.axis('equal')
    plt.show(block=False)
    plt.pause(1)
    plt.close()


def plot_classification(mean_c1, covariance_c1, mean_c2, covariance_c2, class_, samples_classification):
    n_std = 2

    clear_output(wait=True)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()

    idx = np.argwhere(class_ == 0)
    plt.scatter(samples_classification[0, idx], samples_classification[1, idx], c='r', s=4)
    idx = np.argwhere(class_ == 1)
    plt.scatter(samples_classification[0, idx], samples_classification[1, idx], c='b', s=4)

    for mu_c1, sigma_c1, mu_c2, sigma_c2 in zip(mean_c1, covariance_c1, mean_c2, covariance_c2):
        # Class 1
        lambda_c1, v_c1 = np.linalg.eig(sigma_c1)
        lambda_c1 = np.sqrt(lambda_c1)
        ellipse_c1 = Ellipse((mu_c1[0], mu_c1[1]), width=lambda_c1[0] * 2 * n_std, height=lambda_c1[1] * 2 * n_std,
                        angle=np.rad2deg(np.arctan2(*v_c1[:,0][::-1])), edgecolor='r', lw=2)
        ellipse_c1.set_facecolor('none')
        ax.add_patch(ellipse_c1)

        # Class 2
        lambda_c2, v_c2 = np.linalg.eig(sigma_c2)
        lambda_c2 = np.sqrt(lambda_c2)
        ellipse_c2 = Ellipse((mu_c2[0], mu_c2[1]), width=lambda_c2[0] * 2 * n_std, height=lambda_c2[1] * 2 * n_std,
                        angle=np.rad2deg(np.arctan2(*v_c2[:,0][::-1])), edgecolor='b', lw=2)
        ellipse_c2.set_facecolor('none')
        ax.add_patch(ellipse_c2)
    plt.axis('equal')
    plt.show(block=False)
    plt.pause(1)
    plt.close()