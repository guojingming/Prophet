import numpy as np
from matplotlib import pyplot as plt
import os


def draw_price(feature, output, label, marker=['*', '+'], colors=['orange', 'red', 'blue']):
    feature = feature.cpu().detach().numpy().reshape(-1)
    output = output.cpu().detach().numpy().reshape(-1)
    label = label.cpu().detach().numpy().reshape(-1)

    feature_count = 20
    label_count = 10

    # Draw input
    feature = feature[1:].reshape((feature_count, 3))
    x = np.arange(feature_count)
    plt.scatter(x, feature[:, 2].reshape(-1), marker=marker[0], c=colors[0])
    plt.plot(x, feature[:, 2].reshape(-1), color=colors[0])

    # Draw pred and gt
    output = output.reshape((label_count, 3))
    label = label.reshape((label_count, 3))

    x = np.arange(label_count)
    x += feature_count
    plt.scatter(x, output[:, 2].reshape(-1), marker=marker[0], c=colors[1])
    plt.plot(x, output[:, 2].reshape(-1), color=colors[1])
    plt.scatter(x, label[:, 2].reshape(-1), marker=marker[0], c=colors[2])
    plt.plot(x, label[:, 2].reshape(-1), color=colors[2])

    # Connect feature and label
    x = np.array([feature_count-1, feature_count])
    output_y = np.array([feature[-1, 2], output[0, 2]])
    label_y = np.array([feature[-1, 2], label[0, 2]])
    plt.plot(x, output_y, color=colors[1])
    plt.plot(x, label_y, color=colors[2])

    plt.xlabel('time')
    plt.ylabel('price')
    plt.grid()
    plt.show()


def save_draw_price(feature, output, label, marker=['*', '+'], colors=['orange', 'red', 'blue'],
                    save_path='./data/fig', save_name='default.png'):
    feature = feature.cpu().detach().numpy().reshape(-1)
    output = output.cpu().detach().numpy().reshape(-1)
    label = label.cpu().detach().numpy().reshape(-1)

    feature_count = 20
    label_count = 10

    # Draw input
    feature = feature[1:].reshape((feature_count, 3))
    x = np.arange(feature_count)
    plt.scatter(x, feature[:, 2].reshape(-1), marker=marker[0], c=colors[0])
    plt.plot(x, feature[:, 2].reshape(-1), color=colors[0])

    # Draw pred and gt
    output = output.reshape((label_count, 3))
    label = label.reshape((label_count, 3))

    x = np.arange(label_count)
    x += feature_count
    plt.scatter(x, output[:, 2].reshape(-1), marker=marker[0], c=colors[1])
    plt.plot(x, output[:, 2].reshape(-1), color=colors[1])
    plt.scatter(x, label[:, 2].reshape(-1), marker=marker[0], c=colors[2])
    plt.plot(x, label[:, 2].reshape(-1), color=colors[2])

    # Connect feature and label
    x = np.array([feature_count-1, feature_count])
    output_y = np.array([feature[-1, 2], output[0, 2]])
    label_y = np.array([feature[-1, 2], label[0, 2]])
    plt.plot(x, output_y, color=colors[1])
    plt.plot(x, label_y, color=colors[2])

    plt.xlabel('time')
    plt.ylabel('price')
    #plt.grid()
    #plt.show()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, save_name))
    plt.close()