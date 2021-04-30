from matplotlib import pyplot as plt
import numpy as np


def plot_table(val): ###val is a list,composed of  y matrix of several  method, using  append function
    for i in range(len(val)):
        colours = plt.cm.BuPu(np.array(val[i]) / 2)
        plt.subplot(2, 2, i + 1)
        tb = plt.table(cellText=np.array(val[i]), cellLoc='center', loc='center',
                       cellColours=colours, edges='closed')
        tc = tb.properties()['child_artists']
        for cell in tc:
            cell.set_height(1.0 / 5)
            cell.set_width(1.0 / 7)
        for key, cell in tb.get_celld().items():
            cell.set_linewidth(0)
        plt.xticks([])
        plt.axis('off')
        plt.yticks([])
    plt.show()
    plt.close()


def plot_its():
  return

