import matplotlib.pyplot as plt


def triplot_2d(r, f):
    fig, ax = plt.subplots()
    plt.tight_layout(pad=0)
    ax.triplot(r.T[0], r.T[1], f)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()
