import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

a = [0.8323, 0.9082, 0.9129, 0.9083, 0.9032]
b = [0.8286, 0.9031, 0.9113, 0.9128, 0.9054]
c = [0.8401, 0.9196, 0.9285, 0.9111, 0.9109]
d = [0.8354, 0.9107, 0.9118, 0.9066, 0.9037]
e = [0.8439, 0.9121, 0.9186, 0.9048, 0.8979]

a1 = [0.8323, 0.8286, 0.8401, 0.8354, 0.8439]
a2 = [0.9082, 0.9031, 0.9196, 0.9107, 0.9121]
a3 = [0.9129, 0.9113, 0.9285, 0.9118, 0.9186]
a4 = [0.9083, 0.9128, 0.9111, 0.9066, 0.9048]
a5 = [0.9032, 0.9054, 0.9109, 0.9037, 0.8979]

plt.figure()
plt.plot(a1, "g-", label='Raw-DTC')
plt.plot(a2, "r-.", label="1-layer LSTM-DTC")
plt.plot(a3, "r-.", label="2-layer LSTM-DTC")
plt.plot(a4, "r-.", label="3-layer LSTM-DTC")
plt.plot(a5, "r-.", label="4-layer LSTM-DTC")

# plt.axis([10, 50, 100, 150, 200])
plt.xlabel("网络时延/ms")
plt.ylabel("v")
plt.title("a simple example")

plt.grid(True)
plt.legend()
plt.show()

n_bins = 10
x = np.random.randn(1000, 3)

fig, axes = plt.subplots(nrows=2, ncols=2)
ax0, ax1, ax2, ax3 = axes.flatten()

x = [a, b, c]
colors = ['red', 'tan', 'lime']
ax0.hist(x, n_bins, normed=1, histtype='bar', color=colors, label=colors)
ax0.legend(prop={'size': 10})
ax0.set_title('bars with legend')

ax1.hist(x, n_bins, normed=1, histtype='bar', stacked=True)
ax1.set_title('stacked bar')

ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
ax2.set_title('stack step (unfilled)')

# Make a multiple-histogram of data-sets with different length.
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]

x_multi = [a, b, c]

ax3.hist(x_multi, n_bins, histtype='bar')
ax3.set_title('different sample sizes')

fig.tight_layout()
plt.show()
