# plot_speedup.py
import matplotlib.pyplot as plt

draft_lengths = [2, 4, 6, 8, 10]
speedups = [1.01, 1.12, 1.14, 0.91, 1.00]
accepted = [2.04, 4.08, 6.12, 8.16, 10.20]

fig, ax1 = plt.subplots()
ax1.plot(draft_lengths, speedups, 'b-o', label='Speedup')
ax1.set_xlabel('Draft Length (d)')
ax1.set_ylabel('Speedup', color='b')
ax1.axhline(y=1.0, color='gray', linestyle='--', label='Baseline (1.0x)')

ax2 = ax1.twinx()
ax2.plot(draft_lengths, accepted, 'r-s', label='Accepted/Step')
ax2.set_ylabel('Accepted Tokens per Step', color='r')
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0)

plt.title('NextLat Speculative Decoding Performance')
plt.grid(True, alpha=0.3)
plt.show()