import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV with 4 columns
df = pd.read_csv('/home/u869018/ActiveNeRF-GPU4EDU/Original-Test-GPUEDU Complete/ActiveNeRF/training_performance-loss.csv', header=0)

x = list(range(10, 210, 10))
assert len(df) == 20, "Expected 20 rows corresponding to x-axis values"

# --- Figure 1: Plot Training Performance PSNR ---
plt.figure(figsize=(10, 6))
plt.plot(x, df.iloc[:, 0], label='ActiveNeRF', marker='o')
plt.plot(x, df.iloc[:, 1], label='MU-NeRF', marker='s')
plt.xticks(range(0, 211, 10))
plt.yticks(range(18, 29, 1))
plt.xlim(0, 210)
plt.ylim(18, 28)
plt.xlabel('Iteration')
plt.ylabel('PSNR')
plt.title('PSNR: Training Performance (averaged over scenes)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Training-Performance-PSNR.png', dpi=300)
plt.close()

# --- Figure 2: Plot B Training Performance Loss ---
plt.figure(figsize=(10, 6))
plt.plot(x, df.iloc[:, 2], label='ActiveNeRF', marker='o')
plt.plot(x, df.iloc[:, 3], label='MU-NeRF', marker='s')
plt.xticks(range(0, 211, 10))
plt.yticks(np.arange(0, 8.01, 0.2))  # y-axis from 0 to 8 with 0.2 increments
plt.xlim(0, 210)
plt.ylim(0, 8)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss: Training Performance (averaged over scenes)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('Training-Performance-Loss.png', dpi=300)
plt.close()
