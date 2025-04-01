import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Generate synthetic data (e.g., exam scores)
np.random.seed(42)
data = np.random.normal(loc=70, scale=10, size=100)  # Mean=70, Std=10, 100 samples

# Number of bootstrap samples
num_bootstrap_samples = 1000
bootstrap_means = []

# Perform bootstrap resampling
for _ in range(num_bootstrap_samples):
    sample = np.random.choice(data, size=len(data), replace=True)  # Resampling with replacement
    bootstrap_means.append(np.mean(sample))  # Compute mean of each sample

# Calculate confidence intervals (e.g., 95%)
lower_bound = np.percentile(bootstrap_means, 2.5)
upper_bound = np.percentile(bootstrap_means, 97.5)

# Plot results
sns.histplot(bootstrap_means, kde=True, bins=30, color="blue")
plt.axvline(lower_bound, color='red', linestyle='dashed', label=f'95% CI: {lower_bound:.2f}')
plt.axvline(upper_bound, color='red', linestyle='dashed', label=f'95% CI: {upper_bound:.2f}')
plt.axvline(np.mean(data), color='green', linestyle='solid', label=f'Original Mean: {np.mean(data):.2f}')
plt.xlabel("Mean Values from Bootstrap Samples")
plt.ylabel("Frequency")
plt.title("Bootstrap Resampling Distribution")
plt.legend()
plt.show()

# Print results
print(f"Original Sample Mean: {np.mean(data):.2f}")
print(f"95% Confidence Interval for the Mean: [{lower_bound:.2f}, {upper_bound:.2f}]")
