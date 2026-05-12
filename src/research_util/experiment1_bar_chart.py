# code has been created by gemini from results i have found

import matplotlib.pyplot as plt
import numpy as np

# 1. Define the models and the extracted data (Visit Level - MAE)
models = ['Random Forest', 'LSTM', 'Transformer']

baseline = [6.54, 7.00, 6.61]        # Standard DMO
multi_visit = [6.31, 8.1, 6.44]      # DMO[1-3] / DMO[1-4]
delta_first = [4.71, 5.54, 4.57]    # Delta from First Visit
delta_prev = [4.39, 5.46, 4.48]     # Delta from Previous Visit

# 2. Set up the bar chart positioning
x = np.arange(len(models))  # The label locations
width = 0.2  # The width of the bars

# 3. Create the plot with presentation-friendly dimensions
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting each group of bars
# Baseline is grey to be muted, others are distinct colors to stand out
rects1 = ax.bar(x - 1.5*width, baseline, width, label='Baseline', color='#B0B0B0')
rects2 = ax.bar(x - 0.5*width, multi_visit, width, label='Visits [1-4]', color='#4A90E2')
rects3 = ax.bar(x + 0.5*width, delta_first, width, label='Δ First Visit', color='#50E3C2')
rects4 = ax.bar(x + 1.5*width, delta_prev, width, label='Δ Previous Visit', color='#F5A623')

# 4. Add labels, title, and custom x-axis tick labels
ax.set_ylabel('MAE - Lower is Better', fontsize=12)
# ax.set_title('Model Performance: Baseline vs Temporal Gait Features', fontsize=16, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)

# Make the legend clean and place it outside or in a clear spot
ax.legend(fontsize=11, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=4)

# 5. Optional: Add the exact data values on top of the bars
def autolabel(rects):
    """Attach a text label above each bar, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)

# Remove the top and right spines (borders) for a cleaner "modern" look
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout so the legend fits properly at the bottom
fig.tight_layout()

# 6. Save and display the plot
plt.savefig('baseline_vs_temporal.png', dpi=300, bbox_inches='tight') # High resolution for PowerPoint
plt.show()