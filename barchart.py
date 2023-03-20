from matplotlib import pyplot as plt

from resnet50 import decoded_predictions, image

# Create a horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Define the color scheme
colors = ['#90caf9', '#64b5f6', '#42a5f5', '#2196f3', '#1e88e5', '#1976d2', '#1565c0', '#0d47a1', '#082a54']

# Extract the class names and probabilities from the decoded predictions
class_names = [class_name for _, class_name, _ in decoded_predictions]
probabilities = [probability for _, _, probability in decoded_predictions]

# Reverse the order of the data to plot the most probable class first
# class_names.reverse()
# probabilities.reverse()

# Plot the vertical bars with shaded colors
ax.bar(class_names, probabilities, align='center', color=colors)
# Rotate the x-axis labels to avoid overlap
plt.xticks(rotation=15)

# Add informative and concise labels and titles
ax.set_ylabel('Probability', fontsize=16, fontweight='bold', labelpad=10)
ax.set_xlabel('Class Name', fontsize=16, fontweight='bold', labelpad=10)
ax.set_title('Top 10 Predictions', fontsize=20, fontweight='bold', pad=20)
# Increase the font size of the labels and titles
ax.tick_params(axis='both', which='major', labelsize=10)
ax.tick_params(axis='x', which='major', labelsize=8)
ax.tick_params(axis='y', which='major', labelsize=8)
# Remove any unnecessary gridlines or borders
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
# Increase the font size of the class names
# Adjust bottom margin
plt.subplots_adjust(right=0.95, bottom=0.2)
for i, (_, class_name, probability) in enumerate(decoded_predictions):
    ax.text(i - 0.2, probability + 0.005, f"{round(probability * 100, 2)}%", fontsize=6, color='white',
            fontweight='bold', backgroundcolor='black', zorder=10)
# Show the plot
plt.show()
# Show the image
plt.imshow(image)
# Show the plot
plt.show()
