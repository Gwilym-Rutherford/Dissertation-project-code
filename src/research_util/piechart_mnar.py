import matplotlib.pyplot as plt

# Data from MNAR study table
labels = ["Withdrawn", "Uncontactable", "Excused"]
patients = [73, 116, 109]

# Create pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    patients,
    labels=labels,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("MNAR Study Patient Categories")
plt.axis("equal")  # Ensures pie chart is circular

plt.show()