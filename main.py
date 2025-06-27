import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from sklearn.cluster import KMeans

# Declaring a Global variable for image path
image_path = None


# Creating a Function to Upload Image
def upload_image():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if image_path:
        display_image(image_path)


# Creating a Function to Display Image
def display_image(img_path):
    image = Image.open(img_path)
    image = image.resize((300, 300), Image.LANCZOS)  # Resize for GUI
    img = ImageTk.PhotoImage(image)
    panel.configure(image=img, text="")  # Clear the "No Image Uploaded" text
    panel.image = img


# Creating a Function to Process Image and Analyze Disaster
def analyze_disaster():
    global image_path
    if not image_path:
        messagebox.showwarning("Warning", "Please upload an image first!")
        return

    try:
        # Using OpenCv to load Image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Convert image to feature vectors
        pixels = image.reshape((-1, 3))
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(pixels)
        labels = kmeans.labels_

        # Identifying cluster with most pixels
        major_cluster_label = np.argmax(np.bincount(labels))

        # Get all pixels belonging to the major cluster
        major_cluster_pixels = pixels[labels == major_cluster_label]

        # Print the 2D RGB array of the major cluster
        print("Major Cluster Label:", major_cluster_label)
        print("Total Pixels in Major Cluster:", len(major_cluster_pixels))
        print("RGB Values (2D array) of Major Cluster:")
        print(major_cluster_pixels)

        # Find the major cluster label
        major_cluster_label = np.argmax(np.bincount(labels))

        # Create mapped labels: 1 for major, 2 & 3 for others
        mapped_labels = np.zeros_like(labels)
        label_mapping = {}
        current_label = 2

        for i in range(kmeans.n_clusters):
            if i == major_cluster_label:
                label_mapping[i] = 1
            else:
                label_mapping[i] = current_label
                current_label += 1

        # Apply mapping nmethod
        for i in range(len(labels)):
            mapped_labels[i] = label_mapping[labels[i]]

        # Scatter plot with mapped cluster numbers as colors
        # Visualize clusters in RGB space with mapped labels
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        # Extract RGB channels
        r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]

        # Assign colors to each label using cluster centers
        centers = kmeans.cluster_centers_.astype(int)
        cluster_colors = [centers[label] / 255 for label in labels]

        # Scatter plot of all pixels with their cluster color
        ax.scatter(r, g, b, color=cluster_colors, s=1, alpha=0.6)

        # Determine dominant color label (R, G, or B) for each cluster center
        color_labels = []
        for center in centers:
            dominant_index = np.argmax(center)  # 0=R, 1=G, 2=B
            label = ['Red', 'Green', 'Blue'][dominant_index]
            color_labels.append(label)

        # Plot cluster centers with color and label
        for i, center in enumerate(centers):
            ax.scatter(center[0], center[1], center[2], color=center / 255, s=200, marker='X', edgecolor='black')
            ax.text(center[0], center[1], center[2], color_labels[i], fontsize=10, fontweight='bold')

        # Axis labels
        ax.set_xlabel("Red")
        ax.set_ylabel("Green")
        ax.set_zlabel("Blue")
        ax.set_title("KMeans Clusters in RGB Space with Color Mapping")
        plt.show()

        # Compute Disaster Severity
        severity = np.sum(labels == np.argmax(np.bincount(labels))) / len(labels)
        demand = severity * 100  # Total relief demand

        # Set priority-based minimum ratios
        if severity < 0.4:  # Low severity
            min_food = 0.5
            min_water = 0.3
            min_medicine = 0.2
        elif severity < 0.7:  # Medium
            min_food = 0.35
            min_water = 0.35
            min_medicine = 0.3
        else:  # High severity
            min_food = 0.25
            min_water = 0.35
            min_medicine = 0.4

        # LP: Minimize cost
        c = [50, 80, 100]
        A_eq = [[1, 1, 1]]
        b_eq = [demand]

        # Setting bounds using dynamic ratios
        bounds = [
            (min_food * demand, None),  # Food
            (min_water * demand, None),  # Water
            (min_medicine * demand, None)  # Medicine
        ]

        res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

        if res.success:
            food, water, medicine = res.x
            severity_text = f"Disaster Severity: {severity:.2f}\n"
            severity_text += f"Relief Allocation:\n Food: {food:.1f} units\n Water: {water:.1f} units\n Medicine: {medicine:.1f} units"
        else:
            severity_text = "Optimization failed. Try a different image."

        result_label.configure(text=severity_text)

    except Exception as e:
        messagebox.showerror("Error", f"Error processing image: {str(e)}")


# Creating a GUI windo to display the output
ctk.set_appearance_mode("dark")  # Light or Dark Mode
ctk.set_default_color_theme("blue")  # Theme color

root = ctk.CTk()
root.title(" Disaster Management System")
root.geometry("600x650")

# Title Label
title_label = ctk.CTkLabel(root, text=" Disaster Relief System", font=("Arial", 20, "bold"))
title_label.pack(pady=10)

# Upload Button
upload_btn = ctk.CTkButton(root, text=" Upload Image", command=upload_image, font=("Arial", 15))
upload_btn.pack(pady=10)

# Image Display Panel
panel = ctk.CTkLabel(root, text="No Image Uploaded", width=300, height=300, fg_color=("gray75", "gray25"))
panel.pack(pady=10)

# Analyze Button
analyze_btn = ctk.CTkButton(root, text=" Analyze & Plan Relief", command=analyze_disaster, font=("Arial", 15))
analyze_btn.pack(pady=10)

# Result Display
result_label = ctk.CTkLabel(root, text="", font=("Arial", 15))
result_label.pack(pady=20)

# Run Tkinter GUI
root.mainloop()
