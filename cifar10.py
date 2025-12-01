import pickle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# --- Use KNN to implement image classification ---

def load_data(filepath='./cifar10.pkl'):
    """Load saved data"""
    print(f"Loading data from {filepath}...")
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                'dog', 'frog', 'horse', 'ship', 'truck']

data = load_data('./cifar10.pkl')

x_train = data['x_train']
x_val = data['x_val']
x_test = data['x_test']

y_train = data['y_train']
y_val = data['y_val']
y_test = data['y_test']

y_train = y_train.ravel()
y_val = y_val.ravel()
y_test = y_test.ravel()

# Flatten the images
X_train_flat = x_train.reshape(len(x_train), -1)
X_test_flat  = x_test.reshape(len(x_test), -1)

# Create KNN model
knn = KNeighborsClassifier(n_neighbors=5, n_jobs=-1)

# Train the model
knn.fit(X_train_flat, y_train)

# Predict on test set
y_pred = knn.predict(X_test_flat)

# Evaluate accuracy
acc = accuracy_score(y_test, y_pred)
print("KNN baseline accuracy =", acc)

# --- Visualize misclassified images ---

# Identify the indices where the true labels (y_test) do not match the predicted labels (y_pred).
misclassified_indices = np.where(y_test != y_pred)[0]
print(f"\nTotal misclassified images: {len(misclassified_indices)}")

# Determine the number of misclassified images to display. 
# It is set to the minimum of 25 or the total number of misclassified images found.
n_display = min(25, len(misclassified_indices))
sample_indices = np.random.choice(misclassified_indices, n_display, replace=False)

fig, axes = plt.subplots(int(np.ceil(n_display/5)), 5, figsize=(15, 3*int(np.ceil(n_display/5))))
axes = axes.flatten()

# Iterate through the selected sample indices to plot the corresponding images.
for i, idx in enumerate(sample_indices):
    img = x_test[idx]
    
    axes[i].imshow(img)
    axes[i].set_title(f"True: {class_names[y_test[idx]]}\nPred: {class_names[y_pred[idx]]}", 
                      color='red')
    axes[i].axis('off')

for j in range(n_display, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle(f"Selected Misclassified CIFAR-10 Images (KNN, k=5)\nTotal Misclassified: {len(misclassified_indices)}", fontsize=16)
fig.subplots_adjust(hspace=0.6)
plt.show()