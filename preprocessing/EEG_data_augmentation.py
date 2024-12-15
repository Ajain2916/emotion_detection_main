import numpy as np
import pandas as pd
import os

def augment_data(matrix, method="noise", noise_level=0.01):
    """
    Applies data augmentation to the given EEG data matrix.

    Parameters:
        matrix (numpy.ndarray): EEG data to augment.
        method (str): Type of augmentation ('noise', 'scaling', etc.).
        noise_level (float): Standard deviation of noise to add (if method='noise').

    Returns:
        numpy.ndarray: Augmented data matrix.
    """
    if method == "noise":
        augmented_matrix = matrix + np.random.normal(0, noise_level, matrix.shape)
    elif method == "scaling":
        scale_factor = np.random.uniform(0.9, 1.1)
        augmented_matrix = matrix * scale_factor
    elif method == "permutation":
        augmented_matrix = np.random.permutation(matrix)
    elif method == "time_shift":
        shift = np.random.randint(1, matrix.shape[1])
        augmented_matrix = np.roll(matrix, shift, axis=1)
    else:
        raise ValueError("Unsupported augmentation method.")

    return augmented_matrix

def apply_augmentation(dataset, method="noise", noise_level=0.01):
    """
    Applies data augmentation to an entire dataset.

    Parameters:
        dataset (numpy.ndarray): EEG data matrix to augment.
        method (str): Type of augmentation ('noise', 'scaling', etc.).
        noise_level (float): Standard deviation of noise to add (if method='noise').

    Returns:
        numpy.ndarray: Augmented EEG data matrix.
    """
    augmented_dataset = augment_data(dataset, method, noise_level)
    return augmented_dataset

if __name__ == "__main__":
    # Specify the path to your data file
     data_file_path ="D:/Down-loads/emotion_detection/2 electrode data (self made) with initial + 1 more .csv"
    
try:
        # Read the data from Excel into a Pandas DataFrame
        eeg_data = pd.read_excel(data_file_path, engine="openpyxl", header=None)  # Adjust 'header=None' if the file has a header row
        eeg_matrix = eeg_data.to_numpy()
        print(f"Data loaded successfully from {data_file_path} with shape {eeg_matrix.shape}")

        # Apply noise augmentation
        augmented_data = apply_augmentation(eeg_matrix, method="noise", noise_level=0.05)
        print(f"Augmentation complete. Augmented data shape: {augmented_data.shape}")

        # Save augmented data back to an Excel file
        augmented_file_name = "augmented_data.xlsx"
        augmented_df = pd.DataFrame(augmented_data)
        augmented_df.to_excel(augmented_file_name, index=False, header=False, engine="openpyxl")
        print(f"Augmented data saved to {augmented_file_name}")
except FileNotFoundError:
        print(f"Error: File {data_file_path} not found.")
except Exception as e:
        print(f"An error occurred: {e}")
