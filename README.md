# ML-Classifer-SVM
This code demonstrates a machine learning pipeline for image classification using the Support Vector Machine (SVM) algorithm. It trains a model on a dataset of labeled images, extracts features from the images, and uses the trained model to predict labels for new test images. In this case we used the datasets to differentiate between dress shirts and t-shirts.

Setup and Dependencies:
- Python 3.x
- OpenCV (`cv2`)
- Pandas (`pandas`)
- NumPy (`numpy`)
- Matplotlib (`matplotlib`)
- Scikit-learn (`sklearn`)
- Scikit-image (`skimage`)

Usage:
1. Import the necessary libraries and modules. The datasets are not provided in this repository.

2. Load the training and test data from CSV files. Ensure that the data is properly formatted, where the training data is stored in `TrainData.csv`, the corresponding labels are stored in `TrainLabels.csv`, and the test data is stored in `TestData.csv`.

3. Preprocess the training data by reshaping it into a 3D array representing images. The original images are 28x28 grayscale, so the data is reshaped to (-1, 28, 28).

4. Display an example image from the training data to visualize the input.
  <img width="360" alt="dataset_visualization" src="https://github.com/hhumayune/ML-Classifer-SVM/assets/92355531/f5f956e3-d042-44ad-b3fb-f6b6e2596b67">

5. Defined a function `extract_features` to extract features from images. By default, the Histogram of Oriented Gradients (HOG) method is used. You can implement additional methods like edge extraction, color channel extraction, or midpoint extraction if needed.

6. Defined a function `extract_pixel_intensity` to extract features by flattening the image pixels.

7. Called the `extract_features` function to extract features from the training and test sets.

8. Set the hyperparameters for the SVM model, such as the regularization parameter `C` and the kernel type `kernel`.

9. Train the SVM model using the entire training dataset.

10. Save the trained model to a file named 'final_model.pkl' using the `pickle` module for future use.

11. Generate predictions for the test examples using the trained model.

12. Save the predictions to a CSV file named 'myPredictions.csv', which contains the predicted labels for the test examples. The files 'myPredictions.csv' and 'final_model.pkl' files are provided for reference

Note: You can modify the code to use different feature extraction methods or adjust the SVM hyperparameters to improve classification accuracy.
