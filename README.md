# KNN
Understanding KNN's 
# K-Nearest Neighbors (KNN) Classifier on Iris Dataset

This project implements a K-Nearest Neighbors (KNN) classifier from scratch and uses it to classify the famous Iris dataset from the **scikit-learn** library. The code demonstrates data loading, splitting into training and testing sets, model fitting, and calculating accuracy on the test data.

### Overview
The KNN algorithm is a simple, non-parametric method used for classification and regression. In this project, we apply the KNN algorithm to classify the species of iris plants based on their physical characteristics like petal and sepal length/width.

### Libraries Used:
- **TensorFlow**: Imported but not actively used in the current version of the code.
- **Pandas**: For data manipulation.
- **NumPy**: For numerical computations.
- **Scikit-learn**: For loading datasets and splitting the data.
- **Matplotlib**: For data visualization (optional).
- **Custom KNN Implementation**: A custom `KNN` class (presumably from a separate file named `knn.py`) is used to implement the KNN algorithm from scratch.

### Workflow:
1. **Data Loading**:  
   The Iris dataset is loaded using the `datasets` module from **scikit-learn**. The dataset contains three classes of iris species, and each sample has four features (sepal and petal length/width).

2. **Data Splitting**:  
   The dataset is split into training (80%) and testing (20%) sets using `train_test_split`.

3. **Visualization (Optional)**:  
   A scatter plot is generated to visualize the distribution of the data points based on two features: petal length and petal width, with colors indicating the species.

4. **KNN Classifier**:  
   A custom KNN model is initialized with `k=3`, meaning it considers the 3 nearest neighbors for classification. The model is trained on the training set using the `fit` method.

5. **Prediction and Accuracy**:  
   Predictions are made on the test set using the `predict` method, and the accuracy is calculated by comparing the predictions with the true labels.

### Example Output:
The output will display the accuracy of the model on the test dataset. The accuracy is computed as:
```
accuracy = (Number of correct predictions) / (Total test samples)
```

### Visualization:
You can uncomment the provided plotting code to visualize the data distribution using `matplotlib`. The colors will represent different species in the dataset.

### How to Run:
1. Install the required libraries:  
   ```
   pip install numpy pandas scikit-learn matplotlib tensorflow
   ```

2. Ensure you have the custom `knn.py` file in the same directory, which contains the implementation of the KNN algorithm.

3. Run the script using Python:
   ```
   python script_name.py
   ```

---
