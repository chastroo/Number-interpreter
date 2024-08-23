Digit Recognition using Euclidean Distance
Overview
This Python script demonstrates a basic implementation of digit recognition using two approaches based on the Euclidean distance. It processes the digits dataset from scikit-learn, calculates average images for each digit, and then compares an input image against the dataset to identify the digit.

Requirements
To run this script, you will need the following Python packages:

cv2 (from OpenCV)
numpy
scikit-learn
pandas
Install these packages using pip if you haven't already:

bash
Copy code
pip install opencv-python numpy scikit-learn pandas
How the Code Works
1. Dataset Import and Average Calculation
The script starts by importing the digits dataset from scikit-learn.
It then calculates the average image for each digit (0-9) by averaging all the images corresponding to that digit in the dataset.
The averages are rounded and stored in a DataFrame, which is then exported to a CSV file (promedio.csv) for easy visualization.
2. Image Preprocessing
The script reads an external image file (img8.jpg), converts it to grayscale, and resizes it to 8x8 pixels to match the dimensions of the images in the digits dataset.
The pixel values are then normalized to a scale from 0 to 16, which corresponds to the pixel intensity scale used in the digits dataset.
3. Distance Calculation
Two functions are implemented:
distancia_euclidiana: Calculates the Euclidean distance between two matrices (images).
calcular_vecinos_cercanos: Finds the closest matches (neighbors) based on the calculated distances and identifies the most likely digit.
4. Digit Recognition
The script runs two "Artificial Intelligence" models:

AI Model 1: Compares the input image against all images in the digits dataset and uses the three closest matches to predict the digit.
AI Model 2: Compares the input image against the precomputed average images of each digit and uses the closest match to predict the digit.
5. Output
The script prints the predicted digit for both models:

AI Model 1 output: Based on comparison with individual images in the dataset.
AI Model 2 output: Based on comparison with the average images of each digit.
Additionally, the preprocessed input image and the final results are saved as CSV files (numero_escrito.csv).

How to Use
Replace the file path 'img8.jpg' with the path to your own 8x8 grayscale image.
Run the script.
View the predicted digit printed in the console and the CSV files generated (promedio.csv and numero_escrito.csv).
Example Output
yaml
Copy code
SOY LA INTELIGENCIA ARTIFICIAL 1, Y HE DETECTADO QUE EL DIGITO CORRESPONDE AL NUMERO: X
SOY LA INTELIGENCIA ARTIFICIAL 2, Y HE DETECTADO QUE EL DIGITO CORRESPONDE AL NUMERO: Y
Here, X and Y will be the digits predicted by AI Model 1 and AI Model 2, respectively.

Notes
Ensure that the input image is preprocessed correctly to fit the 8x8 pixel format and grayscale color scheme required for comparison.
The code uses simple Euclidean distance and may not perform as accurately as more advanced machine learning models like SVMs or neural networks.
