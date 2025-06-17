🧠 CNN for Skin Lesion Classification

This project implements a Convolutional Neural Network (CNN) model to classify dermoscopic images of skin lesions into various diagnostic categories. The goal is to assist dermatologists and researchers by automating part of the diagnostic process using deep learning techniques.

📁 Project Structure

- CNN_Skin_leasure.ipynb: Jupyter Notebook containing all steps from data preprocessing, model training, evaluation, and visualization.

🩺 Dataset

The project uses the HAM10000 dataset, a benchmark dataset for skin lesion classification. It contains 10,015 dermatoscopic images divided into 7 classes:

- Actinic keratoses (akiec)
- Basal cell carcinoma (bcc)
- Benign keratosis-like lesions (bkl)
- Dermatofibroma (df)
- Melanoma (mel)
- Melanocytic nevi (nv)
- Vascular lesions (vasc)

🧪 Methodology

- Image preprocessing (normalization, resizing)
- CNN model definition using PyTorch
- Training and validation loop
- Evaluation metrics: Accuracy, Confusion Matrix
- Ontology-based explanation for predicted diagnoses

🖥️ Requirements

- Python 3.8+
- PyTorch
- scikit-learn
- NumPy
- Matplotlib
- tqdm
- (Optional) Weights & Biases for experiment tracking

🚀 How to Run

1. Download the HAM10000 dataset from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000.
2. Run the notebook CNN_Skin_leasure.ipynb step by step.
3. Visualize results and evaluate performance.

📊 Results

Model performance will vary depending on preprocessing and hyperparameters. Example results include:
- Validation accuracy ~80–90%

🔍 Future Work

- Incorporate data augmentation
- Improve class balancing
- Add explainability via Grad-CAM or ontology mapping

📄 License

This project is licensed under the MIT License.
