# Transfer Learning–Based Classification of Poultry Diseases for Enhanced Health Management

## Table of Contents
1. [Overview](#overview)  
2. [Features](#features)  
3. [Project Structure](#project-structure)  
4. [Technologies & Requirements](#technologies--requirements)  
5. [Installation](#installation)  
6. [Usage](#usage)  
7. [Report & Documentation](#report--documentation)  
8. [Contributing](#contributing)  
9. [License](#license)  

---

## Overview
This project implements an AI‑powered, mobile‑friendly web application to classify poultry diseases from fecal images using **transfer learning**. It categorizes each sample into one of four classes:

- **Salmonella**  
- **Newcastle Disease**  
- **Coccidiosis**  
- **Healthy**

The goal is to provide immediate, on‑farm diagnostic support—helping farmers reduce economic losses and improve flock health without costly laboratory tests.

---

## Features
- **Image‑based disease classification** via pre‑trained CNN backbones (VGG16, MobileNetV2)  
- **Four‑class output**: Salmonella | Newcastle Disease | Coccidiosis | Healthy  
- **Real‑time inference** through a Flask web interface  
- **Lightweight model** suitable for mobile deployment (TensorFlow Lite)  
- **User-friendly UI**: upload an image, click submit, view prediction  
- **Extensible architecture**: add new disease classes or backbone models  

---

## Project Structure
```

Classification-of-Poultry-Diseases/
├── app.py                  # Flask application entrypoint
├── requirements.txt        # Python dependencies
├── templates/
│   ├── index.html          # Home & upload page
│   └── result.html         # Prediction display page
├── static/
│   └── css/                # Stylesheets
├── models/
│   └── healthy\_vs\_rotten.h5  # Trained CNN model
├── notebooks/              # Jupyter notebooks for data prep & training
├── data/
│   └── poultry-diseases/   # Kaggle dataset (downloaded via Kaggle API)
└── README.md

````

---

## Technologies & Requirements

### Core Libraries
- **Flask** 2.2.x  
- **TensorFlow** 2.13.x  
- **NumPy** 1.24.x  
- **Pillow** 9.5.x (for `load_img`)  
- **Werkzeug** 2.2.x  

### Hardware & Environment
- Python 3.8+  
- GPU (recommended for initial model fine‑tuning)  
- Google Colab (free GPU) or local CUDA‑enabled machine  

---

## Installation

1. **Clone the repository**  
   ```bash
   git clone https://github.com/Durgesh-Vaigandla/Classification-of-Poultry-Diseases.git
   cd Classification-of-Poultry-Diseases
   ```

2. **Create & activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**

   * Create a Kaggle API token (`kaggle.json`) and place it in `~/.kaggle/`
   * Run in a Colab or locally:

     ```bash
     kaggle datasets download chandrashekarnatesh/poultry-diseases -p data/poultry-diseases --unzip
     ```

5. **Ensure the trained model**

   * The pre‑trained `poultry_diseases_detection.keras` file should reside in `/`.
   * If not present, train your own using the Jupyter notebooks in `/main.ipynb`.

---

## Usage

1. **Start the Flask app**

   ```bash
   python app.py
   ```
2. **Open your browser** and navigate to `http://127.0.0.1:5000/`
3. **Upload an image** of poultry feces (JPG, PNG, JPEG)
4. **Submit** and view the predicted disease class and confidence score

---

## Report & Documentation

A full academic‑style project report is available in PDF and DOCX formats:

* **PDF:** [Durgesh\_Project\_Report.pdf](/Durgesh_Project_Report.pdf)

---

## Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/YourFeature`)
3. Commit your changes (`git commit -m 'Add some feature'`)
4. Push to the branch (`git push origin feature/YourFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.
