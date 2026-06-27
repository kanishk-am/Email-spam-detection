# 📧 Spam Email Detection using Machine Learning

## 📌 Project Overview

This project implements a **Spam Email Detection** system using **Machine Learning** in Python. It classifies emails as **Spam** or **Not Spam** by converting text into numerical features with **TF-IDF Vectorization** and training a **Logistic Regression** classifier.

The application also displays the model's training accuracy and visualizes its performance using a **Confusion Matrix**.

---

## 🚀 Features

* Detects whether an email is Spam or Not Spam.
* Uses **TF-IDF Vectorizer** for text feature extraction.
* Implements **Logistic Regression** for classification.
* Calculates training accuracy.
* Accepts custom email input from the user.
* Displays spam probability.
* Visualizes model performance with a Confusion Matrix.

---

## 📂 Project Structure

```
Spam-Email-Detection/
│
├── spam_email_detection.py
├── README.md
└── requirements.txt
```

---

## 🛠️ Technologies Used

* Python 3
* Pandas
* Matplotlib
* Scikit-learn

---

## 📊 Dataset

The project uses a manually created dataset consisting of:

* **15 Spam Emails**
* **15 Not Spam Emails**

Each email is labeled as:

| Label | Meaning  |
| ----- | -------- |
| 1     | Spam     |
| 0     | Not Spam |

### Example Spam Emails

* Win a free iPhone now
* Claim your cash prize
* Get rich quickly
* Free gift card available
* You won cash instantly

### Example Not Spam Emails

* Meeting at 10 AM
* Project review tomorrow
* Submit your assignment
* Lunch with client
* Report is ready for review

---

## ⚙️ Machine Learning Workflow

1. Create the email dataset.
2. Convert text into numerical vectors using **TF-IDF Vectorizer**.
3. Train a **Logistic Regression** model.
4. Predict email categories.
5. Calculate training accuracy.
6. Accept user input for new email classification.
7. Display spam probability.
8. Plot the Confusion Matrix.

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/kanishk-am/Spam-Email-Detection.git
```

Navigate to the project directory:

```bash
cd Spam-Email-Detection
```

Install the required libraries:

```bash
pip install pandas matplotlib scikit-learn
```

Or install using:

```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Project

Run the Python script:

```bash
python spam_email_detection.py
```

Example:

```
Enter Email Text:

Congratulations! You have won a free vacation package.

Spam Probability: 0.98

Result: SPAM EMAIL
```

Another example:

```
Enter Email Text:

Please send the project report before 5 PM.

Spam Probability: 0.06

Result: NOT SPAM EMAIL
```

---

## 📈 Output

The program displays:

* Training Accuracy
* Spam Probability
* Email Classification (Spam / Not Spam)
* Confusion Matrix Visualization

---

## 🧠 Machine Learning Model

### Feature Extraction

* TF-IDF Vectorizer

### Classification Algorithm

* Logistic Regression

### Evaluation

* Training Accuracy
* Confusion Matrix

---

## 📚 Future Improvements

* Use a larger real-world email dataset.
* Split data into training and testing sets.
* Evaluate with additional metrics such as:

  * Precision
  * Recall
  * F1-Score
  * ROC-AUC Score
* Save the trained model using Pickle or Joblib.
* Build a web application using Flask or Streamlit.
* Add support for email attachments and HTML emails.

---

## 📦 Requirements

```
pandas
matplotlib
scikit-learn
```

You can also install them using:

```bash
pip install pandas matplotlib scikit-learn
```

---

## 👨‍💻 Author

**Kanishk A M**

GitHub: https://github.com/kanishk-am

---

## 📄 License

This project is licensed under the MIT License.
