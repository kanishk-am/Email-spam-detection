import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# ----------------------------------
# Dataset
# ----------------------------------

spam_emails = [
    "Win a free iPhone now",
    "Claim your cash prize",
    "You have won a lottery",
    "Limited time discount",
    "Earn money from home",
    "Free vacation package",
    "Exclusive offer for you",
    "Get rich quickly",
    "Congratulations you are selected",
    "Click here to claim reward",
    "Free gift card available",
    "Special offer expires today",
    "You won cash instantly",
    "Get a free smartphone",
    "Claim your reward now"
]

not_spam_emails = [
    "Meeting at 10 AM",
    "Project review tomorrow",
    "Submit your assignment",
    "Team meeting postponed",
    "Lunch with client",
    "Interview scheduled",
    "Project deadline extended",
    "Please send the report",
    "Class starts at 9 AM",
    "Can we discuss the project",
    "Important update regarding project",
    "Assignment submission reminder",
    "Client meeting confirmed",
    "Report is ready for review",
    "Thank you for attending the meeting"
]

emails = spam_emails + not_spam_emails
labels = [1] * len(spam_emails) + [0] * len(not_spam_emails)

df = pd.DataFrame({
    "Email": emails,
    "Spam": labels
})


vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(df["Email"])
y = df["Spam"]


model = LogisticRegression(max_iter=1000)

model.fit(X, y)



predictions = model.predict(X)

correct = (predictions == y).sum()
accuracy = correct / len(y)

print("Training Accuracy:", round(accuracy * 100, 2), "%")


email = input("\nEnter Email Text:\n")

email_vector = vectorizer.transform([email])

spam_probability = model.predict_proba(email_vector)[0][1]

print("\nSpam Probability:", round(spam_probability, 4))

if spam_probability >= 0.5:
    print("Result: SPAM EMAIL")
else:
    print("Result: NOT SPAM EMAIL")


cm = confusion_matrix(y, predictions)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Not Spam", "Spam"]
)

disp.plot()

plt.title("Confusion Matrix - Spam Detection")
plt.show()
