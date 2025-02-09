import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


our_emails=["Free cash available now!",  # spam
    "Important update about your account",  # non-spam
    "Click here to claim your prize!",  # spam
    "Let's meet tomorrow at 10 AM",  # non-spam
    "Special offer for you!",  # spam
    "Meeting confirmation for tomorrow",  # non-spam
    "You've won a $5000 gift card!",  # spam
    "Can you send me the file by 5 PM?",  # non-spam
    "Act now! Limited time offer!",  # spam
    "Reminder for your doctor's appointment"]  # non-spam
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 spam, 0 non-spam


df = pd.DataFrame({     #on fait une conversion en dataframe car c'est une structure tabulaire organise les données et facile a integrir les biblio de machine learning 
    'email': our_emails,
    'label': labels
})

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['email']) 
y = df['label'] #déja se trouve sous la forme des nombres 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)#30% pour test et 70% pour l'apprentissage 
model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)


#affichage
print(accuracy)
print(conf_matrix)