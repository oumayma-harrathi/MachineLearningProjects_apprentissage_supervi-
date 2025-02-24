import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

df = pd.read_excel("factures_clients.xlsx")
X = df.drop(columns=['Produit_Achat_Suivant'])
y = df['Produit_Achat_Suivant']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle :", accuracy)

nouveau_client = [[120, 80, 70, 4, 0]]
prediction = model.predict(nouveau_client)
produit_pred = label_encoder.inverse_transform(prediction)
print("Produit recommandé :", produit_pred[0])
