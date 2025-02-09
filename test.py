import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# 1️⃣ Charger les données depuis un fichier Excel
df = pd.read_excel("factures_clients.xlsx")
# 2️⃣ Séparer les entrées (X) et la sortie (y)
X = df.drop(columns=['Produit_Achat_Suivant'])  # Toutes les colonnes sauf la cible
y = df['Produit_Achat_Suivant']  # La colonne cible (le produit à prédire)
# 3️⃣ Convertir la sortie (y) en nombres
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# 4️⃣ Séparer les données en 80% entraînement, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 5️⃣ Entraîner un modèle Random Forest
model = RandomForestClassifier(n_estimators=50, random_state=42)
model.fit(X_train, y_train)
# 6️⃣ Faire des prédictions et calculer la précision
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Précision du modèle :", accuracy)
# 7️⃣ Tester avec un nouveau client
nouveau_client = [[120, 80, 70, 4,0]]  # Exemple : dépenses d'un client
prediction = model.predict(nouveau_client)
produit_pred = label_encoder.inverse_transform(prediction)  # Reconvertir en mot
print("Produit recommandé :", produit_pred[0])
