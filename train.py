import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"

df = pd.read_csv(url)

# Variables
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# División
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Regresión logística
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

# Red neuronal
nn_model = MLPClassifier(
    hidden_layer_sizes=(32,16),
    max_iter=1000,
    random_state=42
)

nn_model.fit(X_train_scaled, y_train)

# Predicciones
pred = nn_model.predict(X_test_scaled)

# Accuracy
acc = accuracy_score(y_test, pred)

print("Accuracy:", acc)

# Matriz de confusión
cm = confusion_matrix(y_test, pred)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.savefig('static/confusion_matrix.png')

# Guardar modelos
joblib.dump(log_model, 'logistic_model.pkl')
joblib.dump(nn_model, 'nn_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Modelos guardados correctamente")