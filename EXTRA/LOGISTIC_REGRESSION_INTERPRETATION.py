import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Generar datos simulados
np.random.seed(42)
n = 200
glucosa = np.random.normal(120, 30, n)  # Promedio 120 mg/dL, desviación 30
presion = np.random.normal(80, 10, n)   # Promedio 80 mmHg, desviación 10
diabetes = (glucosa + np.random.normal(0, 20, n) > 130).astype(int)  # Etiqueta 1 si glucosa alta

# Crear DataFrame
df = pd.DataFrame({'Glucosa': glucosa, 'Presion': presion, 'Diabetes': diabetes})

# Dividir en entrenamiento y prueba
X = df[['Glucosa', 'Presion']]
y = df['Diabetes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo de regresión logística
modelo = LogisticRegression()
modelo.fit(X_train, y_train)

# Predicciones
y_pred = modelo.predict(X_test)
y_prob = modelo.predict_proba(X_test)[:, 1]

# Matriz de confusión
conf_matrix = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:\n", conf_matrix)

# Reporte de clasificación
print("\nReporte de Clasificación:\n", classification_report(y_test, y_pred))

# Interpretación de coeficientes
coef = modelo.coef_[0]
intercepto = modelo.intercept_[0]
print(f"\nCoeficientes: Glucosa = {coef[0]:.4f}, Presión = {coef[1]:.4f}")
print(f"Intercepto: {intercepto:.4f}")

# Curva ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()
