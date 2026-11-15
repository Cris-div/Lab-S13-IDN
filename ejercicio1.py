import pandas as pd
from sklearn.cluster import KMeans


# Segmentación de clientes según edad e ingreso mensual.
data = {
    "Cliente": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"],
    "Edad": [18, 20, 22, 35, 37, 40, 55, 58, 60, 62],
    "Ingreso": [900, 1000, 1200, 2500, 2700, 3000, 4800, 5000, 5200, 5500],
}

df = pd.DataFrame(data)
modelo = KMeans(n_clusters=3, random_state=42, n_init=10)
etiquetas = modelo.fit_predict(df[["Edad", "Ingreso"]])

# Numerar los grupos de menor a mayor ingreso del centroide.
orden = modelo.cluster_centers_[:, 1].argsort()
grupos = {int(etiqueta): grupo + 1 for grupo, etiqueta in enumerate(orden)}
df["Grupo"] = [grupos[int(etiqueta)] for etiqueta in etiquetas]

print(df.to_string(index=False))
