import pandas as pd
from sklearn.cluster import KMeans


# Clasificación de productos según precio y cantidad vendida.
data = {
    "Producto": ["P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"],
    "Precio": [5, 6, 7, 20, 22, 25, 80, 85, 90, 100],
    "Cantidad_vendida": [300, 280, 260, 150, 140, 130, 50, 45, 40, 35],
}

df = pd.DataFrame(data)
modelo = KMeans(n_clusters=3, random_state=42, n_init=10)
etiquetas = modelo.fit_predict(df[["Precio", "Cantidad_vendida"]])

# Numerar los grupos de menor a mayor precio del centroide.
orden = modelo.cluster_centers_[:, 0].argsort()
grupos = {int(etiqueta): grupo + 1 for grupo, etiqueta in enumerate(orden)}
df["Grupo"] = [grupos[int(etiqueta)] for etiqueta in etiquetas]
df["Clasificacion"] = df["Grupo"].map(
    {1: "Economico", 2: "Intermedio", 3: "Mayor precio"}
)

print(df.to_string(index=False))
