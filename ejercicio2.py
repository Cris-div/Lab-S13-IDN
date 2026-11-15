import pandas as pd
from sklearn.cluster import KMeans


# Agrupación de estudiantes según horas de estudio y nota final.
data = {
    "Estudiante": ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8", "E9", "E10"],
    "Horas_estudio": [1, 2, 2, 3, 4, 5, 6, 7, 8, 9],
    "Nota_final": [8, 10, 11, 12, 14, 15, 16, 17, 18, 19],
}

df = pd.DataFrame(data)
modelo = KMeans(n_clusters=2, random_state=42, n_init=10)
etiquetas = modelo.fit_predict(df[["Horas_estudio", "Nota_final"]])

# Numerar los grupos de menor a mayor nota del centroide.
orden = modelo.cluster_centers_[:, 1].argsort()
grupos = {int(etiqueta): grupo + 1 for grupo, etiqueta in enumerate(orden)}
df["Grupo"] = [grupos[int(etiqueta)] for etiqueta in etiquetas]

print(df.to_string(index=False))
