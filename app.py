from flask import Flask, request, jsonify
from scipy.sparse import load_npz, vstack
import pandas as pd
import numpy as np
import os

# Cargar el dataset de anime
anime_data = pd.read_csv("anime-dataset-2023.csv")

# Función para cargar y combinar los fragmentos de la matriz
def load_similarity_matrix():
    matrix_parts = []
    part_number = 0
    while True:
        part_file = f"anime_similarity_matrix_part_{part_number:03d}.npz"
        if not os.path.exists(part_file):
            break
        matrix_parts.append(load_npz(part_file))
        part_number += 1
    return vstack(matrix_parts)

# Cargar la matriz de similitud al iniciar el servidor
similarity_matrix = load_similarity_matrix()

# Crear la aplicación Flask
app = Flask(__name__)

# Ruta para obtener recomendaciones
@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    anime_id = request.args.get("anime_id")
    top_n = int(request.args.get("top_n", 5))

    if not anime_id:
        return jsonify({"error": "anime_id is required"}), 400

    try:
        anime_id = int(anime_id)
        # Obtener las similitudes para el anime solicitado
        anime_similarities = similarity_matrix[anime_id].toarray().flatten()
        # Ordenar los índices de los animes por similitud (de mayor a menor)
        similar_indices = np.argsort(-anime_similarities)[1:top_n + 1]
        recommendations = anime_data.iloc[similar_indices].to_dict(orient="records")
        return jsonify(recommendations)
    except ValueError:
        return jsonify({"error": "anime_id must be an integer"}), 400

# Ejecutar la aplicación en modo local
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Azure usa el puerto 5000 por defecto para Flask
