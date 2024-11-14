from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.sparse import load_npz, vstack
import pandas as pd
import numpy as np
import os

# Cargar el dataset de anime
anime_data = pd.read_csv("modified_anime_dataset (1).csv")

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
CORS(app)  # Habilita CORS en la aplicación

# Ruta para obtener recomendaciones
@app.route("/recommendations", methods=["GET"])
def get_recommendations():
    anime_ids = request.args.getlist("anime_id")  # Permite recibir una lista de IDs
    top_n = int(request.args.get("top_n", 5))

    if not anime_ids:
        return jsonify({"error": "anime_id or list of anime_ids is required"}), 400

    try:
        # Convertir los IDs a enteros
        anime_ids = [int(anime_id) for anime_id in anime_ids]

        # Obtener las similitudes para cada anime en el arreglo de IDs
        similarities = np.zeros(similarity_matrix.shape[0])
        for anime_id in anime_ids:
            anime_similarities = similarity_matrix[anime_id].toarray().flatten()
            similarities += anime_similarities  # Sumar las similitudes para cada anime

        # Ordenar los índices de los animes por similitud (de mayor a menor)
        similar_indices = np.argsort(-similarities)

        # Filtrar los animes originales usados en la recomendación (anime_ids)
        filtered_indices = [idx for idx in similar_indices if idx not in anime_ids]

        # Limitar los resultados a los top_n y obtener sus datos
        recommendations = anime_data.iloc[filtered_indices[:top_n]].to_dict(orient="records")
        
        return jsonify(recommendations)
    except ValueError:
        return jsonify({"error": "anime_id must be an integer or a list of integers"}), 400
    except IndexError:
        return jsonify({"error": "One or more anime_ids are out of range"}), 400

# Ejecutar la aplicación en modo local
if __name__ == "__main__":
    app.run(debug=True)
