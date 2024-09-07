# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:12:18 2024

@author: radio
"""

import networkx as nx
import osmnx as ox
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np
from random import choice
import time
from datetime import datetime
import folium
from flask import Flask, render_template, jsonify
import threading
from scipy.spatial.distance import cdist
import json
import os
from Utils.MisLibrerias import *

# Define los alcances y el archivo de la cuenta de servicio para Google Sheets
SCOPES = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
SERVICE_ACCOUNT_FILE = '/etc/secrets/serenazgo-431820-3cf6c177b559.json'
#SERVICE_ACCOUNT_FILE = 'D:/colab/serenazgo-431820-3cf6c177b559.json'

SPREADSHEET_ID = '1gS6ZS6lS7Mc5B4TFI8HEK80xq4LANS6nU2O8V8-hEC8'

# Inicializa la aplicación Flask
app = Flask(__name__)

# Variable global para almacenar las posiciones de los serenos
serenos_positions = []
victimas_positions = []

# Variables globales para compartir datos entre hilos
G = None
nodes_df = None
initial_serenos = None
last_row_count = 0

# Función para leer datos de Google Sheets
def read_data(sheet_name):
    sheet = authenticate_google_sheets(sheet_name)
    data = sheet.get_all_records()
    if not data:
        headers = sheet.row_values(1)
        return pd.DataFrame(columns=headers)
    df = pd.DataFrame(data)
    return df

# Resto de funciones (download_graph, extract_nodes, select_random_nodes, etc.) permanecen igual

def update_positions():
    global serenos_positions, victimas_positions, G, nodes_df, initial_serenos, last_row_count
    while True:
        try:
            new_positions_df = simulate_sereno_movement(nodes_df, initial_serenos, G)
            serenos_positions = new_positions_df.to_dict('records')
            
            # Convertir valores numpy a tipos nativos de Python
            for pos in serenos_positions:
                for key, value in pos.items():
                    if isinstance(value, np.number):
                        pos[key] = value.item()
                        
            data_ciudadanos = read_data('Ciudadanos')
            print(data_ciudadanos)
            
            current_row_count = len(data_ciudadanos)
            
            if current_row_count > last_row_count:
                print("¡Alerta de Socorro!")
                
                try:
                    Tabla_Dijkstra = process_alert(data_ciudadanos, serenos_positions, G)
                    print(Tabla_Dijkstra)
                    
                    # Actualizar el estado del sereno con la prioridad más baja
                    sereno_to_alert = Tabla_Dijkstra[Tabla_Dijkstra['Estado'] == 'NORMAL'].iloc[0]
                    for i, sereno in enumerate(serenos_positions):
                        if sereno['Sereno'] == sereno_to_alert['Sereno']:
                            serenos_positions[i]['Estado'] = 'ALERTA'
                            serenos_positions[i]['Ruta'] = sereno_to_alert['Ruta']
                            break
                    
                    # Actualizar las posiciones de las víctimas
                    victimas_positions = data_ciudadanos[['Latitud', 'Longitud']].to_dict('records')
                except Exception as e:
                    print(f"Error al procesar la alerta: {e}")
            
            last_row_count = current_row_count
            
            initial_serenos = pd.DataFrame(serenos_positions)
            
            # Actualizar la hoja de Google Sheets
            sheet = authenticate_google_sheets()
            update_google_sheet(sheet, initial_serenos)
            
            time.sleep(3)
        except Exception as e:
            print(f"Error en update_positions: {e}")
            time.sleep(3)  # Esperar un poco antes de intentar de nuevo

# Rutas de Flask
@app.route('/')
def index():
    """Renderiza la página principal."""
    return render_template('index.html')

@app.route('/get_positions')
def get_positions():
    """Retorna las posiciones actuales de los serenos y víctimas en formato JSON."""
    positions = {
        'serenos': [{k: v.item() if isinstance(v, np.number) else v for k, v in pos.items()} for pos in serenos_positions],
        'victimas': victimas_positions
    }
    return json.dumps(positions, cls=NumpyEncoder)

@app.route('/get_heatmap_data')
def heatmap_data():
    return jsonify(get_heatmap_data())

@app.route('/get_functions_info')
def get_functions_info():
    # ... (el resto del código de esta función permanece igual)

@app.route('/get_functions_info_json')
def get_functions_info_json():
    # ... (el resto del código de esta función permanece igual)

def init_app():
    global G, nodes_df, initial_serenos
    
    # Descargar el grafo de la ciudad
    G = cargar_grafo_pickle('data/Grafo-Pueblo-Libre.gpickle')
    nodes_df = extract_nodes(G)
    
    # Autenticar y obtener datos de Google Sheets
    sheet = authenticate_google_sheets()
    
    # Limpiar la hoja 'Ciudadanos', manteniendo los encabezados
    ciudadanos_sheet = authenticate_google_sheets('Ciudadanos')
    headers = ciudadanos_sheet.row_values(1)  # Obtener los encabezados
    ciudadanos_sheet.clear()  # Limpiar toda la hoja
    ciudadanos_sheet.update('A1', [headers])  # Volver a insertar los encabezados
    
    num_serenos = get_serenos_count(sheet)
    unique_clusters_df = nodes_df.drop_duplicates(subset='Cluster')
    
    # Leer datos de los serenos desde un archivo Excel
    data_serenos = pd.read_excel('data/Serenazgo Pueblo Libre.xlsx')
    data_serenos_inicial = data_serenos.rename(columns={'Latitud': 'lat', 'Longitud': 'lon'})
    
    # Seleccionar nodos aleatorios para los serenos iniciales
    initial_serenos = select_sereno_positions(data_serenos_inicial)
    
    # Añadir información adicional a los serenos
    initial_serenos['Sereno'] = data_serenos['Sereno']
    initial_serenos['Forma de patrullaje'] = data_serenos['Forma de patrullaje']
    initial_serenos['Placa'] = data_serenos['Placa']
    initial_serenos['Turno'] = data_serenos['Turno']
    initial_serenos['Velocidad'] = data_serenos['Velocidad']
    initial_serenos['Estado'] = 'NORMAL'  # Inicializar todos los serenos en estado NORMAL
    initial_serenos['Ruta'] = [[] for _ in range(len(initial_serenos))]  # Inicializar rutas vacías
    initial_serenos['Cluster'] = data_serenos['Cluster']  # Asegúrate de que esta línea esté presente
    
    # Actualizar la hoja de Google Sheets con las posiciones iniciales
    update_google_sheet(sheet, initial_serenos)
    print(initial_serenos.columns)

def run_update_thread():
    update_thread = threading.Thread(target=update_positions)
    update_thread.daemon = True
    update_thread.start()

if __name__ == '__main__':
    init_app()
    run_update_thread()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
