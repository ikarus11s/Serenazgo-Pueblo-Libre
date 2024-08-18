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


# Lista global para almacenar mensajes de depuración
debug_messages = []

def read_data(sheet_name):
    sheet = authenticate_google_sheets(sheet_name)
    data = sheet.get_all_records()
    if not data:
        headers = sheet.row_values(1)
        return pd.DataFrame(columns=headers)
    df = pd.DataFrame(data)
    return df


def download_graph(place):
    """
    Descarga el grafo para el lugar especificado utilizando OSMnx.
    
    :param place: Nombre del lugar para el cual descargar el grafo
    :return: Grafo de NetworkX
    """
    G = ox.graph_from_place(place, network_type="drive")
    return G


def extract_nodes(graph):
    nodes_data = [(node, data['y'], data['x'], data['street_count'], data['Cluster'], data['Criminalidad']) 
                  for node, data in graph.nodes(data=True)]
    nodes_df = pd.DataFrame(nodes_data, columns=['id', 'lat', 'lon', 'street_count', 'Cluster', 'Criminalidad'])
    return nodes_df


def select_random_nodes(nodes, n):
    """
    Selecciona n nodos al azar de los nodos disponibles.
    
    :param nodes: DataFrame con todos los nodos
    :param n: Número de nodos a seleccionar
    :return: DataFrame con los nodos seleccionados
    """
    return nodes.sample(n=n).reset_index(drop=True)



def select_sereno_positions(data_serenos):
    required_columns = ['Sereno', 'Cluster', 'id', 'lat', 'lon', 'Estado']
    
    for column in required_columns:
        if column not in data_serenos.columns:
            raise KeyError(f"Columna '{column}' no encontrada en el DataFrame.")
    
    positions = data_serenos[required_columns].copy()
    positions = positions.reset_index(drop=True)
    
    return positions


def authenticate_google_sheets(sheet_name=None):
    """
    Autentica con Google Sheets y devuelve la hoja especificada o la primera hoja si no se indica ninguna.
    
    :param sheet_name: Nombre de la hoja a abrir (opcional)
    :return: Objeto de hoja de Google Sheets
    """
    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    
    if sheet_name:
        sheet = spreadsheet.worksheet(sheet_name)
    else:
        sheet = spreadsheet.sheet1
    
    return sheet

def get_serenos_count(sheet):
    """
    Obtiene el número de serenos desde la hoja de cálculo.
    
    :param sheet: Objeto de hoja de Google Sheets
    :return: Número de serenos (filas con datos)
    """
    data = sheet.get_all_records()
    return len(data)


def update_google_sheet(sheet, data_frame):
    updated_data = data_frame[['lat', 'lon', 'Estado']].values.tolist()
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_frame['Fecha y Hora'] = date_time
    updated_data_with_time = data_frame[['Fecha y Hora', 'lat', 'lon', 'Estado']].values.tolist()
    
    header = sheet.row_values(1)
    fecha_col_idx = header.index('Fecha y Hora') + 1
    lat_col_idx = header.index('Latitud') + 1
    lon_col_idx = header.index('Longitud') + 1
    estado_col_idx = header.index('Estado') + 1
    
    num_rows = len(data_frame)
    
    fecha_range = f'{chr(64 + fecha_col_idx)}2:{chr(64 + fecha_col_idx)}{num_rows + 1}'
    lat_range = f'{chr(64 + lat_col_idx)}2:{chr(64 + lat_col_idx)}{num_rows + 1}'
    lon_range = f'{chr(64 + lon_col_idx)}2:{chr(64 + lon_col_idx)}{num_rows + 1}'
    estado_range = f'{chr(64 + estado_col_idx)}2:{chr(64 + estado_col_idx)}{num_rows + 1}'
    
    sheet.update(values=[[row[0]] for row in updated_data_with_time], range_name=fecha_range)
    sheet.update(values=[[row[1]] for row in updated_data_with_time], range_name=lat_range)
    sheet.update(values=[[row[2]] for row in updated_data_with_time], range_name=lon_range)
    sheet.update(values=[[row[3]] for row in updated_data_with_time], range_name=estado_range)

def simulate_sereno_movement(nodes, serenos, graph):
    new_positions = []
    for i in range(len(serenos)):
        if serenos.iloc[i]['Estado'] == 'NORMAL':
            current_node_id = serenos.iloc[i]['id']
            current_cluster = serenos.iloc[i]['Cluster']
            
            # Convertir el cluster a string para asegurar comparación correcta
            current_cluster = str(current_cluster)
            
            # Obtener todos los vecinos
            all_neighbors = list(graph.neighbors(current_node_id))
            
            if current_cluster == '-1':
                # Para serenos con cluster -1, implementar el recorrido del agente viajero maximizando criminalidad
                criminalidad_neighbors = [(node, graph.nodes[node]['Criminalidad']) for node in all_neighbors]
                criminalidad_neighbors.sort(key=lambda x: x[1], reverse=True)  # Ordenar por criminalidad descendente
                
                if criminalidad_neighbors:
                    new_node = criminalidad_neighbors[0][0]  # Seleccionar el nodo con mayor criminalidad
                else:
                    new_node = current_node_id  # Si no hay vecinos, quedarse en el mismo lugar
            else:
                # Para otros clusters, mantener la lógica original
                same_cluster_neighbors = [node for node in all_neighbors 
                                          if str(graph.nodes[node]['Cluster']) == current_cluster]
                
                if same_cluster_neighbors:
                    new_node = choice(same_cluster_neighbors)
                elif all_neighbors:
                    new_node = choice(all_neighbors)
                else:
                    new_node = current_node_id
            
            new_lat = graph.nodes[new_node]['y']
            new_lon = graph.nodes[new_node]['x']
            new_cluster = graph.nodes[new_node]['Cluster']
            
            new_positions.append([new_node, new_lat, new_lon, serenos.iloc[i]['Sereno'], 
                                  serenos.iloc[i]['Turno'], serenos.iloc[i]['Forma de patrullaje'], 
                                  serenos.iloc[i]['Placa'], serenos.iloc[i]['Velocidad'], 
                                  serenos.iloc[i]['Estado'], serenos.iloc[i]['Ruta'], new_cluster])
        else:  # Estado ALERTA
            ruta = serenos.iloc[i]['Ruta']
            if ruta:
                # Movemos el sereno al siguiente nodo en su ruta hacia la víctima
                next_node = ruta.pop(0)
                new_lat = graph.nodes[next_node]['y']
                new_lon = graph.nodes[next_node]['x']
                new_cluster = graph.nodes[next_node]['Cluster']
                new_positions.append([next_node, new_lat, new_lon, serenos.iloc[i]['Sereno'], serenos.iloc[i]['Turno'], 
                                      serenos.iloc[i]['Forma de patrullaje'], serenos.iloc[i]['Placa'], 
                                      serenos.iloc[i]['Velocidad'], 'ALERTA', ruta, new_cluster])
            else:
                # Si la ruta está vacía, el sereno ha llegado a la víctima y se queda ahí
                new_positions.append([serenos.iloc[i]['id'], serenos.iloc[i]['lat'], serenos.iloc[i]['lon'], 
                                      serenos.iloc[i]['Sereno'], serenos.iloc[i]['Turno'], 
                                      serenos.iloc[i]['Forma de patrullaje'], serenos.iloc[i]['Placa'], 
                                      serenos.iloc[i]['Velocidad'], 'ALERTA', [], serenos.iloc[i]['Cluster']])

    return pd.DataFrame(new_positions, columns=['id', 'lat', 'lon', 'Sereno', 'Turno', 'Forma de patrullaje', 
                                                'Placa', 'Velocidad', 'Estado', 'Ruta', 'Cluster'])

def update_positions():
    global serenos_positions, victimas_positions, G, nodes_df, initial_serenos, last_row_count
    last_row_count = 0
    new_positions_df = simulate_sereno_movement(nodes_df, initial_serenos, G)
    serenos_positions = new_positions_df.to_dict('records')

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

def calculate_nearest_node(lat, lon, G):
    """
    Calcula el nodo más cercano a las coordenadas dadas.
    
    :param lat: Latitud
    :param lon: Longitud
    :param G: Grafo de NetworkX
    :return: Nodo más cercano
    """
    nodes = ox.graph_to_gdfs(G, edges=False)
    closest_idx = cdist([(lon, lat)], nodes[['x', 'y']]).argmin()
    return nodes.iloc[closest_idx]

def get_length_by_osmid(G, osmid):
    for u, v, data in G.edges(data=True):
        if isinstance(data.get('osmid'), list):
            if osmid in data['osmid']:
                return data.get('length')
        elif data.get('osmid') == osmid:
            return data.get('length')
    return None


def calculate_distances(G, start_node, end_nodes):
    distances, paths = nx.single_source_dijkstra(G, start_node)
    total_distances = {}
    for end_node in end_nodes:
        if end_node in paths:
            path = paths[end_node]
            total_distance = 0
            for i in range(len(path) - 1):
                edge_data = G.get_edge_data(path[i], path[i+1])
                if edge_data:
                    # Sumamos la longitud de cada arista en la ruta
                    length = edge_data[0].get('length', 0)
                    total_distance += length
            total_distances[end_node] = total_distance
    return total_distances, paths



def process_alert(data_ciudadanos, serenos_positions, G):
    victim_data = data_ciudadanos.iloc[-1]
    victim_node = calculate_nearest_node(victim_data['Latitud'], victim_data['Longitud'], G)
    
    rows = []
    for sereno in serenos_positions:
        sereno_node = calculate_nearest_node(sereno['lat'], sereno['lon'], G)
        distances, paths = calculate_distances(G, sereno_node.name, [victim_node.name])
        
        if victim_node.name in distances:
            distance = distances[victim_node.name]
            path = paths[victim_node.name]
            velocity_kmh = sereno['Velocidad']
            velocity_ms = velocity_kmh * 1000 / 3600
            time = distance / velocity_ms if velocity_ms > 0 else float('inf')
            
            row = {
                'Fecha y Hora': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'Datos de la víctima(s)': victim_data['Datos de la víctima(s)'],
                'Latitud_victima': float(victim_data['Latitud']),
                'Longitud_victima': float(victim_data['Longitud']),
                'Sereno': sereno['Sereno'],
                'Forma de patrullaje': sereno['Forma de patrullaje'],
                'Velocidad': float(velocity_kmh),
                'Latitud_sereno': float(sereno['lat']),
                'Longitud_sereno': float(sereno['lon']),
                'Estado': sereno['Estado'],
                'Ruta': [int(node) for node in path],
                'Distancia': float(distance),
                'Tiempo': float(time),
            }
            rows.append(row)    
    
    Tabla_Dijkstra = pd.DataFrame(rows)
    Tabla_Dijkstra = Tabla_Dijkstra.sort_values('Tiempo')
    Tabla_Dijkstra['Prioridad'] = range(1, len(Tabla_Dijkstra) + 1)
    
    # Guardar la tabla con nombre que incluye fecha y hora
    #now = datetime.now()
    #filename = f"Tabla Dijkstra - {now.strftime('%Y-%m-%d %H-%M-%S')}.xlsx"
    #Tabla_Dijkstra.to_excel(os.path.join('D:', filename))
    
    return Tabla_Dijkstra


def get_heatmap_data():
    df = pd.read_excel('data/Delitos - Serenazgo Pueblo Libre.xlsx')
    return df[['Latitud', 'Longitud']].values.tolist()


def verify_main_execution():
    results = {}

    # Verificar la carga del grafo
    try:
        G = cargar_grafo_pickle('data/Grafo-Pueblo-Libre.gpickle')
        results['cargar_grafo_pickle'] = f"Éxito. Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}"
    except Exception as e:
        results['cargar_grafo_pickle'] = f"Error: {str(e)}"

    # Verificar la extracción de nodos
    try:
        nodes_df = extract_nodes(G)
        results['extract_nodes'] = f"Éxito. Filas: {len(nodes_df)}, Columnas: {', '.join(nodes_df.columns)}"
    except Exception as e:
        results['extract_nodes'] = f"Error: {str(e)}"

    # Verificar la autenticación de Google Sheets
    try:
        sheet = authenticate_google_sheets()
        results['authenticate_google_sheets'] = f"Éxito. Hoja: {sheet.title}"
    except Exception as e:
        results['authenticate_google_sheets'] = f"Error: {str(e)}"

    # Verificar la limpieza de la hoja 'Ciudadanos'
    try:
        ciudadanos_sheet = authenticate_google_sheets('Ciudadanos')
        headers = ciudadanos_sheet.row_values(1)
        ciudadanos_sheet.clear()
        ciudadanos_sheet.update('A1', [headers])
        results['clean_ciudadanos_sheet'] = f"Éxito. Encabezados: {', '.join(headers)}"
    except Exception as e:
        results['clean_ciudadanos_sheet'] = f"Error: {str(e)}"

    # Verificar el conteo de serenos
    try:
        num_serenos = get_serenos_count(sheet)
        results['get_serenos_count'] = f"Éxito. Número de serenos: {num_serenos}"
    except Exception as e:
        results['get_serenos_count'] = f"Error: {str(e)}"

    # Verificar la obtención de clusters únicos
    try:
        unique_clusters_df = nodes_df.drop_duplicates(subset='Cluster')
        results['unique_clusters'] = f"Éxito. Clusters únicos: {len(unique_clusters_df)}"
    except Exception as e:
        results['unique_clusters'] = f"Error: {str(e)}"

    # Verificar la lectura de datos de serenos
    try:
        data_serenos = pd.read_excel('data/Serenazgo Pueblo Libre.xlsx')
        data_serenos_inicial = data_serenos.rename(columns={'Latitud': 'lat', 'Longitud': 'lon'})
        results['read_serenos_data'] = f"Éxito. Filas: {len(data_serenos)}, Columnas: {', '.join(data_serenos.columns)}"
    except Exception as e:
        results['read_serenos_data'] = f"Error: {str(e)}"

    # Verificar la selección de posiciones de serenos
    try:
        initial_serenos = select_sereno_positions(data_serenos_inicial)
        results['select_sereno_positions'] = f"Éxito. Serenos seleccionados: {len(initial_serenos)}"
    except Exception as e:
        results['select_sereno_positions'] = f"Error: {str(e)}"

    # Verificar la adición de información adicional a los serenos
    try:
        initial_serenos['Sereno'] = data_serenos['Sereno']
        initial_serenos['Forma de patrullaje'] = data_serenos['Forma de patrullaje']
        initial_serenos['Placa'] = data_serenos['Placa']
        initial_serenos['Turno'] = data_serenos['Turno']
        initial_serenos['Velocidad'] = data_serenos['Velocidad']
        initial_serenos['Estado'] = 'NORMAL'
        initial_serenos['Ruta'] = [[] for _ in range(len(initial_serenos))]
        initial_serenos['Cluster'] = data_serenos['Cluster']
        results['add_serenos_info'] = f"Éxito. Columnas añadidas: {', '.join(initial_serenos.columns)}"
    except Exception as e:
        results['add_serenos_info'] = f"Error: {str(e)}"

    # Verificar la actualización de la hoja de Google Sheets
    try:
        update_google_sheet(sheet, initial_serenos)
        results['update_google_sheet'] = "Éxito. Hoja actualizada con datos iniciales de serenos."
    except Exception as e:
        results['update_google_sheet'] = f"Error: {str(e)}"

    return results
 
def debug_print(*args, **kwargs):
    message = " ".join(map(str, args))
    debug_messages.append(message)
    print(message, **kwargs)  # Esto seguirá imprimiendo en la consola
   

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)



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
    print("Serenos positions:", serenos_positions)
    print("Victimas positions:", victimas_positions)
    return json.dumps(positions, cls=NumpyEncoder)


@app.route('/get_heatmap_data')
def heatmap_data():
    return jsonify(get_heatmap_data())


# aqui inidica el codigo de prueba...
@app.route('/get_functions_info')
def get_functions_info():
    functions_info = ""

    # Cargar el grafo
    G = cargar_grafo_pickle('data/Grafo-Pueblo-Libre.gpickle')
    functions_info += f"Grafo cargado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas\n\n"

    # Extraer nodos
    nodes_df = extract_nodes(G)
    functions_info += f"Nodos extraídos (primeros 5):\n{nodes_df.head().to_string()}\n\n"

    # Leer datos de los serenos
    data_serenos = pd.read_excel('data/Serenazgo Pueblo Libre.xlsx')
    data_serenos_inicial = data_serenos.rename(columns={'Latitud': 'lat', 'Longitud': 'lon'})
    functions_info += f"Datos de serenos (primeros 5):\n{data_serenos_inicial.head().to_string()}\n\n"

    return functions_info

@app.route('/get_functions_info_json')
def get_functions_info_json():
    functions_info = {}

    # Cargar el grafo
    G = cargar_grafo_pickle('data/Grafo-Pueblo-Libre.gpickle')
    functions_info['grafo'] = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges()
    }

    # Extraer nodos
    nodes_df = extract_nodes(G)
    functions_info['nodos'] = nodes_df.head().to_dict()

    # Leer datos de los serenos
    data_serenos = pd.read_excel('data/Serenazgo Pueblo Libre.xlsx')
    data_serenos_inicial = data_serenos.rename(columns={'Latitud': 'lat', 'Longitud': 'lon'})
    functions_info['serenos'] = data_serenos_inicial.head().to_dict()

    return jsonify(functions_info)




@app.route('/verify_main')
def verify_main():
    results = verify_main_execution()
    return jsonify(results)
# aqui termina el codigo de prueba ...



@app.route('/get_debug_info')
def get_debug_info():
    global debug_messages
    return jsonify(debug_messages)



def main():
    """Función principal que inicializa y ejecuta la aplicación."""
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
    #unique_clusters_df.to_excel('data/valores_unicos_clusters.xlsx', index=False)

    
    # Leer datos de los serenos desde un archivo Excel
    data_serenos = pd.read_excel('data/Serenazgo Pueblo Libre.xlsx')
    data_serenos_inicial = data_serenos.rename(columns={'Latitud': 'lat', 'Longitud': 'lon'})
    
    # Seleccionar nodos aleatorios para los serenos iniciales
    #initial_serenos = select_random_nodes(nodes_df, num_serenos)
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

    # Iniciar el hilo de actualización de posiciones
    update_thread = threading.Thread(target=update_positions)
    update_thread.daemon = True
    update_thread.start()

    #borra esta parte
    debug_print("Iniciando la aplicación...")
    
    # Descargar el grafo de la ciudad
    G = cargar_grafo_pickle('data/Grafo-Pueblo-Libre.gpickle')
    debug_print(f"Grafo cargado: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
    
    nodes_df = extract_nodes(G)
    debug_print(f"Nodos extraídos: {len(nodes_df)}")
    
    # Autenticar y obtener datos de Google Sheets
    sheet = authenticate_google_sheets()
    debug_print("Autenticación con Google Sheets completada")

    debug_print("Inicializando la aplicación Flask...")
    debug_print("Serenos positions:", serenos_positions)
    debug_print("Victimas positions:", victimas_positions)
    

    #borra hasta aqui

    
    # Iniciar la aplicación Flask
    app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
