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
#SERVICE_ACCOUNT_FILE = '/etc/secrets/serenazgo-431820-3cf6c177b559.json'
#SERVICE_ACCOUNT_FILE = 'D:/colab/serenazgo-431820-3cf6c177b559.json'

SPREADSHEET_ID = '1gS6ZS6lS7Mc5B4TFI8HEK80xq4LANS6nU2O8V8-hEC8'

# Inicializa la aplicación Flask
app = Flask(__name__)

# Variable global para almacenar las posiciones de los serenos
serenos_positions = []
victimas_positions = []



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
    # Obtén las credenciales de la variable de entorno
    credentials_json = os.environ.get('GOOGLE_SHEETS_CREDENTIALS')
    
    if not credentials_json:
        raise ValueError("Las credenciales de Google Sheets no están configuradas en las variables de entorno.")
    
    # Convierte la cadena JSON en un diccionario
    credentials_info = json.loads(credentials_json)
    
    # Crea las credenciales a partir del diccionario
    creds = Credentials.from_service_account_info(credentials_info, scopes=SCOPES)
    
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
    return json.dumps(positions, cls=NumpyEncoder)


@app.route('/get_heatmap_data')
def heatmap_data():
    return jsonify(get_heatmap_data())


# aqui inidica el codigo de prueba...
@app.route('/get_functions_info')
def get_functions_info():
    functions_info = ""

    # Función read_data
    functions_info += "read_data\n"
    sheet = authenticate_google_sheets('Ciudadanos')  # Usando 'Ciudadanos' como ejemplo
    data = read_data('Ciudadanos')
    functions_info += str(data.head().to_dict()) + "\n\n\n"

    # Función download_graph
    functions_info += "download_graph\n"
    G = download_graph("Pueblo Libre, Lima, Peru")
    functions_info += f"Número de nodos: {G.number_of_nodes()}, Número de aristas: {G.number_of_edges()}\n\n\n"

    # Función extract_nodes
    functions_info += "extract_nodes\n"
    nodes = extract_nodes(G)
    functions_info += str(nodes.head().to_dict()) + "\n\n\n"

    # Función select_random_nodes
    functions_info += "select_random_nodes\n"
    random_nodes = select_random_nodes(nodes, 5)
    functions_info += str(random_nodes.to_dict()) + "\n\n\n"

    # Función select_sereno_positions
    functions_info += "select_sereno_positions\n"
    data_serenos = pd.read_excel('data/Serenazgo Pueblo Libre.xlsx')
    sereno_positions = select_sereno_positions(data_serenos)
    functions_info += str(sereno_positions.head().to_dict()) + "\n\n\n"

    # Función authenticate_google_sheets
    functions_info += "authenticate_google_sheets\n"
    sheet = authenticate_google_sheets()
    functions_info += f"Título de la hoja: {sheet.title}\n\n\n"

    # Función get_serenos_count
    functions_info += "get_serenos_count\n"
    count = get_serenos_count(sheet)
    functions_info += f"Número de serenos: {count}\n\n\n"

    # Función get_heatmap_data
    functions_info += "get_heatmap_data\n"
    heatmap_data = get_heatmap_data()
    functions_info += str(heatmap_data[:5]) + "\n\n\n"

    return functions_info

@app.route('/get_functions_info_json')
def get_functions_info_json():
    functions_info = {}

    # Función read_data
    sheet = authenticate_google_sheets('Ciudadanos')  # Usando 'Ciudadanos' como ejemplo
    data = read_data('Ciudadanos')
    functions_info['read_data'] = data.head().to_dict()

    # Función download_graph
    G = download_graph("Pueblo Libre, Lima, Peru")
    functions_info['download_graph'] = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges()
    }

    # Función extract_nodes
    nodes = extract_nodes(G)
    functions_info['extract_nodes'] = nodes.head().to_dict()

    # Función select_random_nodes
    random_nodes = select_random_nodes(nodes, 5)
    functions_info['select_random_nodes'] = random_nodes.to_dict()

    # Función select_sereno_positions
    data_serenos = pd.read_excel('data/Serenazgo Pueblo Libre.xlsx')
    sereno_positions = select_sereno_positions(data_serenos)
    functions_info['select_sereno_positions'] = sereno_positions.head().to_dict()

    # Función authenticate_google_sheets
    sheet = authenticate_google_sheets()
    functions_info['authenticate_google_sheets'] = {'sheet_title': sheet.title}

    # Función get_serenos_count
    count = get_serenos_count(sheet)
    functions_info['get_serenos_count'] = {'count': count}

    # Función get_heatmap_data
    heatmap_data = get_heatmap_data()
    functions_info['get_heatmap_data'] = heatmap_data[:5]

    return jsonify(functions_info)
# aqui termina el codigo de prueba ...
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

    # Iniciar la aplicación Flask
    #app.run(debug=True, use_reloader=False)

if __name__ == '__main__':
    main()
    app.run(host='0.0.0.0', port=10000, debug=True, use_reloader=False)
