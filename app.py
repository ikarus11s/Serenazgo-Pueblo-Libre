import networkx as nx
import osmnx as ox
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
import numpy as np
from random import choice
import time
from datetime import datetime
from flask import Flask, render_template, jsonify
import threading
from scipy.spatial.distance import cdist
import json
import os
from Utils.MisLibrerias import *

# Define los alcances y el archivo de la cuenta de servicio para Google Sheets
SCOPES = ['https://spreadsheets.google.com/feeds', 'https://www.googleapis.com/auth/drive']
SPREADSHEET_ID = '1gS6ZS6lS7Mc5B4TFI8HEK80xq4LANS6nU2O8V8-hEC8'

# Inicializa la aplicación Flask
app = Flask(__name__)

# Variable global para almacenar las posiciones de los serenos
serenos_positions = []
victimas_positions = []

# Cargar las credenciales desde la variable de entorno
creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
creds_dict = json.loads(creds_json)
creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)

def read_data(sheet_name):
    sheet = authenticate_google_sheets(sheet_name)
    data = sheet.get_all_records()
    if not data:
        headers = sheet.row_values(1)
        return pd.DataFrame(columns=headers)
    df = pd.DataFrame(data)
    return df

def extract_nodes(graph):
    nodes_data = [(node, data['y'], data['x'], data['street_count'], data['Cluster'], data['Criminalidad']) 
                  for node, data in graph.nodes(data=True)]
    nodes_df = pd.DataFrame(nodes_data, columns=['id', 'lat', 'lon', 'street_count', 'Cluster', 'Criminalidad'])
    return nodes_df

def select_sereno_positions(data_serenos):
    required_columns = ['Sereno', 'Cluster', 'id', 'lat', 'lon', 'Estado']
    for column in required_columns:
        if column not in data_serenos.columns:
            raise KeyError(f"Columna '{column}' no encontrada en el DataFrame.")
    positions = data_serenos[required_columns].copy()
    return positions.reset_index(drop=True)

def authenticate_google_sheets(sheet_name=None):
    client = gspread.authorize(creds)
    spreadsheet = client.open_by_key(SPREADSHEET_ID)
    return spreadsheet.worksheet(sheet_name) if sheet_name else spreadsheet.sheet1

def get_serenos_count(sheet):
    return len(sheet.get_all_records())

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
            current_cluster = str(serenos.iloc[i]['Cluster'])
            all_neighbors = list(graph.neighbors(current_node_id))
            
            if current_cluster == '-1':
                criminalidad_neighbors = [(node, graph.nodes[node]['Criminalidad']) for node in all_neighbors]
                criminalidad_neighbors.sort(key=lambda x: x[1], reverse=True)
                new_node = criminalidad_neighbors[0][0] if criminalidad_neighbors else current_node_id
            else:
                same_cluster_neighbors = [node for node in all_neighbors if str(graph.nodes[node]['Cluster']) == current_cluster]
                new_node = choice(same_cluster_neighbors) if same_cluster_neighbors else (choice(all_neighbors) if all_neighbors else current_node_id)
            
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
                next_node = ruta.pop(0)
                new_lat = graph.nodes[next_node]['y']
                new_lon = graph.nodes[next_node]['x']
                new_cluster = graph.nodes[next_node]['Cluster']
                new_positions.append([next_node, new_lat, new_lon, serenos.iloc[i]['Sereno'], serenos.iloc[i]['Turno'], 
                                      serenos.iloc[i]['Forma de patrullaje'], serenos.iloc[i]['Placa'], 
                                      serenos.iloc[i]['Velocidad'], 'ALERTA', ruta, new_cluster])
            else:
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
            
            for pos in serenos_positions:
                for key, value in pos.items():
                    if isinstance(value, np.number):
                        pos[key] = value.item()
                        
            data_ciudadanos = read_data('Ciudadanos')
            current_row_count = len(data_ciudadanos)
            
            if current_row_count > last_row_count:
                try:
                    Tabla_Dijkstra = process_alert(data_ciudadanos, serenos_positions, G)
                    sereno_to_alert = Tabla_Dijkstra[Tabla_Dijkstra['Estado'] == 'NORMAL'].iloc[0]
                    for i, sereno in enumerate(serenos_positions):
                        if sereno['Sereno'] == sereno_to_alert['Sereno']:
                            serenos_positions[i]['Estado'] = 'ALERTA'
                            serenos_positions[i]['Ruta'] = sereno_to_alert['Ruta']
                            break
                    
                    victimas_positions = data_ciudadanos[['Latitud', 'Longitud']].to_dict('records')
                except Exception as e:
                    print(f"Error en process_alert: {e}")
            
            last_row_count = current_row_count
            initial_serenos = pd.DataFrame(serenos_positions)
            
            sheet = authenticate_google_sheets()
            update_google_sheet(sheet, initial_serenos)
            
            time.sleep(3)
        except Exception as e:
            print(f"Error en update_positions: {e}")
            time.sleep(3)

def calculate_nearest_node(lat, lon, G):
    nodes = ox.graph_to_gdfs(G, edges=False)
    closest_idx = cdist([(lon, lat)], nodes[['x', 'y']]).argmin()
    return nodes.iloc[closest_idx]

def calculate_distances(G, start_node, end_nodes):
    distances, paths = nx.single_source_dijkstra(G, start_node)
    total_distances = {}
    for end_node in end_nodes:
        if end_node in paths:
            path = paths[end_node]
            total_distance = sum(G[path[i]][path[i+1]][0].get('length', 0) for i in range(len(path) - 1))
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
    return render_template('index.html')

@app.route('/get_positions')
def get_positions():
    positions = {
        'serenos': [{k: v.item() if isinstance(v, np.number) else v for k, v in pos.items()} for pos in serenos_positions],
        'victimas': victimas_positions
    }
    return json.dumps(positions, cls=NumpyEncoder)

@app.route('/get_heatmap_data')
def heatmap_data():
    return jsonify(get_heatmap_data())
def main():
    global G, nodes_df, initial_serenos
    
    G = cargar_grafo_pickle('data/Grafo-Pueblo-Libre.gpickle')
    nodes_df = extract_nodes(G)
    
    sheet = authenticate_google_sheets()
    
    ciudadanos_sheet = authenticate_google_sheets('Ciudadanos')
    headers = ciudadanos_sheet.row_values(1)
    ciudadanos_sheet.clear()
    ciudadanos_sheet.update('A1', [headers])
    
    num_serenos = get_serenos_count(sheet)
    
    data_serenos = pd.read_excel('data/Serenazgo Pueblo Libre.xlsx')
    data_serenos_inicial = data_serenos.rename(columns={'Latitud': 'lat', 'Longitud': 'lon'})
    
    initial_serenos = select_sereno_positions(data_serenos_inicial)
    
    initial_serenos['Sereno'] = data_serenos['Sereno']
    initial_serenos['Forma de patrullaje'] = data_serenos['Forma de patrullaje']
    initial_serenos['Placa'] = data_serenos['Placa']
    initial_serenos['Turno'] = data_serenos['Turno']
    initial_serenos['Velocidad'] = data_serenos['Velocidad']
    initial_serenos['Estado'] = 'NORMAL'
    initial_serenos['Ruta'] = [[] for _ in range(len(initial_serenos))]
    initial_serenos['Cluster'] = data_serenos['Cluster']
    
    update_google_sheet(sheet, initial_serenos)

    update_thread = threading.Thread(target=update_positions)
    update_thread.daemon = True
    update_thread.start()

if __name__ == '__main__':
    main()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
