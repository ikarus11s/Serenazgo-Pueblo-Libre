# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:17:35 2024

@author: radio
"""


import pandas as pd
import os
import networkx as nx
import numpy as np
from sklearn.cluster import KMeans
import osmnx as ox
import numpy as np
from scipy.spatial import distance
from shapely.geometry import Point, LineString
import time
from math import radians, sin, cos, sqrt, atan2
from scipy import stats
from scipy.stats import poisson
import matplotlib.pyplot as plt
import seaborn as sns
import pickle



def agregar_dia_hora__dataset(ruta_archivo_excel):
    """
    Procesa un archivo Excel para añadir las columnas 'Dia de semana' y 'Horario',
    y reordena las columnas para colocar estas dos nuevas columnas entre 'Turno' y 'Fecha/Hora llegada'.
    
    Parámetros:
    - ruta_archivo_excel: Ruta del archivo Excel de entrada.
    
    Retorna:
    - df: DataFrame modificado.
    """
    # Leer el archivo Excel
    df = pd.read_excel(ruta_archivo_excel)

    # Convertir la columna 'Fecha/Hora alerta' a tipo datetime
    df['Fecha/Hora alerta'] = pd.to_datetime(df['Fecha/Hora alerta'], format='%d/%m/%Y %H:%M:%S')

    # Crear columna 'Dia de semana'
    df['Dia de semana'] = df['Fecha/Hora alerta'].dt.day_name()

    # Función para determinar la franja horaria
    def obtener_horario(hora):
        if hora >= pd.to_datetime('00:00:00').time() and hora < pd.to_datetime('01:00:00').time():
            return '00:00 - 01:00'
        elif hora >= pd.to_datetime('01:00:00').time() and hora < pd.to_datetime('02:00:00').time():
            return '01:00 - 02:00'
        elif hora >= pd.to_datetime('02:00:00').time() and hora < pd.to_datetime('03:00:00').time():
            return '02:00 - 03:00'
        elif hora >= pd.to_datetime('03:00:00').time() and hora < pd.to_datetime('04:00:00').time():
            return '03:00 - 04:00'
        elif hora >= pd.to_datetime('04:00:00').time() and hora < pd.to_datetime('05:00:00').time():
            return '04:00 - 05:00'
        elif hora >= pd.to_datetime('05:00:00').time() and hora < pd.to_datetime('06:00:00').time():
            return '05:00 - 06:00'
        elif hora >= pd.to_datetime('06:00:00').time() and hora < pd.to_datetime('07:00:00').time():
            return '06:00 - 07:00'
        elif hora >= pd.to_datetime('07:00:00').time() and hora < pd.to_datetime('08:00:00').time():
            return '07:00 - 08:00'
        elif hora >= pd.to_datetime('08:00:00').time() and hora < pd.to_datetime('09:00:00').time():
            return '08:00 - 09:00'
        elif hora >= pd.to_datetime('09:00:00').time() and hora < pd.to_datetime('10:00:00').time():
            return '09:00 - 10:00'
        elif hora >= pd.to_datetime('10:00:00').time() and hora < pd.to_datetime('11:00:00').time():
            return '10:00 - 11:00'
        elif hora >= pd.to_datetime('11:00:00').time() and hora < pd.to_datetime('12:00:00').time():
            return '11:00 - 12:00'
        elif hora >= pd.to_datetime('12:00:00').time() and hora < pd.to_datetime('13:00:00').time():
            return '12:00 - 13:00'
        elif hora >= pd.to_datetime('13:00:00').time() and hora < pd.to_datetime('14:00:00').time():
            return '13:00 - 14:00'
        elif hora >= pd.to_datetime('14:00:00').time() and hora < pd.to_datetime('15:00:00').time():
            return '14:00 - 15:00'
        elif hora >= pd.to_datetime('15:00:00').time() and hora < pd.to_datetime('16:00:00').time():
            return '15:00 - 16:00'
        elif hora >= pd.to_datetime('16:00:00').time() and hora < pd.to_datetime('17:00:00').time():
            return '16:00 - 17:00'
        elif hora >= pd.to_datetime('17:00:00').time() and hora < pd.to_datetime('18:00:00').time():
            return '17:00 - 18:00'
        elif hora >= pd.to_datetime('18:00:00').time() and hora < pd.to_datetime('19:00:00').time():
            return '18:00 - 19:00'
        elif hora >= pd.to_datetime('19:00:00').time() and hora < pd.to_datetime('20:00:00').time():
            return '19:00 - 20:00'
        elif hora >= pd.to_datetime('20:00:00').time() and hora < pd.to_datetime('21:00:00').time():
            return '20:00 - 21:00'
        elif hora >= pd.to_datetime('21:00:00').time() and hora < pd.to_datetime('22:00:00').time():
            return '21:00 - 22:00'
        elif hora >= pd.to_datetime('22:00:00').time() and hora < pd.to_datetime('23:00:00').time():
            return '22:00 - 23:00'
        else:
            return '23:00 - 00:00'

    # Crear columna 'Horario'
    df['Horario'] = df['Fecha/Hora alerta'].dt.time.apply(obtener_horario)

    
    # Definir la ruta de salida con el sufijo '_modificado'
    ruta_archivo_salida = os.path.splitext(ruta_archivo_excel)[0] + '.xlsx'

    # Guardar el DataFrame modificado a un nuevo archivo Excel
    df.to_excel(ruta_archivo_salida, index=False)
    
    print(f"El archivo ha sido procesado y guardado como '{ruta_archivo_salida}'.")

    return df





def agregar_campo_a_vertices(G, campo, array):
    """
    Agrega un nuevo campo a los nodos del grafo con los valores proporcionados en el array.
    
    Args:
    G (nx.Graph): El grafo al que se le agregarán los campos.
    campo (str): El nombre del campo a agregar a los nodos.
    array (list): Lista de valores a asignar a los nodos del grafo.
    
    Returns:
    nx.Graph: El grafo modificado con el nuevo campo en los nodos.
    """
    
    # Verificar que el tamaño del array coincida con el número de nodos
    if len(array) != len(G.nodes()):
        raise ValueError("El tamaño del array debe coincidir con el número de nodos en el grafo.")
    
    # Asignar valores del array al campo en los nodos del grafo
    for node, value in zip(G.nodes(), array):
        G.nodes[node][campo] = value
    





def agregar_campo_a_aristas(G,campo ,array):
    """
    Asigna pesos a las aristas del grafo basándose en una lista de valores.

    Parámetros:
    - G: Grafo de NetworkX al que se le asignarán los pesos.
    - lista: Lista de valores que se asignarán a las aristas basándose en el índice.

    Devuelve:
    - G: Grafo con los pesos asignados a las aristas.
    """
    # Verificar que el número de valores coincida con el número de aristas
    num_aristas = len(G.edges())
    if len(array) != num_aristas:
        raise ValueError("El número de valores no coincide con el número de aristas en el grafo.")
    
    # Asignar pesos a las aristas
    for (u, v, data),valor in zip(G.edges(data=True),array):
        data[campo] = valor
    
    
def calcular_centroides_aristas(G):
    """
    Calcula el centroide de cada arista basado en su geometría.
    
    Args:
    G (nx.Graph): El grafo con aristas que contienen geometrías.
    
    Returns:
    dict: Un diccionario con las coordenadas del centroide para cada arista.
    """
    if G is None or not isinstance(G, nx.Graph):
        return {}
    
    centroides = {}
    for u, v, data in G.edges(data=True):
        geometry = data.get('geometry')
        if geometry and isinstance(geometry, LineString):
            centroid = geometry.centroid
            centroides[(u, v)] = (centroid.x, centroid.y)
        else:
            # Si no hay geometría, usar el punto medio entre los nodos
            pos_u = G.nodes[u].get('pos', (G.nodes[u].get('x', 0), G.nodes[u].get('y', 0)))
            pos_v = G.nodes[v].get('pos', (G.nodes[v].get('x', 0), G.nodes[v].get('y', 0)))
            centroides[(u, v)] = ((pos_u[0] + pos_v[0]) / 2, (pos_u[1] + pos_v[1]) / 2)
    return centroides

def clusterizar_vertices_aristas(G, n):
    """
    Realiza la clusterización de los nodos y aristas de un grafo en n clusters usando K-means.
    
    Args:
    G (nx.Graph): El grafo que se va a clusterizar.
    n (int): El número de clusters.
    
    Returns:
    nx.Graph: El grafo modificado con la información de clusters añadida a los nodos y aristas.
    """
    if G is None or not isinstance(G, nx.Graph):
        raise ValueError("G debe ser un objeto nx.Graph válido")
    
    # Obtener las coordenadas de los nodos
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        # Si 'pos' no está, usar 'x' e 'y' para crear el atributo 'pos'
        pos = {node: (G.nodes[node].get('x', 0), G.nodes[node].get('y', 0)) for node in G.nodes()}
    
    # Extraer las coordenadas de los nodos
    X = np.array([pos[node] for node in G.nodes()])
    
    # Aplicar K-means a los nodos
    kmeans = KMeans(n_clusters=n, random_state=0).fit(X)
    labels = kmeans.labels_
    
    # Agregar el campo 'Cluster' a los nodos usando la función proporcionada
    agregar_campo_a_vertices(G, 'Cluster', labels)
    
    # Calcular centroides de aristas
    centroides_aristas = calcular_centroides_aristas(G)
    
    # Extraer coordenadas de centroides para clusterización
    X_edges = np.array([centroides_aristas[edge] for edge in G.edges()])
    
    # Aplicar K-means a las aristas
    if len(X_edges) > 0:
        kmeans_edges = KMeans(n_clusters=n, random_state=0).fit(X_edges)
        labels_edges = kmeans_edges.labels_
    else:
        labels_edges = [0] * len(G.edges())  # Si no hay aristas, asignar cluster 0
    
    # Agregar el campo 'Cluster' a las aristas usando la función proporcionada
    agregar_campo_a_aristas(G, 'Cluster', labels_edges)
    
    return G


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # Radio de la Tierra en kilómetros
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

def point_to_line_distance(point, line):
    point = Point(point)
    return point.distance(line)

def calcular_centroides_aristas(G):
    centroides = {}
    for u, v, data in G.edges(data=True):
        geometry = data.get('geometry')
        if geometry and isinstance(geometry, LineString):
            centroid = geometry.centroid
            centroides[(u, v)] = (centroid.x, centroid.y)
        else:
            pos_u = G.nodes[u].get('y', 0), G.nodes[u].get('x', 0)
            pos_v = G.nodes[v].get('y', 0), G.nodes[v].get('x', 0)
            centroides[(u, v)] = ((pos_u[1] + pos_v[1]) / 2, (pos_u[0] + pos_v[0]) / 2)
    return centroides

def delitos_cluster(ruta, G):
    # Cargar el archivo Excel
    df = pd.read_excel(ruta)
    
    # Inicializar las nuevas columnas
    nuevas_columnas = ['ClusterVertice', 'VerticeMasCercano', 'LatVertice', 'LonVertice',
                       'ClusterArista', 'AristaMasCercana', 'LatArista', 'LonArista']
    for col in nuevas_columnas:
        df[col] = None
    
    # Calcular centroides de aristas
    centroides_aristas = calcular_centroides_aristas(G)
    
    for idx, row in df.iterrows():
        try:
            lat_delito, lon_delito = row['Latitud'], row['Longitud']         
            # Encontrar el vértice más cercano
            min_dist_vertice = float('inf')
            vertice_cercano = None
            
            for node, data in G.nodes(data=True):
                lat, lon = data.get('y'), data.get('x')
                if lat is not None and lon is not None:
                    dist = haversine_distance(lat_delito, lon_delito, lat, lon)
                    if dist < min_dist_vertice:
                        min_dist_vertice = dist
                        vertice_cercano = node
            
            # Actualizar información del vértice más cercano
            if vertice_cercano is not None:
                node_data = G.nodes[vertice_cercano]
                df.at[idx, 'ClusterVertice'] = node_data.get('Cluster', -1)
                df.at[idx, 'VerticeMasCercano'] = vertice_cercano
                df.at[idx, 'LatVertice'] = node_data.get('y')
                df.at[idx, 'LonVertice'] = node_data.get('x')
            
            # Encontrar la arista más cercana
            min_dist_arista = float('inf')
            arista_cercana = None
            
            for edge in G.edges():
                u, v = edge
                data = G.get_edge_data(u, v)
                geometry = data.get('geometry')
                if geometry and isinstance(geometry, LineString):
                    line = geometry
                else:
                    # Si no hay geometría, crear una línea recta entre los nodos
                    u_data, v_data = G.nodes[u], G.nodes[v]
                    line = LineString([(u_data['x'], u_data['y']), (v_data['x'], v_data['y'])])
                dist = point_to_line_distance((lon_delito, lat_delito), line)
                if dist < min_dist_arista:
                    min_dist_arista = dist
                    arista_cercana = (u, v)
            
            
            # Actualizar información de la arista más cercana
            if arista_cercana is not None:
                edge_data = G.get_edge_data(*arista_cercana)
                df.at[idx, 'ClusterArista'] = edge_data.get('Cluster', -1)
                df.at[idx, 'AristaMasCercana'] = f"{arista_cercana[0]}-{arista_cercana[1]}"
                centroide = centroides_aristas.get(arista_cercana, (None, None))
                df.at[idx, 'LatArista'] = centroide[1]
                df.at[idx, 'LonArista'] = centroide[0]
        
        except Exception as e:
            print(f"Error procesando fila {idx}: {e}")
    
    # Guardar el DataFrame modificado
    ruta_salida = ruta.replace('.xlsx', '.xlsx')
    df.to_excel(ruta_salida, index=False)
    
    return df



def Tabla_frecuencia_y_montecarlo_vertices(ruta, filtro_dia=None, filtro_turno=None, filtro_hora=None, num_simulaciones=1000):
    # Cargar el archivo Excel
    df = pd.read_excel(ruta)
    
    # Filtrar por día de la semana, turno y hora si se proporcionan
    if filtro_dia is not None:
        df = df[df['Dia de semana'].isin(filtro_dia)]
    if filtro_turno is not None:
        df = df[df['Turno'].isin(filtro_turno)]
    if filtro_hora is not None:
        df = df[df['Horario'].isin(filtro_hora)]
    
    # Asegúrate de que las columnas de coordenadas existen
    if 'Latitud' not in df.columns or 'Longitud' not in df.columns:
        raise ValueError("El archivo debe contener las columnas 'Latitud' y 'Longitud'.")
    
    # Crear un campo con coordenadas combinadas
    df['Coordenadas'] = df['Latitud'].astype(str) + ' ' + df['Longitud'].astype(str)
    
    # Contar la cantidad de delitos por 'VerticeMasCercano'
    frecuencia = df.groupby(['VerticeMasCercano', 'Coordenadas'])['Número de parte'].nunique().reset_index()
    frecuencia.columns = ['VerticeMasCercano', 'Coordenadas', 'Cantidad de delitos']
    
    # Calcular el porcentaje
    total_delitos = frecuencia['Cantidad de delitos'].sum()
    frecuencia['%'] = (frecuencia['Cantidad de delitos'] / total_delitos) * 100
    
    # Simulación de Monte Carlo
    def simular_criminalidad(lambda_param, num_simulaciones):
        #print(poisson.rvs(lambda_param, size=num_simulaciones))
        return poisson.rvs(lambda_param, size=num_simulaciones)
    
    # Calcular la tasa media de delitos por día
    dias_totales = (df['Fecha/Hora alerta'].max() - df['Fecha/Hora alerta'].min()).days + 1
    tasa_media_diaria = frecuencia['Cantidad de delitos'] / dias_totales
    
    # Realizar simulación y calcular probabilidades
    resultados_simulacion = []
    for _, row in frecuencia.iterrows():
        simulaciones = simular_criminalidad(row['Cantidad de delitos'] / dias_totales, num_simulaciones)
        prob_crimen = np.mean(simulaciones > 0)  # Probabilidad de que ocurra al menos un crimen
        resultados_simulacion.append({
            'VerticeMasCercano': row['VerticeMasCercano'],
            'Media': np.mean(simulaciones),
            'Desviación Estándar': np.std(simulaciones),
            'Probabilidad Criminalidad': prob_crimen
        })
    
    # Crear DataFrame con resultados de simulación
    df_simulacion = pd.DataFrame(resultados_simulacion)
    
    # Unir resultados de simulación con la tabla de frecuencia
    frecuencia = pd.merge(frecuencia, df_simulacion, on='VerticeMasCercano')
    
    # Crear la carpeta 'data' si no existe
    carpeta_salida = 'data'
    if not os.path.exists(carpeta_salida):
        os.makedirs(carpeta_salida)
    
    # Crear y guardar el archivo Excel en la carpeta 'data'
    output_file = os.path.join(carpeta_salida, 'Tabla-frecuencia-con-simulacion.xlsx')
    frecuencia.to_excel(output_file, index=False)
    
    return frecuencia




def asignar_probabilidad_criminalidad(G, archivo_excel):
    """
    Asigna valores de probabilidad de criminalidad a los nodos del grafo basándose en un archivo Excel.
    
    Args:
    G (nx.Graph): El grafo al que se le asignarán los valores.
    archivo_excel (str): Ruta del archivo Excel que contiene los datos.
    hoja (str): Nombre de la hoja del archivo Excel que contiene los datos.
    
    Returns:
    nx.Graph: El grafo modificado con el campo 'Criminalidad' actualizado.
    """
    
    # Leer el archivo Excel
    df = pd.read_excel(archivo_excel)
    
    # Verificar que las columnas necesarias existen
    if 'VerticeMasCercano' not in df.columns or 'Probabilidad Criminalidad' not in df.columns:
        raise ValueError("El archivo Excel debe contener las columnas 'VerticeMasCercano' y 'Probabilidad Criminalidad'.")
    
    # Crear un diccionario con los valores de probabilidad de criminalidad
    probabilidad_dict = df.set_index('VerticeMasCercano')['Probabilidad Criminalidad'].to_dict()
    
    # Crear una lista de valores para asignar a los nodos
    array = []
    for node in G.nodes():
        array.append(probabilidad_dict.get(node, 0))
    
    # Agregar el campo 'Criminalidad' a los nodos del grafo
    agregar_campo_a_vertices(G, 'Criminalidad', array)
    
    return G




def exportar_nodos_a_excel(G, archivo_excel):
    """
    Exporta todos los valores de los nodos del grafo G a un archivo Excel.
    
    Args:
    G (nx.Graph): El grafo del cual se exportarán los nodos.
    archivo_excel (str): El nombre del archivo Excel donde se guardarán los datos.
    """
    # Extraer los datos de los nodos
    nodos_datos = []
    for nodo, atributos in G.nodes(data=True):
        datos_nodo = {'Nodo': nodo}
        datos_nodo.update(atributos)
        nodos_datos.append(datos_nodo)

    # Crear un DataFrame de pandas con los datos de los nodos
    df_nodos = pd.DataFrame(nodos_datos)
    
    # Exportar el DataFrame a un archivo Excel
    df_nodos.to_excel(archivo_excel, index=False, engine='openpyxl')

    print(f"Datos de nodos exportados a {archivo_excel}")



def guardar_grafo(G, ruta):
    """
    Guarda el grafo G en un archivo con formato gpickle.
    
    Args:
    G (nx.Graph): El grafo a guardar.
    ruta (str): La ruta donde se guardará el archivo gpickle.
    """
    with open(ruta, 'wb') as f:
        pickle.dump(G, f)
    print(f"Grafo guardado en {ruta}")






def cargar_grafo_pickle(ruta):
    """
    Carga un grafo desde un archivo utilizando pickle.
    
    Args:
    ruta (str): La ruta del archivo pickle.
    
    Returns:
    nx.Graph: El grafo cargado.
    """
    with open(ruta, 'rb') as f:
        G = pickle.load(f)
    print(f"Grafo cargado desde {ruta}")
    return G




    
    
    
def crear_reloj_datos(ruta_archivo):
    # Leer el archivo Excel
    df = pd.read_excel(ruta_archivo)
    
    # Asegurarse de que las columnas necesarias existen
    if 'Dia de semana' not in df.columns or 'Horario' not in df.columns:
        raise ValueError("El archivo debe contener las columnas 'Dia de semana' y 'Horario'.")
    
    # Crear un diccionario para mapear los días de la semana a números
    dias_map = {
        'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
        'Friday': 4, 'Saturday': 5, 'Sunday': 6
    }
    
    # Convertir los días de la semana a números
    df['Dia_Num'] = df['Dia de semana'].map(dias_map)
    
    # Extraer la hora de inicio del rango horario
    df['Hora'] = df['Horario'].apply(lambda x: int(x.split(':')[0]))
    
    # Crear una tabla pivote para contar los eventos por día y hora
    pivot = pd.pivot_table(df, values='Dia de semana', index='Hora', columns='Dia_Num', aggfunc='count', fill_value=0)
    
    # Asegurarse de que todos los días y horas estén representados
    for dia in range(7):
        if dia not in pivot.columns:
            pivot[dia] = 0
    pivot = pivot.reindex(columns=range(7))
    pivot = pivot.reindex(index=range(24))
    pivot = pivot.fillna(0)
    
    # Configurar el estilo de seaborn
    sns.set(style="whitegrid")
    
    # Crear la figura y los ejes
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    
    # Configurar los ángulos y radios
    theta = np.linspace(0, 2*np.pi, 8)  # 8 para cerrar el círculo
    r = np.arange(25)  # 25 para incluir la medianoche
    
    # Crear la cuadrícula polar
    ax.set_thetagrids(np.degrees(theta[:-1]), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    ax.set_rgrids(r[::3], labels=[f'{h%24:02d}:00' for h in r[::3]], angle=0)
    ax.set_ylim(0, 24)
    
    # Crear el heatmap
    heatmap = ax.pcolormesh(theta, r, np.column_stack((pivot.values, pivot.values[:, 0])), cmap='YlOrRd')
    
    # Añadir una barra de color
    cbar = plt.colorbar(heatmap, ax=ax, pad=0.1)
    cbar.set_label('Número de eventos', rotation=270, labelpad=15)
    
    # Configurar el título
    plt.title('Reloj de Datos: Eventos por Día y Hora', y=1.1)
    
    # Ajustar el diseño
    plt.tight_layout()
    
    # Guardar la figura en la misma carpeta que el archivo de entrada
    carpeta_salida = os.path.dirname(ruta_archivo)
    nombre_archivo = os.path.splitext(os.path.basename(ruta_archivo))[0]
    ruta_salida = os.path.join(carpeta_salida, f'{nombre_archivo}_reloj_datos.png')
    plt.savefig(ruta_salida, dpi=300, bbox_inches='tight')
    
    print(f"Reloj de datos guardado como: {ruta_salida}")
    
    # Cerrar la figura para liberar memoria
    plt.close(fig)