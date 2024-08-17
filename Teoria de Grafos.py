# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 16:22:14 2024

@author: radio
"""


import pandas as pd
import osmnx as ox

from Utils.MisLibrerias import *
import pickle

ruta = 'data/Delitos - Serenazgo Pueblo Libre.xlsx'
# Ejemplo de uso de la función
data = agregar_dia_hora__dataset(ruta)



lugar = "Pueblo Libre, Lima, Peru"

G = ox.graph_from_place(lugar, network_type="drive")


n = 47 # numero de serenos
Gmod = clusterizar_vertices_aristas(G, n)


data = delitos_cluster(ruta, G)



# Ejemplo de uso:
df_resultado = Tabla_frecuencia_y_montecarlo_vertices(ruta)




# Ejemplo de uso
archivo_excel = 'data/Tabla-frecuencia-con-simulacion.xlsx'


# Asignar los valores de probabilidad de criminalidad a los nodos del grafo
G = asignar_probabilidad_criminalidad(G, archivo_excel)

guardar_grafo(G, 'data/Grafo-Pueblo-Libre.gpickle')

exportar_nodos_a_excel(G, 'data/Tabla-clysters-G.xlsx')