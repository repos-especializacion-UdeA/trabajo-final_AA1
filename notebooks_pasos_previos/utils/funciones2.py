#Librerias
import math
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.io
from scipy import signal


# Funciones de prueba

def test_hola():
    print("Hola amigos")

####################################################################################################
# Funciones sobre las señales
####################################################################################################


# Numero del estimulo
def indice_numero(df, num):
  return (df.index[df == num][0],df.index[df == num][-1])

# Cambios de nivel
def detectar_cambios_nivel(data, valor):
    cambios = []
    for i in range(1, len(data)):
        if data[i] == valor and data[i-1] != valor:
            cambios.append((i-1,0))
            cambios.append((i,valor))
        elif data[i] != valor and data[i-1] == valor:
            cambios.append((i-1,valor))
            cambios.append((i,0))
    cambios.pop(0)
    cambios.pop()
    return cambios

# Promt to @Chat-GPT: 
def segment_signal(data, window_size, step_size):
    """
    Segmenta una señal en ventanas deslizantes.
    
    Parámetros:
    data : array-like
        La señal EMG que se desea segmentar.
    window_size : int
        El tamaño de cada ventana (en número de muestras).
    step_size : int
        El número de muestras entre el inicio de ventanas consecutivas.
    
    Retorno:
    windows : list
        Una lista de ventanas segmentadas.
    """
    windows = []
    for start in range(0, len(data) - window_size + 1, step_size):
        windows.append(data[start:start + window_size])
    return np.array(windows)

# Promt to @Chat-GPT: dame una funcion que realice la compresion mu sobre una ventana de datos
def mu_compression(data, mu=255):
    """
    Aplica la compresión μ-law a una ventana de datos.
    
    Parámetros:
    data : array-like
        Los datos de entrada a comprimir (ventana de datos).
    mu : int, opcional
        El parámetro de compresión μ (por defecto es 255, comúnmente usado en telecomunicaciones).

    Retorno:
    compressed_data : array-like
        Los datos comprimidos usando la compresión μ-law.
    """
    # Normaliza los datos en el rango [-1, 1]
    data_normalized = np.clip(data / np.max(np.abs(data)), -1, 1)
    
    # Aplica la compresión μ-law
    compressed_data = np.sign(data_normalized) * np.log1p(mu * np.abs(data_normalized)) / np.log1p(mu)
    
    return compressed_data

# Promt to @Chat-GPT: 
def normalize_signal(data):
    """
    Normaliza la señal EMG en el rango [-1, 1].
    
    Parámetros:
    data : array-like
        La señal EMG que se desea normalizar.
    
    Retorno:
    normalized_data : array-like
        La señal normalizada en el rango [-1, 1].
    """
    data_max = np.max(np.abs(data))
    normalized_data = data / data_max
    return normalized_data

def filter_signal(data, 
                f_sampling = 100, 
                cutoff = 1,
                butterworth_order = 2, 
                btype = 'lowpass'):

    nyquist = f_sampling/2
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(butterworth_order, normal_cutoff, btype)

    data_filtered = pd.DataFrame()
    for _col in data.columns:
        data_filtered[_col] = signal.filtfilt(b, a, data[_col])

    return  data_filtered


def segmentar_datos(emg_data, 
                    postura_data, 
                    repeticion_data, 
                    window_size = None, 
                    step_size = None):
    
    ################################################################## 
    #             Verificacion de prerequisitos basicos              #
    ##################################################################

    if not isinstance(emg_data, pd.DataFrame):
        raise TypeError("Los datos deben ser un dataframe")
    
    if isinstance(postura_data, (list, np.ndarray)):
        postura_data = pd.DataFrame({"label": postura_data})
    
    if isinstance(repeticion_data, (list, np.ndarray)):
        repeticion_data = pd.DataFrame({"rep": repeticion_data})

    #print(emg_data)
    ventanas = []
    ventana = pd.DataFrame()
    for i in range(0, len(emg_data) - window_size + 1, step_size):
        label_window = postura_data.iloc[i:i + window_size]
        rep_window = repeticion_data.iloc[i:i + window_size]
        if label_window.nunique()[0] == 1:
            # label = label_window.mode.iloc[0,0] # se usa la moda
            ventana = pd.concat([emg_data.iloc[i:i + window_size].copy().reset_index(drop=True), 
                                 rep_window.copy().reset_index(drop=True),
                                 label_window.copy().reset_index(drop=True)], 
                                 axis=1)
            ventanas.append(ventana)
            print(f"Indice inicial de la ventana agregada: {i}")
        # print(emg_data.iloc[i:i + window_size])
    print(len(ventanas))
    return ventanas


####################################################################################################
# Funciones de graficado
####################################################################################################

def graficar_medida(medida, 
                    fs = None,
                    columnas = None, 
                    titulo = None, 
                    etiqueta_x = None, 
                    etiqueta_y = None):
    plt.figure(figsize=(20, 5))  # Tamaño del gráfico
    
    # Iterar sobre cada columna en la lista de columnas
    if fs is None:
        t = medida.index
    else:
        t = 1/fs*medida.index

    if not isinstance(medida, pd.DataFrame):
        t = np.arange(0,len(medida))
        if fs is not None:
            t = 1/fs*t
        plt.plot(t,medida)  # Graficar cada columna
    else:
        if (columnas is None):
            columnas = medida.columns      
        for columna in columnas:
            plt.plot(t, medida[columna], label=columna)  # Graficar cada columna

    # Añadir títulos y etiquetas
    if etiqueta_x is None: 
        etiqueta_x = "muestras [n]"
        if fs is not None:
            etiqueta_x = "tiempo [s]"

    if etiqueta_y is None: 
        etiqueta_y = "Amplitud"
    
    plt.title(titulo)
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.legend()  # Añadir la leyenda para distinguir las columnas
    plt.grid(True)  # Añadir cuadrícula
    plt.show()

    """
    graficar_varias_columnas(emgs,
                         columnas = emgs.columns,
                         titulo = "Grafico canales EMG",
                         etiqueta_x="n",
                         etiqueta_y="Amplitud")
    """


def graficar_medida2(medida, 
                     columnas = None, 
                     labels = None,
                     num = 0, 
                     fs = None,
                     titulo=None, 
                     etiqueta_x=None, 
                     etiqueta_y=None):
    [inicio,fin]= indice_numero(labels, num)
    num_puntos = fin - inicio
    ban_end = False
    ban_add_vertical_lines = False
    lim = [0 , 0]
    limites_x = []

    """
    IMPORTANTE: Aun no funciona para graficar en escala de segundos
    """

    # Si se especifica num_puntos, selecciona solo los primeros num_puntos de la Serie
    fig, ax = plt.subplots(figsize=(20, 5))
    
    # Iterar sobre cada columna en la lista de columnas
    


    if num_puntos:
        if (columnas is None):
            columnas = medida.columns      
        for columna in columnas:
            df_col = medida[columna].iloc[inicio:inicio + num_puntos]
            if fs is None:
                t = df_col.index
            else:
                t = 1/fs*df_col.index
            plt.plot(t, df_col, label=columna)  # Graficar cada columna



    cambios_nivel = detectar_cambios_nivel(labels, num)
    # Añadir las bandas verticales sombreadas con los límites proporcionados
    for cambio_nivel in cambios_nivel:
      if(cambio_nivel[1] == 0):
        if ban_end == False:
          lim[0] = cambio_nivel[0]
          ban_end = True
        else:
          lim[1] = cambio_nivel[0]
          ban_end = False
          ax.axvspan(lim[0], lim[1], color='gray', alpha=0.3, label=f'Sombreado entre {lim[0]} y {lim[1]}')

    # Añadir títulos y etiquetas
    if etiqueta_x is None: 
        etiqueta_x = "muestras [n]"
        if fs is not None:
            etiqueta_x = "tiempo [s]"

    if etiqueta_y is None: 
        etiqueta_y = "Amplitud"

    plt.title(titulo)
    plt.xlabel(etiqueta_x)
    plt.ylabel(etiqueta_y)
    plt.grid(True)  # Activa la cuadrícula
    plt.show()

if __name__ == "__main__":
    muestra_mat = scipy.io.loadmat("./S1_A1_E1.mat")
    df_emg = pd.DataFrame(muestra_mat['emg'])
    print(df_emg.shape)
    df_restimulus = pd.DataFrame(muestra_mat['restimulus'])
    print(df_restimulus.shape)
    df_repetition = pd.DataFrame(muestra_mat['rerepetition'])
    print(df_repetition.shape)
    keys_mat_data = list(muestra_mat.keys())
    column_names = keys_mat_data[3:]
    # graficar_medida(df_emg)
    # graficar_medida(df_emg, fs = 100)
    # graficar_medida(df_emg, columnas = [0])    
    # graficar_medida(df_emg, fs = 100, columnas = [0])    
    # graficar_medida(df_emg[0])    
    # graficar_medida(df_emg[0], fs = 100)
    s_filt =  filter_signal(df_emg)  
    # graficar_medida(s_filt, fs = 100)
    # print(detectar_cambios_nivel(df_restimulus[0], 1))
    print(df_emg.head())
    print(s_filt.head())
    # graficar_medida2(s_filt, labels = df_restimulus[0], num = 1, fs = None, titulo="Señales EMG", etiqueta_x=None, etiqueta_y=None)
    # graficar_medida2(s_filt, columnas = [0,1,2], labels = df_restimulus[0], num = 1, fs = None, titulo="Señales EMG", etiqueta_x=None, etiqueta_y=None)
    

