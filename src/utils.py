import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d
from scipy.spatial import cKDTree

# Ejercicio 1 ---------------------------------------------------------------------------------------------------------
def detectar_esquinas(carpeta, esquinas_internas, cuadrado_size_milimetros):
    """
    Detecta y refina las esquinas del tablero en los pares estéreo válidos.

    Parámetros:
        carpeta (str): carpeta con las imágenes left_*.jpg y right_*.jpg.
        esquinas_internas (tuple): cantidad de esquinas internas (columnas, filas).
        cuadrado_size_milimetros (float): lado de cada cuadrado en mm.

    Retorna:
        tuple: puntos 3D del tablero, puntos 2D izquierdos y derechos, y tamaño (ancho, alto) en píxeles.
    """
    # Puntos 3d del tablero en su sistema dentro del checkboard de calibracion (Z = 0 porque es plano)
    cols, rows = esquinas_internas
    posiciones_objetos = np.zeros((rows * cols, 3), np.float32)
    posiciones_objetos[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    posiciones_objetos *= cuadrado_size_milimetros

    # Criterio de terminación para el refinamiento subpíxel, es decir, se define un número máximo de iteraciones y un epsilon (error mínimo) para detener la busqueda de esquina
    criterio_esquinas = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_puntos, puntos_img_izq, puntos_img_der = [], [], []

    for nombre_archivo in sorted(os.listdir(carpeta)):
        if "left_" in nombre_archivo and nombre_archivo.endswith(".jpg"):
            path_l = os.path.join(carpeta, nombre_archivo)
            path_r = path_l.replace("left_", "right_")

            # Cargamos las imagenes en escala de grises porque trabajamos con intensidades
            img_izq = cv2.imread(path_l, cv2.IMREAD_GRAYSCALE)
            img_der = cv2.imread(path_r, cv2.IMREAD_GRAYSCALE)

            # Detectamos las esquinas del tablero de ajedrez en ambas imágenes
            todas_las_esquinas_izq, esquinas_izq = cv2.findChessboardCorners(img_izq, esquinas_internas)
            todas_las_esquinas_der, esquinas_der = cv2.findChessboardCorners(img_der, esquinas_internas)

            if not (todas_las_esquinas_izq and todas_las_esquinas_der):
                print(f"Descartamos {path_l} porque no se detecto el tablero en ambas imagenes")
                continue

            # Refinamiento subpíxel: mejora mucho el error de reproyección
            esquinas_izq = cv2.cornerSubPix(img_izq, esquinas_izq, (11, 11), (-1, -1), criterio_esquinas)
            esquinas_der = cv2.cornerSubPix(img_der, esquinas_der, (11, 11), (-1, -1), criterio_esquinas)

            obj_puntos.append(posiciones_objetos)
            puntos_img_izq.append(esquinas_izq)
            puntos_img_der.append(esquinas_der)

    image_size = img_izq.shape[::-1] 
    print(f"{len(obj_puntos)} pares válidos")
    return obj_puntos, puntos_img_izq, puntos_img_der, image_size

def calibracion_estereo(obj_puntos, puntos_img_izq, puntos_img_der, imagen_size, print_ = False):
    """
    Calibra ambas cámaras y estima la transformación de izquierda a derecha.

    Parámetros:
        obj_puntos (list): coordenadas 3D conocidas del tablero en mm, por captura.
        puntos_img_izq (list): esquinas detectadas en la imagen izquierda, en píxeles.
        puntos_img_der (list): esquinas detectadas en la imagen derecha, en píxeles.
        imagen_size (tuple): tamaño (ancho, alto) de las imágenes en píxeles.
        print_ (bool): indica si se muestran los errores RMS y el baseline.

    Retorna:
        dict: intrínsecos, distorsión, R, T en mm, E, F, tamaño de imagen y errores RMS en píxeles.
    """
    # Calibración de la cámara izquierda, None ya que estimamos K y la distorsion desde 0
    error_izq, K_izq, distorsion_izq, _, _ = cv2.calibrateCamera(obj_puntos, puntos_img_izq, imagen_size, None, None)

    # Calibración de la cámara derecha, None ya que estimamos K y la distorsion desde 0
    error_der, K_der, distorsion_der, _, _ = cv2.calibrateCamera(obj_puntos, puntos_img_der, imagen_size, None, None)

    # Calibración estéreo, tomamos a la camara izquierda como referencia (""centro del universo"") y la derecha como secundaria (se mueve respecto a la izquierda)
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    # Calculamos los parametros intrinsecos utilizando las dos camaras a la vez
    error_estereo, K_izq, distorsion_izq, K_der, distorsion_der, R, T, E, F = cv2.stereoCalibrate(obj_puntos, puntos_img_izq, puntos_img_der, K_izq, distorsion_izq, K_der, distorsion_der, imagen_size, criteria = criterio, flags = cv2.CALIB_USE_INTRINSIC_GUESS)

    if print_:
        print(f'Error RMS camara izquierda: {error_izq:.3f} pixeles')
        print(f'Error RMS camara derecha: {error_der:.3f} pixeles')
        print(f'Error RMS estéreo: {error_estereo:.3f} pixeles')
        print(f'Baseline en cuadrados: {np.linalg.norm(T):.3f} mm')
    
    return {
        "K_izq": K_izq, "distorsion_izq": distorsion_izq,
        "K_der": K_der, "distorsion_der": distorsion_der,
        "img_size": imagen_size,
        "error_der": error_der, "error_izq": error_izq,
        "error_estereo": error_estereo,
        "R": R, "T": T, "E": E, "F": F}

# Como queremos calcular la profundidad para cada pixel de la imagen izzquierda, debemos encontrar el mismo punto en la imagen derecha. 
# Comoe so es costoso, lo mas util es rectificar las imagenes de forma que las lineas epipolares queden horizontales y a la misma altura
def rectificacion(calibracion, alpha = 0):
    """
    Calcula las matrices y los mapas para rectificar un par estéreo.

    Parámetros:
        calibracion (dict): parámetros de ambas cámaras, R, T y tamaño de imagen.
        alpha (float): 0 recorta bordes inválidos; 1 conserva todo el campo original.

    Retorna:
        dict: rotaciones, proyecciones, matriz Q y mapas de rectificación de ambas cámaras.
    """
    K_izq, distorsion_izq = calibracion["K_izq"], calibracion["distorsion_izq"]
    K_der, distorsion_der = calibracion["K_der"], calibracion["distorsion_der"]
    R, T, size_img = calibracion["R"], calibracion["T"], calibracion["img_size"]

    # Rectificación estéreo, obtenemos las matrices de rectificación y las proyecciones de cada cámara
    # Definimos alpha = 0 para que la region de interes ocupe toda la imagen, evitando asi bordes negros
    R_izq, R_der, P_izq, P_der, Q, region_interes_izq, region_interes_der = cv2.stereoRectify(K_izq, distorsion_izq, K_der, distorsion_der, size_img, R, T, alpha = alpha)

    # Mapas de rectificacion del tamaño de la imagen. Para cada (u,v) de la imagen rectificada, dice la posición (x,y) en la original
    mapa_izq_x, mapa_izq_y = cv2.initUndistortRectifyMap(K_izq, distorsion_izq, R_izq, P_izq, size_img, cv2.CV_32FC1)
    mapa_der_x, mapa_der_y = cv2.initUndistortRectifyMap(K_der, distorsion_der, R_der, P_der, size_img, cv2.CV_32FC1)

    return {
        "R_izq": R_izq, "R_der": R_der, "P_izq": P_izq, "P_der": P_der, "Q": Q,
        "mapa_izq": (mapa_izq_x, mapa_izq_y),
        "mapa_der": (mapa_der_x, mapa_der_y)}

def rectificar(imagen_izquierda, imagen_derecha, datos_rectificacion):
    """
    Corrige la distorsión y rectifica las dos imágenes usando los mapas guardados.

    Parámetros:
        imagen_izquierda (np.ndarray): imagen original izquierda.
        imagen_derecha (np.ndarray): imagen original derecha.
        datos_rectificacion (dict): mapas de ambas cámaras, en mapa_izq y mapa_der.

    Retorna:
        tuple: imágenes izquierda y derecha rectificadas, como arrays.
    """
    rectificacion_izq = cv2.remap(imagen_izquierda, *datos_rectificacion["mapa_izq"], cv2.INTER_LINEAR)
    rectificacion_der = cv2.remap(imagen_derecha, *datos_rectificacion["mapa_der"], cv2.INTER_LINEAR)

    return rectificacion_izq, rectificacion_der

# Dibujamos guías horizontales para comparar la altura de puntos correspondientes.
# En el par original no representan las líneas epipolares calculadas.
def dibujar_lineas_epipolares(imagen_izq, imagen_derecha, paso = 50, grosor = 2):
    """
    Une las imágenes y dibuja guías horizontales; no calcula las líneas epipolares.

    Parámetros:
        imagen_izq (np.ndarray): imagen izquierda en BGR.
        imagen_derecha (np.ndarray): imagen derecha en BGR, de la misma altura.
        paso (int): separación entre guías en píxeles.
        grosor (int): ancho de las líneas en píxeles.

    Retorna:
        np.ndarray: par de imágenes en BGR con guías rojas.
    """
    par_img = np.hstack([imagen_izq, imagen_derecha])

    for _, y in enumerate(range(0, par_img.shape[0], paso)):
        cv2.line(par_img, (0, y), (par_img.shape[1], y), (0, 0, 255), grosor)

    return par_img

def graficar_rectificacion(imagen_izq, imagen_der, datos_rectificacion, recorte = None, paso = 70, grosor = 2):
    """
    Compara los pares original y rectificado en gris, con guías horizontales.

    Parámetros:
        imagen_izq (np.ndarray): imagen original izquierda en BGR.
        imagen_der (np.ndarray): imagen original derecha en BGR.
        datos_rectificacion (dict): mapas de rectificación de ambas cámaras.
        recorte (tuple o None): límites (y0, y1, x0, x1) para la visualización.
        paso (int): separación entre guías en píxeles.
        grosor (int): ancho de las guías en píxeles.

    Retorna:
        None: muestra la figura.
    """
    rect_izq, rect_der = rectificar(imagen_izq, imagen_der, datos_rectificacion)
    
    titulos = ["Par original", "Par rectificado"]
    rotulos = [('Izquierda original', 'Derecha original'), ('Izquierda rectificada', 'Derecha rectificada')]
    
    pares = [(imagen_izq, imagen_der), (rect_izq, rect_der)]
    if recorte is not None:
        y0, y1, x0, x1 = recorte
        pares = [(izq[y0:y1, x0:x1], der[y0:y1, x0:x1]) for izq, der in pares]

    # Escala de grises solo para visualizar
    pares = [
        (cv2.cvtColor(cv2.cvtColor(izq, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR, ), 
         cv2.cvtColor(cv2.cvtColor(der, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR),
        )
        for izq, der in pares
    ]

    _, ax = plt.subplots(2, 1, figsize=(16, 9))
    for a, (izq, der), titulo, (rotulo_izq, rotulo_der) in zip(ax, pares, titulos, rotulos):
        a.imshow(cv2.cvtColor(dibujar_lineas_epipolares(izq, der, paso, grosor), cv2.COLOR_BGR2RGB))
        a.set_title(titulo)
        a.axis("off")

        ancho = izq.shape[1]
        for x, texto in [(10, rotulo_izq), (ancho + 10, rotulo_der)]:
            a.text(x, 20, texto, ha = "left", va = "top", color = "white", fontsize = 12, bbox = dict(facecolor = "black", alpha = 0.5, edgecolor = "none"))
    
    plt.tight_layout()
    plt.show()


# Ejercicio 2 ---------------------------------------------------------------------------------------------------------
def calcular_disparidad(imagen_izq_rect, imagen_der_rect, num_disparidades, tam_bloque, uniqueness, speckle_window, speckle_range):
    """
    Calcula la disparidad con SGBM sobre las imágenes convertidas a gris.

    Parámetros:
        imagen_izq_rect (np.ndarray): imagen izquierda rectificada en BGR.
        imagen_der_rect (np.ndarray): imagen derecha rectificada en BGR.
        num_disparidades (int): amplitud de búsqueda, múltiplo de 16, en píxeles.
        tam_bloque (int): lado impar del bloque de comparación, en píxeles.
        uniqueness (int): margen porcentual para aceptar la mejor correspondencia.
        speckle_window (int): tamaño máximo de regiones aisladas a eliminar; 0 desactiva el filtro.
        speckle_range (int): variación de disparidad permitida dentro de esas regiones, en píxeles.

    Retorna:
        np.ndarray: mapa float32 en píxeles; -1 indica disparidades inválidas para el rango usado.
    """
    # Trabajamos en gris para comparar intensidades
    gris_izq = cv2.cvtColor(imagen_izq_rect, cv2.COLOR_BGR2GRAY)
    gris_der = cv2.cvtColor(imagen_der_rect, cv2.COLOR_BGR2GRAY)

    # Configuración inicial tomada del tutorial 6
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disparidades, blockSize=tam_bloque)

    # Penalizaciones adaptadas al tamaño del bloque
    stereo.setP1(8 * tam_bloque**2)
    stereo.setP2(32 * tam_bloque**2)
    stereo.setUniquenessRatio(uniqueness)
    stereo.setSpeckleWindowSize(speckle_window)
    stereo.setSpeckleRange(speckle_range)
    stereo.setDisp12MaxDiff(1)
    stereo.setPreFilterCap(63)

    # OpenCV devuelve la disparidad multiplicada por 16
    # La convertimos a float y a unidades de píxeles
    disparidad = stereo.compute(gris_izq, gris_der).astype(np.float32) / 16.0

    return disparidad

# Definimos el valor maximo de numero de disparidad para que el mismo color en el mapa de disparidad signifique siempre la misma distancia (util al momento de compararlos)
def graficar_disparidad(imagen_izq_rect, disparidad, max_value = 176):
    """
    Muestra la imagen izquierda y la disparidad; los valores no positivos quedan en gris.

    Parámetros:
        imagen_izq_rect (np.ndarray): imagen izquierda rectificada en BGR.
        disparidad (np.ndarray): mapa de disparidad en píxeles.
        max_value (float): límite superior de la escala de colores, en píxeles.

    Retorna:
        None: muestra la figura.
    """
    disparidad_visible = np.ma.masked_less_equal(disparidad, 0)

    colores = plt.get_cmap("inferno").copy()
    colores.set_bad("lightgray")

    fig, ax = plt.subplots(1, 2, figsize = (16, 6), sharex = True, sharey = True)

    ax[0].imshow(cv2.cvtColor(imagen_izq_rect, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen izquierda rectificada")

    mapa = ax[1].imshow(disparidad_visible, cmap = colores, interpolation = "nearest", vmin = 0, vmax = max_value)
    ax[1].set_title("Mapa de disparidad")

    for a in ax:
        a.axis("off")

    fig.colorbar(mapa, ax = ax[1], label = "Disparidad (píxeles)", shrink = 0.75)

    plt.tight_layout()
    plt.show()

def probar_sgbm(imagen_izq_rect, imagen_der_rect, num_disparidades, tam_bloque, uniqueness, speckle_window, speckle_range):
    """
    Prueba una configuración SGBM y muestra el mapa y el porcentaje de disparidades positivas.

    Parámetros:
        imagen_izq_rect (np.ndarray): imagen izquierda rectificada en BGR.
        imagen_der_rect (np.ndarray): imagen derecha rectificada en BGR.
        num_disparidades (int): amplitud de búsqueda, múltiplo de 16, en píxeles.
        tam_bloque (int): lado impar del bloque de comparación, en píxeles.
        uniqueness (int): margen porcentual para aceptar la mejor correspondencia.
        speckle_window (int): tamaño máximo de regiones aisladas a eliminar; 0 desactiva el filtro.
        speckle_range (int): variación de disparidad permitida dentro de esas regiones, en píxeles.

    Retorna:
        None: imprime los parámetros y la cobertura, y muestra la figura.
    """
    parametros_prueba = {
        "num_disparidades": num_disparidades,
        "tam_bloque": tam_bloque,
        "uniqueness": uniqueness,
        "speckle_window": speckle_window,
        "speckle_range": speckle_range,
    }

    disparidad_prueba = calcular_disparidad(imagen_izq_rect, imagen_der_rect, **parametros_prueba)

    porcentaje = 100 * np.mean(disparidad_prueba > 0)
    print(f"Píxeles con disparidad positiva: {porcentaje:.1f}%")
    print("Parámetros probados:", parametros_prueba)

    graficar_disparidad(imagen_izq_rect, disparidad_prueba, max_value = 256)

def filtrar_bordes_disparidad(disparidad, Q, umbral = 8.0, ventana = 5):
    """
    Marca con -1 las disparidades próximas a saltos grandes de profundidad.

    Parámetros:
        disparidad (np.ndarray): mapa de disparidad en píxeles.
        Q (np.ndarray): matriz de reproyección 4x4 con escala en mm.
        umbral (float): diferencia máxima de profundidad en la ventana, en mm.
        ventana (int): lado de la ventana de comparación en píxeles.

    Retorna:
        np.ndarray: copia del mapa con los píxeles descartados marcados como inválidos.
    """
    # En los bordes de los objetos SGBM mezcla el fondo con el frente y da disparidades intermedias: esos pixeles
    # quedan como puntos "flotantes" entre el objeto y lo que esta detras. Los detectamos como pixeles cuya
    # profundidad cambia mucho dentro de una ventana chica (mas de umbral mm entre el mas cercano y el mas lejano)
    validos = disparidad > 0

    # Profundidad de cada pixel a partir de Q: Z = Q[2,3] / (Q[3,2] * d + Q[3,3])
    Z = np.zeros_like(disparidad, dtype = np.float32)
    Z[validos] = Q[2, 3] / (Q[3, 2] * disparidad[validos] + Q[3, 3])

    # Maxima y minima profundidad en la ventana alrededor de cada pixel (dilatacion y erosion).
    # Los pixeles invalidos no deben contar: valen 0 para el maximo y un valor enorme para el minimo
    nucleo = np.ones((ventana, ventana), np.uint8)
    Z_max = cv2.dilate(Z, nucleo)
    Z_min = cv2.erode(np.where(validos, Z, 1e9).astype(np.float32), nucleo)

    # Marcamos como invalidos (disparidad negativa, igual que SGBM) los pixeles en un salto de profundidad
    filtrada = disparidad.copy()
    filtrada[validos & (Z_max - Z_min > umbral)] = -1

    return filtrada

def nube_de_puntos_local(disparidad, imagen_izq_rectificada, Q, prof_max = 1000):
    """
    Reconstruye puntos en la cámara izquierda rectificada y descarta valores inválidos.

    Parámetros:
        disparidad (np.ndarray): mapa de disparidad en píxeles.
        imagen_izq_rectificada (np.ndarray): imagen izquierda rectificada en BGR.
        Q (np.ndarray): matriz de reproyección 4x4 con escala en mm.
        prof_max (float): profundidad máxima admitida, sin incluirla, en mm.

    Retorna:
        tuple: puntos (N, 3) en mm con 0 < Z < prof_max y colores RGB (N, 3) entre 0 y 1.
    """
    # Pasamos cada pixel (x,y,d) a un punto 3D (X,Y,Z) en mm, en el sistema de coordenadas de la camara izquierda rectificada
    # para ello, usamos la matriz Q obtenida en la rectificacion estereo
    puntos_3d = cv2.reprojectImageTo3D(disparidad, Q)
    
    # Descartamos disparidades no positivas, coordenadas no finitas
    # y profundidades fuera del rango de trabajo.
    # Esta máscara no garantiza que las correspondencias sean correctas.
    puntos_validos = (
        np.isfinite(disparidad)
        & (disparidad > 0)
        & np.isfinite(puntos_3d).all(axis=2)
        & (puntos_3d[:, :, 2] > 0)
        & (puntos_3d[:, :, 2] < prof_max)
    )

    # Tenemos un array de forma (n, 3) con una fila por punto una fila por cada canal BGR
    puntos_finales = puntos_3d[puntos_validos] 

    # Como estabamos trabajando con imagenes BGR, pasamos a RGB invirtiendo (porque matplotlib toma siempre el primero como rojo, el segundo como verde y el tercero como azul ) y normalizando porque matplotlib y Open3D esperan colores en el rango [0,1]
    colores = imagen_izq_rectificada[puntos_validos][:,::-1] / 255.0  

    return puntos_finales, colores

def graficar_nube_de_puntos(puntos, colores, limites = None, n_puntos = 1000000, elev = -140, azim = -70, titulo = 'Nube de puntos local en el sistema de coordenadas de la cámara izquierda'):
    """
    Muestra una muestra aleatoria de la nube con igual escala en los tres ejes.

    Parámetros:
        puntos (np.ndarray): coordenadas (N, 3) en mm.
        colores (np.ndarray): colores RGB (N, 3) entre 0 y 1.
        limites (tuple o None): pares (mínimo, máximo) por eje para recortar la vista.
        n_puntos (int): cantidad máxima de puntos a dibujar.
        elev (float): elevación de la vista en grados.
        azim (float): azimut de la vista en grados.
        titulo (str): título de la figura.

    Retorna:
        None: muestra la figura.
    """
    # Recortamos a una región, solo para visualizar
    # limites = ((x_min, x_max), (y_min, y_max), (z_min, z_max)) en mm
    if limites is not None:
        dentro = np.ones(len(puntos), dtype = bool)
        for eje, (minimo, maximo) in enumerate(limites):
            dentro &= (puntos[:, eje] >= minimo) & (puntos[:, eje] <= maximo)
        puntos, colores = puntos[dentro], colores[dentro]
    
    n_muestras = min(n_puntos, len(puntos))
    indices = np.random.choice(len(puntos), n_muestras, replace = False)
    puntos, colores = puntos[indices], colores[indices]

    # matplotlib no ordena por profundidad: dibujamos primero los puntos más lejanos
    # al observador y al final los más cercanos, para que estos queden encima
    e, a = np.radians(elev), np.radians(azim)
    direccion_vista = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    orden = np.argsort(puntos[:, 0] * direccion_vista[0] + puntos[:, 1] * direccion_vista[1] + puntos[:, 2] * direccion_vista[2])
    puntos, colores = puntos[orden], colores[orden]

    fig = plt.figure(figsize = (12, 8))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(puntos[:, 0], puntos[:, 1], puntos[:, 2], c = colores, s = 1, depthshade = False)

    # Misma escala en los tres ejes
    minimo, maximo = puntos.min(axis = 0), puntos.max(axis = 0)
    centro = (minimo + maximo) / 2
    radio = (maximo - minimo).max() / 2
    ax.set_xlim(centro[0] - radio, centro[0] + radio)
    ax.set_ylim(centro[1] - radio, centro[1] + radio)
    ax.set_zlim(centro[2] - radio, centro[2] + radio)
    ax.set_box_aspect((1, 1, 1))
    
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title(titulo)    
    ax.view_init(elev = elev, azim = azim)

    plt.show()

# Ahora, debemos concentrarnos en detectar el objeto de referencia para poder estimar la pose de la camara
def analizar_charuco(cuadrados = (9, 13), square = 42.00, marker = 31.5, diccionario = cv2.aruco.DICT_5X5_100):
    """
    Crea el tablero ChArUco, su versión de marcadores y el detector.

    Parámetros:
        cuadrados (tuple): cantidad de cuadrados (columnas, filas).
        square (float): lado de cada cuadrado en mm.
        marker (float): lado de cada marcador en mm.
        diccionario (int): identificador del diccionario ArUco de OpenCV.

    Retorna:
        tuple: CharucoBoard, Board de marcadores y CharucoDetector.
    """
    dict_ = cv2.aruco.getPredefinedDictionary(diccionario)
    charuco = cv2.aruco.CharucoBoard(cuadrados, square, marker, dict_)

    charuco_marcadores = cv2.aruco.Board(charuco.getObjPoints(), dict_, charuco.getIds())

    parametros_charuco = cv2.aruco.CharucoParameters()
    parametros_charuco.tryRefineMarkers = True
    detector = cv2.aruco.CharucoDetector(charuco, parametros_charuco, cv2.aruco.DetectorParameters())

    return charuco, charuco_marcadores, detector

def estimar_pose_camara(imagen_rectificada, K, charuco, charuco_marcadores, detector, min_esquinas = 10):
    """
    Estima la pose mundo-cámara con ChArUco o, como alternativa, con al menos dos marcadores.

    Parámetros:
        imagen_rectificada (np.ndarray): imagen izquierda rectificada en BGR.
        K (np.ndarray): matriz intrínseca 3x3 de la cámara rectificada.
        charuco (cv2.aruco.CharucoBoard): tablero con dimensiones en mm.
        charuco_marcadores (cv2.aruco.Board): marcadores del mismo tablero.
        detector (cv2.aruco.CharucoDetector): detector configurado para el tablero.
        min_esquinas (int): mínimo de esquinas para usar ChArUco.

    Retorna:
        dict o None: R, vector r en radianes, t en mm, RMS en píxeles, método y número de puntos; None si falla.
    """
    # Convertimos la imagen a escala de grises porque trabajamos con intensidades
    img_gris = cv2.cvtColor(imagen_rectificada, cv2.COLOR_BGR2GRAY)
    esquinas, ids_esquinas, marcadores, ids_marcadores = detector.detectBoard(img_gris)

    # Elegimos las esquinas del ChArUco porque son las más precisas
    if ids_esquinas is not None and len(ids_esquinas) >= min_esquinas:
        puntos_obj, puntos_img = charuco.matchImagePoints(esquinas, ids_esquinas)
        metodo = "charuco"
    elif ids_marcadores is not None and len(ids_marcadores) >= 2:
        puntos_obj, puntos_img = charuco_marcadores.matchImagePoints(marcadores, ids_marcadores)
        metodo = "marcadores"

    else:
        return None

    # En la imagen rectificada tenemos que la K es P_izq[:,:3] porque la ultima columna es la traslacion, que no exste porque la ocnsideramos en el centro del universo. La distorsion es cero porque ya rectificamos
    hay_solucion, r, t = cv2.solvePnP(puntos_obj, puntos_img, K, None)
    if not hay_solucion:
        return None

    # Proyectamos los puntos 3D de referencia a la imagen para calcular el error de reproyección
    proyectados, _ = cv2.projectPoints(puntos_obj, r, t, K, None)
    # Llevamos ambos a forma (n, 2): segun la version de OpenCV pueden venir como (n, 1, 2) o (n, 2),
    # y si las formas no coinciden numpy hace broadcasting y el error da cualquier cosa sin avisar
    proyectados, puntos_img = proyectados.reshape(-1, 2), puntos_img.reshape(-1, 2)
    error_reproyeccion = np.sqrt(np.mean(np.sum((proyectados - puntos_img) ** 2, axis = -1)))

    R, _ = cv2.Rodrigues(r)

    return {"R": R, "r": r, "t": t, "error_reproyeccion": error_reproyeccion,
            "metodo": metodo, "n_puntos": len(puntos_obj)}

def transformar_nube_a_mundo_optimizada(puntos, R, t):
    """
    Lleva los puntos de la cámara al mundo mediante la inversa de la pose.

    Parámetros:
        puntos (np.ndarray): coordenadas (N, 3) en la cámara, en mm.
        R (np.ndarray): rotación 3x3 del mundo a la cámara.
        t (np.ndarray): traslación del mundo a la cámara, en mm.

    Retorna:
        np.ndarray: puntos (N, 3) en el sistema del tablero, en mm.
    """
    # solvePnp nos da X_camara = R. X_mundo + t -> X_mundo = R.T . (X_camara - t) -> hacemos la cuenta para todos los puntos a la vez
    # Con los puntos como filas (n, 3): (R.T . (X_camara - t)).T = (X_camara - t).T . R, asi que alcanza con multiplicar por R a derecha
    with np.errstate(divide = "ignore", over = "ignore", invalid = "ignore"):
        return (puntos - t.reshape(1, 3)) @ R

def graficar_poses(nube, colores, poses, limites = ((-50, 430), (-50, 600)), n_muestras = 100000, largo_eje = 80, elev = 35, azim = -60):
    """
    Muestra la nube global y los centros y ejes ópticos de las cámaras.

    Parámetros:
        nube (np.ndarray): puntos (N, 3) del mundo en mm.
        colores (np.ndarray): colores RGB (N, 3) entre 0 y 1.
        poses (list): diccionarios con R y t del mundo a cada cámara.
        limites (tuple): pares (mínimo, máximo) para los ejes que se recortan, en mm.
        n_muestras (int): cantidad máxima de puntos a dibujar.
        largo_eje (float): longitud dibujada del eje óptico en mm.
        elev (float): elevación de la vista en grados.
        azim (float): azimut de la vista en grados.

    Retorna:
        None: muestra la figura.
    """
    # Recorte a la zona del tablero, solo para visualizar:
    # limites = ((x_min, x_max), (y_min, y_max), (z_min, z_max)) en mm, sistema del ChArUco
    dentro = np.ones(len(nube), dtype = bool)
    for eje, (minimo, maximo) in enumerate(limites):
        dentro &= (nube[:, eje] >= minimo) & (nube[:, eje] <= maximo)
    nube, colores = nube[dentro], colores[dentro]

    # Muestra aleatoria de la nube global
    n_muestras = min(n_muestras, len(nube))
    indices = np.random.choice(len(nube), n_muestras, replace = False)

    fig = plt.figure(figsize = (12, 9))
    ax = fig.add_subplot(projection = "3d")
    ax.scatter(nube[indices, 0], nube[indices, 1], nube[indices, 2], c = colores[indices], s = 0.3, depthshade = False)

    for pose in poses:
        R, t = pose["R"], pose["t"]
        centro = (-R.T @ t).ravel()        # posición de la cámara en el mundo
        eje_optico = R[2, :]               # dirección en la que mira la cámara (su eje Z, en el mundo)
        ax.scatter(*centro, color = "red", s = 40)
        ax.quiver(*centro, *eje_optico, length = largo_eje, color = "red", linewidth = 2)
        ax.text(*centro, str(pose["indice"]), fontsize = 9)

    # Ejes del sistema global (origen en la esquina del ChArUco): X rojo, Y verde, Z azul
    for direccion, color in zip(np.eye(3), ["r", "g", "b"]):
        ax.quiver(0, 0, 0, *direccion, length = 100, color = color, linewidth = 2)

    ax.set_xlim(-300, 700)
    ax.set_ylim(-250, 750)
    ax.set_zlim(-500, 100)
    ax.set_box_aspect((1000, 1000, 600))
    ax.invert_zaxis() # Z del ChArUco apunta hacia la mesa: lo invertimos para ver las cámaras "arriba"
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Poses de las cámaras y nube global (sistema del ChArUco)")
    ax.view_init(elev = elev, azim = azim)
    plt.show()

def draw_geometries(geometries, note = True, **kwargs):
    """
    Muestra geometrías de Open3D en el notebook o en una ventana externa.

    Parámetros:
        geometries (list): nubes, mallas o líneas de Open3D a mostrar.
        note (bool): True usa Plotly; False abre el visor de Open3D.
        **kwargs: opciones adicionales para el visor elegido.

    Retorna:
        None: abre la visualización.
    """
    if note:
        o3d.visualization.draw_plotly(geometries, **kwargs)
    else:
        o3d.visualization.draw_geometries(geometries, **kwargs)


# Ejercicio 3 ---------------------------------------------------------------------------------------------------------
def crear_nube_open3d(puntos, colores):
    """
    Convierte los puntos y colores a una nube de Open3D.

    Parámetros:
        puntos (np.ndarray): coordenadas (N, 3) en mm.
        colores (np.ndarray): colores RGB (N, 3) entre 0 y 1.

    Retorna:
        o3d.geometry.PointCloud: nueva nube con puntos y colores en float64.
    """
    # Open3D trabaja con float64, por eso convertimos (nuestras nubes estaban en float32 para ahorrar memoria)
    nube = o3d.geometry.PointCloud()
    nube.points = o3d.utility.Vector3dVector(np.asarray(puntos, dtype = np.float64))
    nube.colors = o3d.utility.Vector3dVector(np.asarray(colores, dtype = np.float64))

    return nube

def crear_caja(centro, extension, angulo_z:int = 0):
    """
    Construye una caja orientada a partir de sus ocho esquinas.

    Parámetros:
        centro (tuple): posición (X, Y, Z) del centro en mm.
        extension (tuple): dimensiones de la caja en sus tres ejes, en mm.
        angulo_z (float): rotación alrededor de Z en grados.

    Retorna:
        o3d.geometry.OrientedBoundingBox: caja para recortar la región de interés.
    """
    # Caja orientada en el sistema del ChArUco: como la zapatilla está apoyada en diagonal sobre el tablero, alcanza con girarla alrededor de Z
    # centro = (x, y, z) y extension = (largo, ancho, alto) en mm, angulo_z en grados
    a = np.radians(angulo_z)
    R = np.array([[np.cos(a), -np.sin(a), 0],
                  [np.sin(a),  np.cos(a), 0],
                  [0,          0,         1]])

    # Armamos las 8 esquinas: centro +/- la mitad del tamaño en cada eje, rotadas con R.
    # El constructor OrientedBoundingBox(centro, R, extension) y caja.color se caen con la versión de numpy que usamos, por eso usamos create_from_points
    signos = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype = np.float64)
    esquinas = np.asarray(centro, dtype = np.float64) + (signos * np.asarray(extension, dtype = np.float64) / 2) @ R.T

    # Creamos la caja usando la función de Open3D
    caja = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(esquinas))
    return caja

def caja_para_graficar(caja:o3d.geometry.OrientedBoundingBox, color:tuple[float] = (1.0, 0.0, 0.0)) -> o3d.geometry.LineSet:
    """
    Convierte las aristas de la caja en líneas del color indicado.

    Parámetros:
        caja (o3d.geometry.OrientedBoundingBox): caja a representar.
        color (tuple): color RGB entre 0 y 1; rojo por defecto.

    Retorna:
        o3d.geometry.LineSet: aristas de la caja listas para visualizar.
    """
    # Armamos el esqueleto de la caja conectando las 8 esquinas, lo cual "genera" 12 segmentos
    lineas = o3d.geometry.LineSet.create_from_oriented_bounding_box(caja)
    # Asignamos el color rojo (Vector3dVector) a los 12 segmentos
    lineas.colors = o3d.utility.Vector3dVector(np.tile(color, (len(lineas.lines), 1)))

    return lineas

def crear_ejes(largo:int = 100) -> o3d.geometry.LineSet:
    """
    Construye los ejes globales: X rojo, Y verde y Z azul, desde el origen.

    Parámetros:
        largo (float): longitud de cada eje en mm.

    Retorna:
        o3d.geometry.LineSet: tres segmentos con sus colores.
    """
    # (create_coordinate_frame se cae con numpy 2, por eso los armamos a mano)
    ejes = o3d.geometry.LineSet()

    # Definimos los 4 puntos a utilizar: el origen (indice 0), un punto a largo mm sobre X (indice 1), otro sobre Y (indice 2) y otro sobre Z (indice 3)
    ejes.points = o3d.utility.Vector3dVector(np.array([[0, 0, 0], [largo, 0, 0], [0, largo, 0], [0, 0, largo]], dtype = np.float64))
    
    # 3 segmentos, indicados por los indices de los puntos que unen: origen-X, origen-Y, origen-Z
    ejes.lines = o3d.utility.Vector2iVector(np.array([[0, 1], [0, 2], [0, 3]], dtype = np.int32))
    # Definimos un color según el segmento utilizando la Identidad -> (1,0,0) = rojo, (0,1,0) = verde, (0,0,1) = azul
    ejes.colors = o3d.utility.Vector3dVector(np.eye(3))

    return ejes

def filtrar_por_caja(nube:o3d.geometry.PointCloud, caja:o3d.geometry.OrientedBoundingBox) -> o3d.geometry.PointCloud:
    """
    Conserva los puntos dentro de la caja, incluidos sus colores, sin modificar la nube original.

    Parámetros:
        nube (o3d.geometry.PointCloud): nube de puntos a filtrar.
        caja (o3d.geometry.OrientedBoundingBox): caja que delimita la región de interés.

    Retorna:
        o3d.geometry.PointCloud: nueva nube con los puntos del interior de la caja.
    """
    # Lista con las posiciones (indices) de los puntos que caen dentro de la caja
    indices = caja.get_point_indices_within_bounding_box(nube.points)
    # Armamos una nube nueva solo con esos puntos. La nube original no se modifica
    return nube.select_by_index(indices)

# Opcional: refinamiento de las poses con ICP
def refinar_con_icp(nube, referencia, distancia_max = 5.0, tam_voxel = 2.0):
    """
    Ajusta una nube a la referencia con ICP punto a plano y compara la alineación.

    Parámetros:
        nube (o3d.geometry.PointCloud): nube origen, ya ubicada en el sistema global.
        referencia (o3d.geometry.PointCloud): nube destino; se agregan normales si faltan.
        distancia_max (float): límite de distancia entre correspondencias en mm.
        tam_voxel (float): lado del voxel para submuestrear el origen, en mm.

    Retorna:
        dict: transformación T (4x4), fitness y RMSE en mm antes y después del ajuste.
    """
    # Bajamos la resolucion para que ICP sea rapido (la transformacion despues se aplica a la nube completa)
    origen = nube.voxel_down_sample(voxel_size = tam_voxel)

    # Point-to-plane minimiza la distancia de cada punto al plano tangente de su par en la referencia, por eso necesita sus normales
    if not referencia.has_normals():
        referencia.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = 3 * tam_voxel, max_nn = 30))

    # Las funciones de registro de Open3D (registration_icp, evaluate_registration) se caen con numpy 2,
    # asi que implementamos ICP con numpy, usando un KD-tree de scipy para buscar el vecino mas cercano
    puntos_origen = np.asarray(origen.points)
    puntos_ref, normales_ref = np.asarray(referencia.points), np.asarray(referencia.normals)
    arbol = cKDTree(puntos_ref)

    # Calidad de la alineacion antes de ICP: la nube ya esta en el sistema del ChArUco, asi que partimos de la identidad
    fitness_antes, rmse_antes = evaluar_alineacion(puntos_origen, arbol, distancia_max)

    # Solo se emparejan puntos a menos de distancia_max mm, por eso ICP solo corrige desfasajes chicos
    T = icp_punto_a_plano(puntos_origen, puntos_ref, normales_ref, arbol, distancia_max, max_iteraciones = 50)

    puntos_alineados = puntos_origen @ T[:3, :3].T + T[:3, 3]
    fitness, rmse = evaluar_alineacion(puntos_alineados, arbol, distancia_max)

    # fitness: fraccion de puntos que encontraron un par, rmse: distancia media (en mm) entre los pares
    return {"T": T,
            "fitness_antes": fitness_antes, "rmse_antes": rmse_antes,
            "fitness": fitness, "rmse": rmse}

def evaluar_alineacion(puntos, arbol, distancia_max):
    """
    Calcula la proporción de puntos con correspondencia y el RMSE de sus distancias.

    Parámetros:
        puntos (np.ndarray): coordenadas (N, 3) a evaluar, en mm.
        arbol (scipy.spatial.cKDTree): árbol de búsqueda de la nube de referencia.
        distancia_max (float): límite de distancia para aceptar un par, en mm.

    Retorna:
        tuple: fitness y RMSE en mm; devuelve (0.0, 0.0) si no hay correspondencias.
    """
    # Para cada punto buscamos el mas cercano de la referencia; los que estan a mas de distancia_max no tienen par
    distancias, _ = arbol.query(puntos, distance_upper_bound = distancia_max)
    con_par = np.isfinite(distancias)
    if not con_par.any():
        return 0.0, 0.0

    # fitness: fraccion de puntos con par, rmse: raiz del error cuadratico medio de esos pares (mm)
    return con_par.mean(), np.sqrt(np.mean(distancias[con_par] ** 2))

def icp_punto_a_plano(puntos, puntos_ref, normales_ref, arbol, distancia_max, max_iteraciones = 50, tolerancia = 1e-6):
    """
    Estima una transformación rígida con ICP punto a plano usando NumPy y un KD-tree.

    Parámetros:
        puntos (np.ndarray): puntos origen (N, 3) en mm.
        puntos_ref (np.ndarray): puntos de referencia (M, 3) en mm.
        normales_ref (np.ndarray): normales (M, 3) de los puntos de referencia.
        arbol (scipy.spatial.cKDTree): árbol construido con puntos_ref, en el mismo orden.
        distancia_max (float): límite de distancia entre correspondencias en mm.
        max_iteraciones (int): cantidad máxima de ajustes.
        tolerancia (float): umbral de parada sobre la norma del incremento de rotación y traslación.

    Retorna:
        np.ndarray: transformación 4x4 del origen a la referencia, con traslación en mm.
    """
    # Busca la transformacion rigida T (4x4) que lleva "puntos" sobre la superficie de la referencia.
    # En cada iteracion: 1) emparejamos cada punto con el mas cercano de la referencia,
    # 2) calculamos el movimiento que minimiza la distancia de cada punto al plano tangente de su par,
    # 3) aplicamos ese movimiento y repetimos hasta que deja de cambiar
    T = np.eye(4)
    actuales = puntos.copy()

    for _ in range(max_iteraciones):
        # 1) Correspondencias (solo pares a menos de distancia_max mm)
        distancias, indices = arbol.query(actuales, distance_upper_bound = distancia_max)
        con_par = np.isfinite(distancias)
        if con_par.sum() < 6:
            break
        p, q, n = actuales[con_par], puntos_ref[indices[con_par]], normales_ref[indices[con_par]]

        # Centramos para que el sistema este bien condicionado (las coordenadas son de cientos de mm)
        centro = p.mean(axis = 0)
        p_c, q_c = p - centro, q - centro

        # 2) Error punto a plano: ((R p + t - q) · n). Para rotaciones chicas R ≈ I + [w]x, y queda lineal en (w, t):
        #    (p x n) · w + n · t = (q - p) · n  ->  lo resolvemos por cuadrados minimos
        A = np.hstack([np.cross(p_c, n), n])
        b = np.sum((q_c - p_c) * n, axis = 1)
        x, *_ = np.linalg.lstsq(A, b, rcond = None)
        w, t = x[:3], x[3:]

        # Rotacion a partir del vector w (forma de Rodrigues), aplicada alrededor del centro
        R_inc, _ = cv2.Rodrigues(w)
        t_inc = centro + t - R_inc @ centro

        # 3) Aplicamos el incremento y lo acumulamos en T
        actuales = actuales @ R_inc.T + t_inc
        T_inc = np.eye(4)
        T_inc[:3, :3], T_inc[:3, 3] = R_inc, t_inc
        T = T_inc @ T

        if np.linalg.norm(x) < tolerancia:
            break

    return T

def alinear_capturas_icp(nubes, colores, poses, caja, error_max_referencia = 1.5, distancia_max = 5.0, tam_voxel = 2.0, traslacion_max = 15.0, rotacion_max = 5.0, print_ = True):
    """
    Refina las capturas contra una referencia ChArUco y conserva solo las correcciones aceptadas.

    Parámetros:
        nubes (list): arrays de puntos (N, 3) en el sistema global, en mm.
        colores (list): arrays de colores RGB entre 0 y 1, uno por nube.
        poses (list): poses con índice, método y error de reproyección, en el orden de las nubes.
        caja (o3d.geometry.OrientedBoundingBox): región utilizada para el ajuste.
        error_max_referencia (float): límite de RMS en píxeles para formar la referencia ChArUco.
        distancia_max (float): límite de distancia entre correspondencias en mm.
        tam_voxel (float): lado del voxel para el ajuste en mm.
        traslacion_max (float): máxima traslación aceptada en mm.
        rotacion_max (float): máximo giro aceptado en grados.
        print_ (bool): indica si se muestran las métricas de cada captura.

    Retorna:
        tuple: lista de nubes corregidas y lista de métricas con la aceptación de cada ajuste.
    """
    # Recortamos cada captura a la zona del objeto: el fondo no aporta y el tablero solo, al ser un plano, deja a ICP "deslizarse"
    recortes = [filtrar_por_caja(crear_nube_open3d(n, c), caja) for n, c in zip(nubes, colores)]

    # Referencia: unimos las capturas mas confiables (esquinas ChArUco con error de reproyeccion bajo)
    referencia = o3d.geometry.PointCloud()
    for recorte, pose in zip(recortes, poses):
        if pose["metodo"] == "charuco" and pose["error_reproyeccion"] < error_max_referencia:
            referencia += recorte
    referencia = referencia.voxel_down_sample(voxel_size = tam_voxel)

    nubes_corregidas, resultados = [], []
    for nube, recorte, pose in zip(nubes, recortes, poses):
        resultado = refinar_con_icp(recorte, referencia, distancia_max, tam_voxel)

        # Tamaño de la correccion: cuanto se movio la nube y cuanto giro (angulo de la rotacion a partir de su traza)
        T = resultado["T"]
        resultado["traslacion"] = np.linalg.norm(T[:3, 3])
        resultado["rotacion"] = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))

        # Aceptamos la correccion solo si mejora la alineacion (mas puntos con par y a menor distancia) y si es chica:
        # ICP es un refinamiento de la pose de solvePnP, una correccion grande indica que convergio a una solucion equivocada
        mejora = resultado["fitness"] >= resultado["fitness_antes"] and resultado["rmse"] <= resultado["rmse_antes"]
        chica = resultado["traslacion"] <= traslacion_max and resultado["rotacion"] <= rotacion_max
        resultado["aceptada"] = mejora and chica
        if not resultado["aceptada"]:
            T = np.eye(4)

        # Aplicamos la correccion a la nube completa: X' = R . X + t, con los puntos como filas
        nubes_corregidas.append((nube @ T[:3, :3].T + T[:3, 3]).astype(np.float32))
        resultados.append({"indice": pose["indice"], "metodo": pose["metodo"], **resultado})

        if print_:
            print(f"Captura {pose['indice']} ({pose['metodo']}): fitness {resultado['fitness_antes']:.2f} -> {resultado['fitness']:.2f}, "
                  f"rmse {resultado['rmse_antes']:.2f} -> {resultado['rmse']:.2f} mm, "
                  f"correccion {resultado['traslacion']:.1f} mm y {resultado['rotacion']:.2f}°{'' if resultado['aceptada'] else ' (descartada)'}")

    return nubes_corregidas, resultados

def graficar_corte(nubes, caja, ancho = 4.0, n_puntos = 50000):
    """
    Compara cortes centrales de las nubes en las coordenadas de la caja.

    Parámetros:
        nubes (dict): nombres y nubes de Open3D a comparar.
        caja (o3d.geometry.OrientedBoundingBox): referencia para ubicar y orientar el corte.
        ancho (float): espesor de la franja sobre el primer eje de la caja, en mm.
        n_puntos (int): cantidad máxima de puntos a dibujar por nube.

    Retorna:
        None: muestra ancho y altura relativos al centro de la caja.
    """
    # Corte transversal: nos quedamos con una franja fina en el medio de la caja, a lo ancho del objeto,
    # y la miramos de costado. Si las capturas no estan bien alineadas, la superficie se ve "doble"
    fig, ax = plt.subplots(1, len(nubes), figsize = (7 * len(nubes), 5), sharex = True, sharey = True)
    ax = np.atleast_1d(ax)

    for a, (nombre, nube) in zip(ax, nubes.items()):
        # Pasamos los puntos al sistema de la caja: eje 0 a lo largo, eje 1 a lo ancho, eje 2 en altura
        puntos = (np.asarray(nube.points) - caja.center) @ caja.R
        colores = np.asarray(nube.colors)
        franja = np.abs(puntos[:, 0]) < ancho / 2

        indices = np.flatnonzero(franja)
        indices = np.random.choice(indices, min(n_puntos, len(indices)), replace = False)

        a.scatter(puntos[indices, 1], -puntos[indices, 2], c = colores[indices], s = 1)
        a.set_aspect("equal")
        a.set_xlabel("Ancho (mm)")
        a.set_ylabel("Altura (mm, desde el centro de la caja)")
        a.set_title(f"{nombre}: corte de {ancho:.0f} mm")

    plt.tight_layout()
    plt.show()

def optimizar_nube(nube, caja, tam_voxel = 1.0, vecinos = 30, std_ratio = 2.0, print_ = True):
    """
    Recorta la nube, aplica submuestreo por voxels y elimina outliers estadísticos.

    Parámetros:
        nube (o3d.geometry.PointCloud): nube original en mm.
        caja (o3d.geometry.OrientedBoundingBox): región que se conserva.
        tam_voxel (float): lado de cada voxel en mm.
        vecinos (int): cantidad de vecinos del filtro estadístico.
        std_ratio (float): desviaciones estándar permitidas sobre la distancia media global.
        print_ (bool): indica si se muestra la cantidad de puntos de cada etapa.

    Retorna:
        dict: nubes de las etapas Original, Caja, Voxel y Outliers.
    """
    # Guardamos la nube de cada etapa para poder compararlas y reportar los puntos sobrevivientes
    etapas = {"Original": nube}

    # 1. Filtrado espacial: descartamos todo lo que no es el objeto
    etapas["Caja"] = filtrar_por_caja(nube, caja)

    # 2. Submuestreo: un solo punto por voxel (cubo de lado tam_voxel mm), promediando posicion y color.
    # Elimina la redundancia de tener la misma zona vista desde muchas capturas
    etapas["Voxel"] = etapas["Caja"].voxel_down_sample(voxel_size = tam_voxel)

    # 3. Outliers: descartamos los puntos cuya distancia media a sus vecinos es mucho mayor que la del resto
    # (mas de std_ratio desvios estandar por encima del promedio), es decir, los puntos "flotantes"
    etapas["Outliers"], _ = etapas["Voxel"].remove_statistical_outlier(nb_neighbors = vecinos, std_ratio = std_ratio)

    if print_:
        for nombre, etapa in etapas.items():
            print(f"{nombre}: {len(etapa.points):,} puntos ({100 * len(etapa.points) / len(nube.points):.2f}% del original)")

    return etapas

def graficar_etapas(etapas, n_puntos = 100000, elev = 30, azim = -60):
    """
    Compara las etapas de filtrado con una escala común, sin mostrar Original.

    Parámetros:
        etapas (dict): nombres y nubes de Open3D de cada etapa.
        n_puntos (int): cantidad máxima de puntos a dibujar por etapa.
        elev (float): elevación de la vista en grados.
        azim (float): azimut de la vista en grados.

    Retorna:
        None: muestra las figuras con la cantidad total de puntos por etapa.
    """
    # Una vista por etapa (sin la original, que tiene toda la escena), con la misma escala para poder compararlas
    nombres = [nombre for nombre in etapas if nombre != "Original"]
    fig = plt.figure(figsize = (6 * len(nombres), 6))

    # Tomamos los limites de la primera etapa (la mas grande) para todas
    referencia = np.asarray(etapas[nombres[0]].points)
    centro = (referencia.min(axis = 0) + referencia.max(axis = 0)) / 2
    radio = (referencia.max(axis = 0) - referencia.min(axis = 0)).max() / 2

    for i, nombre in enumerate(nombres):
        puntos, colores = np.asarray(etapas[nombre].points), np.asarray(etapas[nombre].colors)

        # Muestra aleatoria, solo para visualizar
        indices = np.random.choice(len(puntos), min(n_puntos, len(puntos)), replace = False)

        ax = fig.add_subplot(1, len(nombres), i + 1, projection = "3d")
        ax.scatter(puntos[indices, 0], puntos[indices, 1], puntos[indices, 2], c = colores[indices], s = 0.5, depthshade = False)

        ax.set_xlim(centro[0] - radio, centro[0] + radio)
        ax.set_ylim(centro[1] - radio, centro[1] + radio)
        ax.set_zlim(centro[2] - radio, centro[2] + radio)
        ax.set_box_aspect((1, 1, 1))
        ax.invert_zaxis()                  # Z del ChArUco apunta hacia la mesa: lo invertimos para ver el objeto "parado"
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        ax.set_zlabel("Z (mm)")
        ax.set_title(f"{nombre}: {len(puntos):,} puntos")
        ax.view_init(elev = elev, azim = azim)

    plt.tight_layout()
    plt.show()

def exportar_ply(nube, ruta):
    """
    Guarda la nube en PLY binario y muestra el tamaño del archivo.

    Parámetros:
        nube (o3d.geometry.PointCloud): nube a exportar.
        ruta (str): ruta del archivo .ply; crea las carpetas que falten.

    Retorna:
        None: escribe el archivo en disco.
    """
    # write_ascii = False -> formato binario, mucho mas liviano que el de texto
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok = True)
    o3d.io.write_point_cloud(ruta, nube, write_ascii = False)
    print(f"Nube guardada en {ruta} ({os.path.getsize(ruta) / 1e6:.1f} MB)")

# Opcional: normales y mallado de superficie con Poisson
def estimar_normales(nube, radio = 6.0, max_vecinos = 30, vecinos_orientacion = 15):
    """
    Estima y orienta las normales; modifica la nube recibida.

    Parámetros:
        nube (o3d.geometry.PointCloud): nube en mm, con la parte superior hacia -Z.
        radio (float): radio de búsqueda de vecinos en mm.
        max_vecinos (int): máximo de vecinos para estimar cada normal.
        vecinos_orientacion (int): vecinos usados para hacer coherentes las orientaciones.

    Retorna:
        o3d.geometry.PointCloud: la misma nube, con normales orientadas hacia -Z en promedio arriba.
    """
    # La normal de cada punto es la direccion perpendicular al plano que mejor ajusta a sus vecinos
    # (los que estan a menos de radio mm, hasta max_vecinos)
    nube.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius = radio, max_nn = max_vecinos))

    # El ajuste del plano no define el sentido de la normal (puede apuntar hacia afuera o hacia adentro del objeto).
    # Las orientamos para que sean coherentes entre vecinos, recorriendo la superficie de punto en punto
    nube.orient_normals_consistent_tangent_plane(vecinos_orientacion)

    # Ese recorrido puede dejarlas todas al reves: la parte de arriba del objeto tiene que mirar hacia arriba,
    # que en el sistema del ChArUco es -Z. Si en promedio miran hacia +Z, las invertimos todas
    puntos, normales = np.asarray(nube.points), np.asarray(nube.normals)
    arriba = puntos[:, 2] < np.percentile(puntos[:, 2], 10)
    if normales[arriba, 2].mean() > 0:
        nube.normals = o3d.utility.Vector3dVector(-normales)

    return nube

def crear_malla_poisson(nube, profundidad = 7, percentil_densidad = 5, caja = None, fraccion_min_componente = 0.01, iteraciones_suavizado = 20, print_ = True):
    """
    Genera una malla Poisson, elimina zonas poco respaldadas y fragmentos, y la suaviza.

    Parámetros:
        nube (o3d.geometry.PointCloud): nube con normales estimadas y orientadas.
        profundidad (int): profundidad máxima del octree; controla el detalle del mallado.
        percentil_densidad (float): percentil bajo el que se eliminan vértices de poca densidad.
        caja (o3d.geometry.OrientedBoundingBox o None): región de recorte opcional.
        fraccion_min_componente (float): fracción mínima de triángulos por componente para conservarla.
        iteraciones_suavizado (int): iteraciones de Taubin; 0 desactiva el suavizado.
        print_ (bool): indica si se muestran los triángulos de cada etapa.

    Retorna:
        o3d.geometry.TriangleMesh: malla filtrada, con normales de los triángulos y vértices.
    """
    # Poisson busca una superficie cerrada cuyas normales coincidan con las de la nube.
    # profundidad: niveles del octree que discretiza el espacio. Cada nivel divide las celdas a la mitad, asi que
    # el lado de la celda es aprox. el tamaño del objeto / 2^profundidad. Con mas profundidad hay mas detalle,
    # pero si la celda es mas chica que el ruido de la nube, la malla copia ese ruido
    malla, densidades = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(nube, depth = profundidad)
    densidades = np.asarray(densidades)
    if print_:
        print(f"Malla de Poisson: {len(malla.triangles):,} triangulos")

    # Como la superficie es cerrada, Poisson "inventa" caras donde no habia puntos (por ejemplo, debajo de la suela).
    # La densidad indica cuantos puntos respaldan cada vertice: eliminamos los de densidad mas baja
    malla.remove_vertices_by_mask(densidades < np.percentile(densidades, percentil_densidad))
    if print_:
        print(f"Sin vertices de baja densidad: {len(malla.triangles):,} triangulos")

    # Recortamos con la misma caja que la nube, para descartar lo que quede por debajo del tablero
    if caja is not None:
        malla = malla.crop(caja)
        if print_:
            print(f"Recorte con la caja: {len(malla.triangles):,} triangulos")

    # Eliminamos los pedazos sueltos: componentes conexas con menos del fraccion_min_componente de los triangulos
    componentes, triangulos_por_componente, _ = malla.cluster_connected_triangles()
    componentes, triangulos_por_componente = np.asarray(componentes), np.asarray(triangulos_por_componente)
    malla.remove_triangles_by_mask(triangulos_por_componente[componentes] < fraccion_min_componente * len(malla.triangles))
    malla.remove_unreferenced_vertices()
    if print_:
        print(f"Sin componentes chicas: {len(malla.triangles):,} triangulos")

    # Suavizado de Taubin: alterna un paso que acerca cada vertice al promedio de sus vecinos y otro que lo aleja un poco,
    # asi suaviza las irregularidades chicas sin "encoger" la malla (como si pasa con el suavizado laplaciano)
    if iteraciones_suavizado > 0:
        malla = malla.filter_smooth_taubin(number_of_iterations = iteraciones_suavizado)

    # Normales de la malla, para que se vea con iluminacion
    malla.compute_vertex_normals()

    return malla

def exportar_malla(malla, ruta):
    """
    Guarda la malla en PLY binario y muestra el tamaño del archivo.

    Parámetros:
        malla (o3d.geometry.TriangleMesh): malla a exportar.
        ruta (str): ruta del archivo .ply; crea las carpetas que falten.

    Retorna:
        None: escribe el archivo en disco.
    """
    # Igual que con la nube, formato binario
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok = True)
    o3d.io.write_triangle_mesh(ruta, malla, write_ascii = False)
    print(f"Malla guardada en {ruta} ({os.path.getsize(ruta) / 1e6:.1f} MB)")

