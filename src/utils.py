import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import open3d as o3d

## Ejercicio 1 ---------------------------------------------------------------------------------------------------------
def detectar_esquinas(carpeta, esquinas_internas, cuadrado_size_milimetros):
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
        "R": R, "T": T, "E": E, "F": F}

# Como queremos calcular la profundidad para cada pixel de la imagen izzquierda, debemos encontrar el mismo punto en la imagen derecha. 
# Comoe so es costoso, lo mas util es rectificar las imagenes de forma que las lineas epipolares queden horizontales y a la misma altura
def rectificacion(calibracion, alpha = 0):
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
    rectificacion_izq = cv2.remap(imagen_izquierda, *datos_rectificacion["mapa_izq"], cv2.INTER_LINEAR)
    rectificacion_der = cv2.remap(imagen_derecha, *datos_rectificacion["mapa_der"], cv2.INTER_LINEAR)

    return rectificacion_izq, rectificacion_der

# Dibujamos guías horizontales para comparar la altura de puntos correspondientes.
# En el par original no representan las líneas epipolares calculadas.
def dibujar_lineas_epipolares(imagen_izq, imagen_derecha, paso = 50, grosor = 2):
    par_img = np.hstack([imagen_izq, imagen_derecha])

    for _, y in enumerate(range(0, par_img.shape[0], paso)):
        cv2.line(par_img, (0, y), (par_img.shape[1], y), (0, 0, 255), grosor)

    return par_img

def graficar_rectificacion(imagen_izq, imagen_der, datos_rectificacion, recorte = None, paso = 70, grosor = 2):
    rect_izq, rect_der = rectificar(imagen_izq, imagen_der, datos_rectificacion)
    
    titulos = ["Par original", "Par rectificado"]
    rotulos = [('Izquierda original', 'Derecha original'), ('Izquierda rectificada', 'Derecha rectificada')]
    
    pares = [(imagen_izq, imagen_der), (rect_izq, rect_der)]
    if recorte is not None:
        y0, y1, x0, x1 = recorte
        pares = [(izq[y0:y1, x0:x1], der[y0:y1, x0:x1]) for izq, der in pares]

    # Escala de grises solo para visualizar.
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

def nube_de_puntos_local(disparidad, imagen_izq_rectificada, Q, prof_max = 1000):
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
    dict_ = cv2.aruco.getPredefinedDictionary(diccionario)
    charuco = cv2.aruco.CharucoBoard(cuadrados, square, marker, dict_)

    charuco_marcadores = cv2.aruco.Board(charuco.getObjPoints(), dict_, charuco.getIds())

    parametros_charuco = cv2.aruco.CharucoParameters()
    parametros_charuco.tryRefineMarkers = True
    detector = cv2.aruco.CharucoDetector(charuco, parametros_charuco, cv2.aruco.DetectorParameters())

    return charuco, charuco_marcadores, detector

def estimar_pose_camara(imagen_rectificada, K, charuco, charuco_marcadores, detector, min_esquinas = 10):
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

# def transformar_nube_a_mundo(puntos, R, t):
#     # solvePnp nos da X_camara = R. X_mundo + t -> X_mundo = R.T . (X_camara - t)
#     mundo = []
#     for p in puntos:
#         # Definimos los puntos como columnas para poder hacer el producto con R.T
#         columna = p.reshape(3,1)
#         mundo.append(R.T @ (columna - t))
#
#     return np.array(mundo).reshape(-1,3)

def transformar_nube_a_mundo_optimizada(puntos, R, t):
    # Misma cuenta, X_mundo = R.T . (X_camara - t), pero para todos los puntos a la vez.
    # Con los puntos como filas (n, 3): (R.T . (X - t)).T = (X - t).T . R, asi que alcanza con multiplicar por R a derecha
    return (puntos - t.reshape(1, 3)) @ R

def graficar_poses(nube, colores, poses, limites = ((-50, 430), (-50, 600)), n_muestras = 100000, largo_eje = 80, elev = 35, azim = -60):
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
    if note:
        o3d.visualization.draw_plotly(geometries, **kwargs)
    else:
        o3d.visualization.draw_geometries(geometries, **kwargs)


#Ejercicio 3 ---------------------------------------------------------------------------------------------------------
def crear_nube_open3d(puntos, colores):
    # Open3D trabaja con float64, por eso convertimos (nuestras nubes estaban en float32 para ahorrar memoria)
    nube = o3d.geometry.PointCloud()
    nube.points = o3d.utility.Vector3dVector(np.asarray(puntos, dtype = np.float64))
    nube.colors = o3d.utility.Vector3dVector(np.asarray(colores, dtype = np.float64))

    return nube

def crear_caja(centro, extension, angulo_z = 0):
    # Caja orientada en el sistema del ChArUco: como el objeto esta apoyado sobre el tablero, alcanza con girarla alrededor de Z
    # centro = (x, y, z) y extension = (largo, ancho, alto) en mm, angulo_z en grados
    R = o3d.geometry.get_rotation_matrix_from_xyz((0, 0, np.radians(angulo_z)))
    caja = o3d.geometry.OrientedBoundingBox(np.asarray(centro, dtype = np.float64), R, np.asarray(extension, dtype = np.float64))
    caja.color = (1, 0, 0)

    return caja

def filtrar_por_caja(nube, caja):
    # Nos quedamos solo con los puntos que caen dentro de la caja, el resto (mesa, tablero, fondo) se descarta
    indices = caja.get_point_indices_within_bounding_box(nube.points)

    return nube.select_by_index(indices)

def optimizar_nube(nube, caja, tam_voxel = 1.0, vecinos = 30, std_ratio = 2.0, print_ = True):
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
    # write_ascii = False -> formato binario, mucho mas liviano que el de texto
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok = True)
    o3d.io.write_point_cloud(ruta, nube, write_ascii = False)
    print(f"Nube guardada en {ruta} ({os.path.getsize(ruta) / 1e6:.1f} MB)")

