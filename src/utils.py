import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

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

# Dibujamos las lineas epipolares para verificar que la rectificacion fue correcta. Si lo fue, deben quedar horizontalmente paralelas
def dibujar_lineas_epipolares(imagen_izq, imagen_derecha, paso = 50, grosor = 2):
    par_img = np.hstack([imagen_izq, imagen_derecha])

    for _, y in enumerate(range(0, par_img.shape[0], paso)):
        cv2.line(par_img, (0, y), (par_img.shape[1], y), (0, 0, 255), grosor)

    return par_img

def graficar_rectificacion(imagen_izq, imagen_der, datos_rectificacion, recorte = None, paso = 70, grosor = 2):
    rect_izq, rect_der = rectificar(imagen_izq, imagen_der, datos_rectificacion)

    pares = [(imagen_izq, imagen_der), (rect_izq, rect_der)]
    if recorte is not None:
        y0, y1, x0, x1 = recorte
        pares = [(izq[y0:y1, x0:x1], der[y0:y1, x0:x1]) for izq, der in pares]

    # Escala de grises solo para visualizar.
    pares = [
        (cv2.cvtColor(cv2.cvtColor(izq, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR, ), 
         cv2.cvtColor(cv2.cvtColor(der, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR,),
        )
        for izq, der in pares
    ]

    _, ax = plt.subplots(2, 1, figsize=(16, 9))
    for a, (izq, der), titulo in zip(ax, pares, ["Par original", "Par rectificado"]):
        a.imshow(cv2.cvtColor(dibujar_lineas_epipolares(izq, der, paso, grosor), cv2.COLOR_BGR2RGB))
        a.set_title(titulo)
        a.axis("off")
    plt.tight_layout()
    plt.show()


## Ejercicio 2 ---------------------------------------------------------------------------------------------------------
def calcular_disparidad(imagen_izq_rect, imagen_der_rect, num_disparidades=128, tam_bloque=7,):
    # Trabajamos en gris para comparar intensidades
    gris_izq = cv2.cvtColor(imagen_izq_rect, cv2.COLOR_BGR2GRAY)
    gris_der = cv2.cvtColor(imagen_der_rect, cv2.COLOR_BGR2GRAY)

    # Configuración inicial tomada del tutorial 6
    stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disparidades, blockSize=tam_bloque,)

    # Penalizaciones adaptadas al tamaño del bloque
    stereo.setP1(8 * tam_bloque**2)
    stereo.setP2(32 * tam_bloque**2)
    stereo.setUniquenessRatio(7)
    stereo.setSpeckleWindowSize(512)
    stereo.setSpeckleRange(64)
    stereo.setDisp12MaxDiff(1)
    stereo.setPreFilterCap(63)

    # OpenCV devuelve la disparidad multiplicada por 16
    # La convertimos a float y a unidades de píxeles
    disparidad = stereo.compute(gris_izq, gris_der).astype(np.float32) / 16.0

    return disparidad

def graficar_disparidad(imagen_izq_rect, disparidad):
    disparidad_visible = np.ma.masked_less_equal(disparidad, 0)

    colores = plt.get_cmap("inferno").copy()
    colores.set_bad("lightgray")

    fig, ax = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

    ax[0].imshow(cv2.cvtColor(imagen_izq_rect, cv2.COLOR_BGR2RGB))
    ax[0].set_title("Imagen izquierda rectificada")

    mapa = ax[1].imshow(disparidad_visible, cmap=colores, interpolation="nearest")
    ax[1].set_title("Mapa de disparidad")

    for a in ax:
        a.axis("off")

    fig.colorbar(mapa, ax=ax[1], label="Disparidad (píxeles)", shrink=0.75)

    plt.tight_layout()
    plt.show()