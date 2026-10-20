# TP2 – Reconstrucción 3D

I308 Visión Artificial – Universidad de San Andrés

Abril Diaco, Pilar Otero, Federico Pizarro Dal Maso

Reconstrucción 3D de una zapatilla a partir de capturas estéreo y estimación de su tamaño en milímetros. El pipeline va desde la calibración del sistema estéreo hasta la fusión y optimización de las nubes de puntos con Open3D.

## Entregable principal

La nube de puntos final optimizada está en [`resultados/nube_final.ply`](resultados/nube_final.ply) (PLY binario, ~10 MB). Se puede abrir con [MeshLab](https://www.meshlab.net/).

También se incluye la malla obtenida con Poisson en [`resultados/malla_poisson.ply`](resultados/malla_poisson.ply).

## Estructura del repositorio

```
tp2-vision/
├── data/
│   ├── calib/                    # pares de calibración (left_N.jpg / right_N.jpg), tablero 9x6 esquinas internas
│   ├── captures/                 # 38 pares estéreo de la zapatilla sobre el ChArUco
│   ├── stereo_calibration.pkl    # calibración estéreo generada con calib-stereo
│   └── stereo_maps.pkl           # mapas de rectificación, P1, P2 y Q generados con calib-stereo
├── src/
│   ├── notebook.ipynb            # pipeline completo, con explicaciones y figuras
│   └── utils.py                  # funciones usadas por el notebook
├── resultados/                   # salidas: nube final y malla (.ply)
├── informe/                      # informe en LaTeX
└── requirements.txt
```

## Instalación

Probado con **Python 3.9** en macOS (Apple Silicon). Las versiones exactas están en `requirements.txt`:

| Paquete | Versión |
|---|---|
| numpy | 2.0.2 |
| opencv-python | 4.10.0.84 |
| open3d | 0.18.0 |
| scipy | 1.13.1 |
| matplotlib | 3.9.4 |
| plotly | 7.1.0 |
| nbformat | 5.10.4 |
| ipywidgets | 8.1.9 |
| ipykernel | 6.31.0 |

```bash
git clone https://github.com/PilarOtero/tp2-vision.git
cd tp2-vision

python3 -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Uso

1. Abrir `src/notebook.ipynb` en Jupyter o VS Code, con el entorno anterior como kernel.
2. Ejecutar todas las celdas en orden, desde arriba (*Run All*). El notebook lee los datos con rutas relativas (`../data`), así que tiene que ejecutarse desde la carpeta `src/`.
3. El procesamiento de las 38 capturas tarda unos minutos. Al final se escriben `resultados/nube_final.ply` y `resultados/malla_poisson.ply`.

Las nubes de puntos se visualizan dentro del notebook con plotly (`draw_plotly`), por lo que no se abre ninguna ventana externa.

## Pipeline

| Etapa | Qué se hace | Funciones principales (`utils.py`) |
|---|---|---|
| 1. Calibración y rectificación | Carga de la calibración de `calib-stereo` y rectificación de los pares. Se valida con una calibración propia (`findChessboardCorners`, `calibrateCamera`, `stereoCalibrate`). | `detectar_esquinas`, `calibracion_estereo`, `rectificacion`, `rectificar`, `graficar_rectificacion` |
| 2. Disparidad | SGBM sobre el par rectificado y filtrado de bordes. | `calcular_disparidad`, `filtrar_bordes_disparidad`, `graficar_disparidad` |
| 2. Nube local | `reprojectImageTo3D` con la matriz Q: nube en el sistema de la cámara izquierda rectificada, en mm. | `nube_de_puntos_local` |
| 2. Pose global | Detección del ChArUco (9×13, cuadrado de 42 mm, marcador de 31,5 mm, `DICT_5X5_100`) y `solvePnP` + `Rodrigues`. Se descartan las capturas con error de reproyección de la pose mayor a 3 px. | `analizar_charuco`, `estimar_pose_camara`, `transformar_nube_a_mundo_optimizada`, `graficar_poses` |
| 3. Fusión y optimización | Unión de las nubes en el sistema del ChArUco, recorte con una caja orientada, `voxel_down_sample` y `remove_statistical_outlier`. | `crear_nube_open3d`, `crear_caja`, `filtrar_por_caja`, `optimizar_nube` |
| 3. Opcionales | Refinamiento de las poses con ICP, estimación de normales y malla de Poisson. | `alinear_capturas_icp`, `estimar_normales`, `crear_malla_poisson` |
| 5. Medición | Altura de la zapatilla respecto del plano del tablero (Z = 0). | celdas finales del notebook |

## Nota sobre numpy 2 y Open3D

Open3D 0.18 (la última versión disponible para Python 3.9) está compilado para numpy 1.x. Con numpy 2, algunas funciones de Open3D terminan el proceso con un *segmentation fault* cuando reciben arrays de numpy de forma directa. Para que el notebook funcione con numpy 2:

- La caja se construye a partir de sus 8 esquinas con `OrientedBoundingBox.create_from_points` (`crear_caja`), en lugar del constructor `OrientedBoundingBox(centro, R, extension)`.
- Los ejes del sistema global y la caja se dibujan como `LineSet` (`crear_ejes`, `caja_para_graficar`), en lugar de `create_coordinate_frame` y `caja.color`.
- El ICP está implementado con numpy y `scipy.spatial.cKDTree` (`icp_punto_a_plano`, `evaluar_alineacion`), porque `registration_icp` y `evaluate_registration` no funcionan.
- Los colores siempre se pasan a Open3D envueltos en `o3d.utility.Vector3dVector`.

En Mac, numpy 2 también puede mostrar avisos `RuntimeWarning: divide by zero / overflow encountered in matmul` en productos de matrices. Son falsos (un problema conocido de la librería Accelerate) y no afectan los resultados.
