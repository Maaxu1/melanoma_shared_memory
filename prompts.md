# Historial de Prompts — Clasificador de Melanoma con Memoria Compartida (Kax)

Registro cronológico de las instrucciones utilizadas para diseñar, implementar, depurar y documentar el sistema de inferencia de melanoma con FastAPI, TensorFlow/Keras, Kax y Next.js.

---

## 1. Arquitectura Inicial, Backend y Biblioteca Kax

1. **Prompt:** Definir la arquitectura inicial del sistema usando Python, FastAPI, TensorFlow (`.keras` con entrada `128x128x3`), Kax para memoria compartida en RAM o disco y React para el frontend, incluyendo endpoints por lote con buffer y secuencial sin buffer.
   * **Resultado:** Se estructuró el gestor de memoria compartida (`kax_manager.py`), la API en FastAPI (`main.py`) con `/predict/batch-buffer` y `/predict/batch-stream` (NDJSON), y una interfaz base en React con grid progresivo y vista de detalle.

---

## 2. Desarrollo y Estilizado del Frontend en Next.js

2. **Prompt:** Adaptar el frontend para utilizar Next.js con React.
   * **Resultado:** Se creó el componente cliente `MelanomaClassifier.tsx` (`"use client"`), configurado con `NEXT_PUBLIC_API_URL`, vistas previas locales mediante `ObjectURL` y consumo directo del stream NDJSON.

3. **Prompt:** Ajustar el estilo visual del componente en Next.js utilizando la configuración base de `layout.tsx` y `globals.css`.
   * **Resultado:** Se migraron los estilos en línea a clases utilitarias de Tailwind CSS v4 con tipografía Geist, soporte de modo oscuro, barras de probabilidad y modal accesible con la tecla `Escape`.

---

## 3. Depuración y Portabilidad del Backend

4. **Prompt:** Solucionar el error `PermissionError: [Errno 13] Permission denied: '/home/user'` al iniciar Uvicorn.
   * **Resultado:** Se identificó el uso de una ruta fija inexistente (`/home/user/store`) y se orientó el directorio de almacenamiento hacia una ruta válida del usuario.

5. **Prompt:** Hacer el proyecto reproducible en cualquier equipo sin depender de rutas absolutas de un usuario específico.
   * **Resultado:** Se implementó resolución dinámica de directorios con `pathlib.Path.home()` y respaldo en `tempfile.gettempdir()`, junto con variables de entorno (`.env.example`), rutas relativas al modelo y el script `run.sh`.

6. **Prompt:** Corregir el error `AttributeError: 'bytes' object has no attribute 'encode'` dentro de `kax.putDataToShm`.
   * **Resultado:** Se adaptó la serialización de datos para pasar cadenas compatibles con la firma interna de `Kax`.

7. **Prompt:** Evaluar la refactorización del backend usando `load_res`, `getRes` y `removeShm` en Kax, métricas desglosadas (`write`, `read`, `cleanup`, `total`) y `mp.set_start_method("fork", force=True)`.
   * **Resultado:** Se detectaron problemas de orden en la inicialización de `multiprocessing`, medición de tiempos en el bloque `finally` y manejo de excepciones dentro de `StreamingResponse`.

8. **Prompt:** Aplicar las correcciones propuestas respetando el flujo original de buffer y streaming.
   * **Resultado:** Se movió `mp.set_start_method("fork", force=True)` al inicio de `main.py`, se aislaron los tiempos `t_w`, `t_r` y `t_end` y se añadió manejo por imagen de archivos corruptos dentro del generador NDJSON.

---

## 4. Métricas de Tiempo y Generación de Reportes

9. **Prompt:** Mostrar el tiempo total al finalizar el procesamiento secuencial sin buffer (`/predict/batch-stream`).
   * **Resultado:** Se agregó un evento final `{"done": true, "tiempo_total_ms": ...}` en el backend y su lectura en el frontend para actualizar el indicador de tiempo total del lote.

10. **Prompt:** Analizar por qué el buffer en disco puede resultar más rápido que el buffer en RAM en las pruebas locales.
    * **Resultado:** Se explicó el impacto de la sobrecarga de serialización, las llamadas IPC/red de Kax y el *page cache* del kernel de Linux/WSL2 en archivos recientes.

11. **Prompt:** Agregar un botón "Generar reporte" junto a la etiqueta de tiempo total del lote que muestre `write`, `read` y `cleanup`, además del promedio y la mediana de los tiempos por imagen.
    * **Resultado:** Se incorporó el cálculo estadístico en el frontend y un modal de reporte con vista previa en texto y exportación a `.TXT` y `.JSON`.

12. **Prompt:** Corregir el error de Next.js `Encountered the unstable value new Date() in a Client Component` durante el renderizado.
    * **Resultado:** Se movió la generación de marcas de tiempo (`new Date().toISOString()` y `Date.now()`) fuera de `useMemo` hacia el manejador de descarga `handleDownloadReport`.

13. **Prompt:** Diferenciar los campos del reporte entre el modo con buffer (`write`, `read`, `cleanup` y `total`) y sin buffer (`promedio`, `mediana`, `mínimo` y `máximo`), conservando datos comunes (modo, total de imágenes, peso total en MB, tiempo total del lote y detalle por archivo).
    * **Resultado:** Se actualizaron las tarjetas del modal y las plantillas `.TXT`/`.JSON` para renderizar dinámicamente las métricas correspondientes al modo ejecutado junto con el peso total en MB.

---

## 5. Entorno Python y Documentación

14. **Prompt:** Generar el `README.md` del proyecto.
    * **Resultado:** Se redactó la documentación inicial con arquitectura, estructura de carpetas, configuración de variables de entorno y endpoints.

15. **Prompt:** Revisar compatibilidad de `requirements.txt` y `main.py` con `python3.10` y actualizar la guía de instalación.
    * **Resultado:** Se analizaron las restricciones de versiones *nightly*/recientes frente a Python 3.10 y se documentó el flujo de instalación mediante `requirements.txt`.

16. **Prompt:** Confirmar la disponibilidad del entorno virtual en Python 3.14 tras intentar crear uno con `python3.10`.
    * **Resultado:** Se verificó que todas las dependencias quedaron resueltas en el entorno `.venv` con Python 3.14.

17. **Prompt:** Actualizar el `README.md` especificando Python 3.14 como versión oficial del proyecto y entregarlo en formato Markdown listo para copiar.
    * **Resultado:** Se actualizó la sección de requisitos e instalación para Python 3.14 y se entregó el documento íntegro en bloque Markdown.