# Clasificador de Melanoma con Memoria Compartida (Kax) y Streaming

Aplicación web para la clasificación de lesiones cutáneas en **melanoma maligno** o **melanoma benigno** mediante una red neuronal convolucional en TensorFlow/Keras (`melanoma_cnn_classifier.keras`), backend en **FastAPI** con integración a **Kax** para memoria compartida y almacenamiento local, y frontend en **Next.js (App Router) + React + Tailwind CSS**.

---

## Estructura del Repositorio

```text
melanoma_shared_memory/
├── backend/
│   ├── kax_manager.py                  # Gestión de memoria compartida (Kax) y staging en disco
│   ├── main.py                         # API FastAPI y endpoints de inferencia con/sin buffer
│   ├── melanoma_cnn_classifier.keras   # Modelo CNN entrenado (entrada 128x128x3)
│   ├── requirements.txt                # Dependencias de Python
│   ├── .env.example                    # Plantilla de variables de entorno
│   └── run.sh                          # Script de arranque del servidor Uvicorn
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   │   └── MelanomaClassifier.tsx  # Interfaz principal, métricas y generación de reportes
│   │   ├── globals.css                 # Estilos globales
│   │   ├── layout.tsx                  # Layout raíz de Next.js
│   │   └── page.tsx                    # Página principal de la aplicación
│   ├── public/                         # Recursos estáticos
│   └── package.json                    # Dependencias y scripts de Node.js
├── ejecuciones/                        # Reportes (.TXT / .JSON) de pruebas con 10 a 1000 imágenes
├── prompts.md                          # Registro de prompts utilizados durante el desarrollo
└── README.md                           # Guía de instalación, arquitectura y uso
```

---

## Arquitectura General y Gestión de Memoria

El proyecto aplica conceptos de jerarquía de memoria, memoria compartida y *buffering* sobre un flujo de inferencia médica:

- **Jerarquía de memoria y localidad:** Las imágenes cargadas pasan de la memoria de aplicación a una capa intermedia de almacenamiento (`buffer-ram` en memoria compartida o `buffer-disk` en sistema de archivos local) antes de apilarse en un tensor contiguo $(N, 128, 128, 3)$ en RAM para aprovechar la localidad espacial y temporal durante la inferencia por lotes en CPU.
- **Memoria compartida (`Kax`):** En modo `ram`, `KaxBufferManager` registra los recursos en memoria compartida mediante `load_res`, los recupera por hash con `getRes` y libera los segmentos al finalizar con `removeShm`.
- **Buffering (Productor-Consumidor):**
  - `POST /predict/batch-buffer`: Desacopla la recepción del lote de la inferencia. Almacena temporalmente todas las imágenes en el buffer (`ram` o `disk`), mide los tiempos de `write`, `read`, `cleanup` y `total`, y ejecuta una única inferencia vectorizada sobre el lote completo.
  - `POST /predict/batch-stream`: Ejecuta el flujo sin buffer compartido (`no-buffer`), procesando, infiriendo y transmitiendo cada imagen una por una en formato **NDJSON** (`StreamingResponse`) junto con un evento final con el tiempo total del lote.
- **Modelo (`.keras`):** Recibe tensores de forma `(128, 128, 3)` normalizados en $[0.0, 1.0]$ y devuelve la probabilidad de **melanoma maligno** en $[0, 1]$. La probabilidad de **melanoma benigno** se calcula como $1 - p_{\text{maligno}}$.
- **Frontend (`Next.js` + `React`):** Permite seleccionar múltiples imágenes de lesiones cutáneas, elegir el modo de ejecución (`buffer` en `ram`/`disk` o `no-buffer`), visualizar las predicciones en un grid progresivo, inspeccionar cada diagnóstico y exportar reportes en `.TXT` y `.JSON`.

---

## Requisitos Previos y Dataset

- **Sistema operativo:** Linux o WSL2 (recomendado Ubuntu). *Nota: No es compatible con Windows nativo porque el backend utiliza `multiprocessing` con método `fork`.*
- **Python:** **3.14** y `pip`.
- **Node.js:** 18+ y `npm` / `pnpm` / `yarn`.
- **Servicios de Kax:** Instancia activa de registro (puerto `5060`), almacenamiento (puerto `5070`) y broker MQTT (puerto `1883`), o configuración equivalente en `.env`.
- **Imágenes de prueba:** Las imágenes de melanoma utilizadas para las pruebas pueden obtenerse desde Kaggle en [Melanoma Skin Cancer Dataset of 10000 Images](https://www.kaggle.com/datasets/hasnainjaved/melanoma-skin-cancer-dataset-of-10000-images).

---

## Instalación y Configuración del Backend

1. **Ubícate en la carpeta `backend` y crea el entorno virtual con Python 3.14:**

   ```bash
   cd backend
   python3.14 -m venv .venv
   source .venv/bin/activate
   ```

2. **Actualiza `pip` e instala las dependencias desde `requirements.txt`:**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. **Configura las variables de entorno:**

   Copia `.env.example` a `.env` y ajusta las rutas o URLs/puertos de Kax según tu entorno (por defecto se usa `~/kax_store` y `melanoma_cnn_classifier.keras` junto a `main.py`):

   ```bash
   cp .env.example .env
   ```

   Ejemplo de `.env`:

   ```env
   # KAX_STORE_DIR=/tmp/kax_store
   # MODEL_PATH=melanoma_cnn_classifier.keras
   KAX_REGISTRY_URL=http://localhost:5060/api
   KAX_STORAGE_URL=http://localhost:5070/api
   KAX_BROKER=localhost
   KAX_PORT=1883
   KAX_WORKERS=1
   KAX_CHUNK_SIZE=2
   ```

4. **Ejecuta el servidor FastAPI:**

   ```bash
   chmod +x run.sh
   ./run.sh
   ```

   El servicio quedará disponible en `http://localhost:8000` (documentación interactiva en `http://localhost:8000/docs`).

---

## Instalación y Configuración del Frontend (Next.js)

1. **Ubícate en la carpeta `frontend` e instala dependencias:**

   ```bash
   cd frontend
   npm install
   ```

2. **Configura la URL del backend en `.env.local`:**

   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Inicia el servidor de desarrollo:**

   ```bash
   npm run dev
   ```

   Abre `http://localhost:3000` en el navegador.

---

## Endpoints del Backend

### `POST /predict/batch-buffer`

- **Parámetros (`multipart/form-data`):**
  - `files`: Lista de archivos de imagen.
  - `buffer_mode`: `"ram"` (por defecto) o `"disk"`.
- **Respuesta (`application/json`):** Incluye `buffer_mode`, `total_imagenes`, `tiempo_total_ms`, `buffer_ms` (`write`, `read`, `cleanup`, `total`) y el arreglo `resultados` con `prob_maligno`, `prob_benigno`, `diagnostico`, `confianza` y `tiempo_ms` (tiempo amortizado por imagen).

### `POST /predict/batch-stream`

- **Parámetros (`multipart/form-data`):**
  - `files`: Lista de archivos de imagen.
- **Respuesta (`application/x-ndjson`):** Emite una línea JSON por cada imagen procesada de forma secuencial (con su `tiempo_ms` individual) y, al concluir, una línea final `{"done": true, "tiempo_total_ms": ...}`.

---

## Reporte en el Frontend

Junto al indicador de **Tiempo total del lote**, el botón **Generar reporte** abre un modal con:

- **Datos comunes:** Modo de ejecución, total de imágenes, peso total de las imágenes (`MB`), tiempo total del lote y detalle por archivo.
- **Con buffer (`buffer-ram` / `buffer-disk`):** Tiempos de `write`, `read`, `cleanup` y `total` del buffer.
- **Sin buffer (`no-buffer`):** Promedio, mediana, mínimo y máximo de tiempo por imagen.
- **Exportación:** Copiar al portapapeles o descargar en formato `.TXT` o `.JSON`.

---

## Notas Técnicas

- **Inferencia en CPU:** TensorFlow bajo Python 3.14 en este entorno ejecuta la inferencia en CPU automáticamente y emite advertencias de GPU al iniciar; esto no afecta la validez de las comparativas entre modos de memoria.
- **Umbral de clasificación:** Se clasifica como **Melanoma maligno** cuando `prob_maligno >= 0.5` y **Melanoma benigno** en caso contrario.