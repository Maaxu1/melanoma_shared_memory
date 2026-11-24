"use client";

import React, { useState, useEffect, useRef, useMemo } from "react";

interface PredictionItem {
  id: string;
  filename: string;
  mode: string;
  prob_maligno: number;
  prob_benigno: number;
  diagnostico: string;
  confianza: number;
  tiempo_ms: number;
  indice?: number;
  previewUrl: string;
}

interface BufferMetricsMs {
  write: number;
  read: number;
  cleanup: number;
  total: number;
}

interface ImageTimeStats {
  mean: number | null;
  median: number | null;
  min: number | null;
  max: number | null;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

function computeImageStats(values: number[]): ImageTimeStats {
  if (!values.length) {
    return { mean: null, median: null, min: null, max: null };
  }

  const sum = values.reduce((acc, v) => acc + v, 0);
  const mean = Number((sum / values.length).toFixed(2));

  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const medianRaw =
    sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
  const median = Number(medianRaw.toFixed(2));
  const min = Number(sorted[0].toFixed(2));
  const max = Number(sorted[sorted.length - 1].toFixed(2));

  return { mean, median, min, max };
}

export default function MelanomaClassifier() {
  const [selectedFiles, setSelectedFiles] = useState<File[]>([]);
  const [previews, setPreviews] = useState<Record<string, string>>({});
  const previewUrls = useRef<string[]>([]);
  const [executionMode, setExecutionMode] = useState<"buffer" | "no-buffer">("buffer");
  const [bufferStorage, setBufferStorage] = useState<"ram" | "disk">("ram");
  const [predictions, setPredictions] = useState<PredictionItem[]>([]);
  const [selectedCard, setSelectedCard] = useState<PredictionItem | null>(null);
  const [batchTotalMs, setBatchTotalMs] = useState<number | null>(null);
  const [bufferMetrics, setBufferMetrics] = useState<BufferMetricsMs | null>(null);
  const [showReportModal, setShowReportModal] = useState(false);
  const [copiedReport, setCopiedReport] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");

  useEffect(() => {
    return () => {
      previewUrls.current.forEach((url) => URL.revokeObjectURL(url));
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setSelectedCard(null);
        setShowReportModal(false);
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);

  const totalWeightMb = useMemo(() => {
    if (!selectedFiles.length) return 0;
    const totalBytes = selectedFiles.reduce((acc, file) => acc + file.size, 0);
    return Number((totalBytes / (1024 * 1024)).toFixed(2));
  }, [selectedFiles]);

  const imageStats = useMemo(() => {
    const times = predictions
      .map((p) => p.tiempo_ms)
      .filter((t): t is number => typeof t === "number" && !Number.isNaN(t));
    return computeImageStats(times);
  }, [predictions]);

  const isBufferReport = executionMode === "buffer";

  const reportData = useMemo(() => {
    const activeModeLabel = isBufferReport ? `buffer-${bufferStorage}` : "no-buffer";

    const modeSpecificTextLines = isBufferReport
      ? [
          "TIEMPOS DE BUFFER (ms)",
          `  write   : ${bufferMetrics ? `${bufferMetrics.write} ms` : "N/A"}`,
          `  read    : ${bufferMetrics ? `${bufferMetrics.read} ms` : "N/A"}`,
          `  cleanup : ${bufferMetrics ? `${bufferMetrics.cleanup} ms` : "N/A"}`,
          `  total   : ${bufferMetrics ? `${bufferMetrics.total} ms` : "N/A"}`,
        ]
      : [
          "ESTADÍSTICAS DE TIEMPO POR IMAGEN (ms)",
          `  Promedio : ${imageStats.mean !== null ? `${imageStats.mean} ms` : "N/A"}`,
          `  Mediana  : ${imageStats.median !== null ? `${imageStats.median} ms` : "N/A"}`,
          `  Mínimo   : ${imageStats.min !== null ? `${imageStats.min} ms` : "N/A"}`,
          `  Máximo   : ${imageStats.max !== null ? `${imageStats.max} ms` : "N/A"}`,
        ];

    const textReport = [
      "REPORTE DE INFERENCIA — CLASIFICADOR DE MELANOMA (128x128)",
      "============================================================",
      `Modo de ejecución : ${activeModeLabel}`,
      `Total de imágenes : ${predictions.length}`,
      `Peso total (MB)   : ${totalWeightMb} MB`,
      `Tiempo total lote : ${batchTotalMs !== null ? `${batchTotalMs} ms` : "N/A"}`,
      "",
      ...modeSpecificTextLines,
      "",
      "DETALLE POR ARCHIVO",
      ...predictions.map(
        (item, idx) =>
          `  ${idx + 1}. ${item.filename} | ${item.diagnostico} | Maligno: ${(item.prob_maligno * 100).toFixed(2)}% | Benigno: ${(item.prob_benigno * 100).toFixed(2)}% | ${item.tiempo_ms} ms`
      ),
    ].join("\n");

    const jsonReport = {
      execution_mode: activeModeLabel,
      total_imagenes: predictions.length,
      peso_total_mb: totalWeightMb,
      tiempo_total_lote_ms: batchTotalMs,
      ...(isBufferReport
        ? {
            buffer_ms: bufferMetrics,
          }
        : {
            tiempos_imagenes_ms: {
              promedio: imageStats.mean,
              mediana: imageStats.median,
              minimo: imageStats.min,
              maximo: imageStats.max,
            },
          }),
      resultados: predictions.map(({ previewUrl: _omit, ...rest }) => rest),
    };

    return { textReport, jsonReport };
  }, [
    isBufferReport,
    bufferStorage,
    predictions,
    totalWeightMb,
    batchTotalMs,
    bufferMetrics,
    imageStats,
  ]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files || []);
    previewUrls.current.forEach((url) => URL.revokeObjectURL(url));

    const objectUrls: Record<string, string> = {};
    const nextUrls: string[] = [];
    files.forEach((file, index) => {
      const url = URL.createObjectURL(file);
      objectUrls[`${file.name}-${index}`] = url;
      nextUrls.push(url);
    });

    previewUrls.current = nextUrls;
    setPreviews(objectUrls);
    setSelectedFiles(files);
    setPredictions([]);
    setSelectedCard(null);
    setBatchTotalMs(null);
    setBufferMetrics(null);
    setCopiedReport(false);
    setErrorMsg("");
  };

  const handleProcessBatch = async () => {
    if (selectedFiles.length === 0) {
      setErrorMsg("Selecciona al menos una imagen desde tu explorador.");
      return;
    }

    setLoading(true);
    setPredictions([]);
    setSelectedCard(null);
    setBatchTotalMs(null);
    setBufferMetrics(null);
    setCopiedReport(false);
    setErrorMsg("");

    const formData = new FormData();
    selectedFiles.forEach((file) => formData.append("files", file));

    try {
      if (executionMode === "buffer") {
        formData.append("buffer_mode", bufferStorage);
        const res = await fetch(`${API_BASE}/predict/batch-buffer`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || "Error al procesar el lote con buffer.");
        }

        const data = await res.json();
        setBatchTotalMs(data.tiempo_total_ms ?? null);
        setBufferMetrics(data.buffer_ms ?? null);

        const enriched: PredictionItem[] = (data.resultados || []).map(
          (item: Omit<PredictionItem, "id" | "previewUrl">, idx: number) => ({
            ...item,
            id: `${item.filename}-${idx}`,
            previewUrl: previews[`${item.filename}-${idx}`] || "",
          })
        );
        setPredictions(enriched);
      } else {
        const streamStart = performance.now();
        let receivedTotalMs = false;

        const res = await fetch(`${API_BASE}/predict/batch-stream`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok || !res.body) {
          throw new Error("Error en el procesamiento secuencial sin buffer.");
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let buffer = "";
        let itemCounter = 0;

        while (true) {
          const { value, done } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (!line.trim()) continue;
            const parsed = JSON.parse(line);

            if (parsed.done) {
              const totalMs =
                typeof parsed.tiempo_total_ms === "number"
                  ? parsed.tiempo_total_ms
                  : Number((performance.now() - streamStart).toFixed(2));
              setBatchTotalMs(totalMs);
              receivedTotalMs = true;
              continue;
            }

            if (parsed.error) {
              continue;
            }

            const currentIdx = parsed.indice ?? itemCounter;
            itemCounter += 1;

            const newCard: PredictionItem = {
              ...parsed,
              id: `${parsed.filename}-${currentIdx}`,
              previewUrl: previews[`${parsed.filename}-${currentIdx}`] || "",
            };

            setPredictions((prev) => [...prev, newCard]);
          }
        }

        if (!receivedTotalMs) {
          setBatchTotalMs(Number((performance.now() - streamStart).toFixed(2)));
        }
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Ocurrió un error inesperado.";
      setErrorMsg(message);
    } finally {
      setLoading(false);
    }
  };

  const handleCopyReport = async () => {
    try {
      await navigator.clipboard.writeText(reportData.textReport);
      setCopiedReport(true);
      setTimeout(() => setCopiedReport(false), 2000);
    } catch {
      setErrorMsg("No se pudo copiar el reporte al portapapeles.");
    }
  };

  const handleDownloadReport = (format: "txt" | "json") => {
    const payload =
      format === "json"
        ? { generated_at: new Date().toISOString(), ...reportData.jsonReport }
        : null;

    const content =
      format === "txt"
        ? reportData.textReport
        : JSON.stringify(payload, null, 2);
    const mime =
      format === "txt"
        ? "text/plain;charset=utf-8"
        : "application/json;charset=utf-8";
    const blob = new Blob([content], { type: mime });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `reporte-melanoma-${Date.now()}.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const canGenerateReport = predictions.length > 0 || batchTotalMs !== null;

  return (
    <main className="mx-auto w-full max-w-6xl px-4 py-8 sm:px-6 lg:px-8 font-(family-name:--font-geist-sans)">
      {/* Encabezado */}
      <header className="mb-7 flex flex-col gap-3 border-b border-neutral-200 dark:border-neutral-800 pb-5 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <span className="inline-flex items-center rounded-full bg-neutral-100 dark:bg-neutral-800/80 px-3 py-1 text-xs font-medium text-neutral-700 dark:text-neutral-300">
            TensorFlow · Kax SHM · 128×128
          </span>
          <h1 className="mt-2 text-2xl sm:text-3xl font-semibold tracking-tight">
            Clasificador de Melanoma
          </h1>
          <p className="mt-1 text-sm text-neutral-600 dark:text-neutral-400">
            Procesamiento por lotes con memoria compartida (RAM o disco) o flujo secuencial 1 a 1.
          </p>
        </div>

        {/* Etiqueta de tiempo total del lote + botón generar reporte */}
        <div className="flex flex-wrap items-center gap-2">
          {batchTotalMs !== null && (
            <div className="rounded-lg border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-900 px-3.5 py-2 text-xs text-neutral-600 dark:text-neutral-400">
              Tiempo total del lote:{" "}
              <span className="font-mono font-semibold text-neutral-900 dark:text-neutral-100">
                {batchTotalMs} ms
              </span>
            </div>
          )}

          <button
            type="button"
            onClick={() => setShowReportModal(true)}
            disabled={!canGenerateReport || loading}
            className="inline-flex items-center gap-1.5 rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-900 px-3.5 py-2 text-xs font-semibold text-neutral-800 dark:text-neutral-200 shadow-2xs transition hover:bg-neutral-100 dark:hover:bg-neutral-800 disabled:cursor-not-allowed disabled:opacity-45"
          >
            Generar reporte
          </button>
        </div>
      </header>

      {/* Panel de controles */}
      <section className="mb-6 rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50/80 dark:bg-neutral-900/60 p-4 shadow-2xs">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap sm:items-center">
            <label className="group relative inline-flex cursor-pointer items-center gap-2 rounded-lg border border-dashed border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3.5 py-2 text-sm text-neutral-700 dark:text-neutral-300 transition hover:border-neutral-400 dark:hover:border-neutral-600">
              <span className="font-medium">Seleccionar imágenes</span>
              <span className="rounded bg-neutral-100 dark:bg-neutral-800 px-1.5 py-0.5 font-mono text-xs text-neutral-600 dark:text-neutral-400">
                {selectedFiles.length}
              </span>
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={handleFileChange}
                className="sr-only"
              />
            </label>

            <select
              value={executionMode}
              onChange={(e) => setExecutionMode(e.target.value as "buffer" | "no-buffer")}
              aria-label="Modo de procesamiento"
              className="rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm font-medium text-neutral-800 dark:text-neutral-200 focus:border-neutral-500 focus:outline-none"
            >
              <option value="buffer">Lote con Buffer (Kax)</option>
              <option value="no-buffer">Sin Buffer (1 a 1)</option>
            </select>

            {executionMode === "buffer" && (
              <select
                value={bufferStorage}
                onChange={(e) => setBufferStorage(e.target.value as "ram" | "disk")}
                aria-label="Tipo de buffer"
                className="rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-950 px-3 py-2 text-sm font-medium text-neutral-800 dark:text-neutral-200 focus:border-neutral-500 focus:outline-none"
              >
                <option value="ram">Buffer en RAM</option>
                <option value="disk">Buffer en Disco</option>
              </select>
            )}
          </div>

          <button
            type="button"
            onClick={handleProcessBatch}
            disabled={loading || selectedFiles.length === 0}
            className="inline-flex items-center justify-center rounded-lg bg-neutral-900 dark:bg-neutral-100 px-4 py-2 text-sm font-medium text-white dark:text-neutral-950 transition hover:bg-neutral-800 dark:hover:bg-neutral-200 disabled:cursor-not-allowed disabled:opacity-45"
          >
            {loading
              ? `Procesando (${predictions.length}/${selectedFiles.length})...`
              : `Analizar ${selectedFiles.length} imagen(es)`}
          </button>
        </div>

        {selectedFiles.length > 0 && (
          <p className="mt-3 truncate text-xs text-neutral-500 dark:text-neutral-400">
            Archivos listos ({totalWeightMb} MB): {selectedFiles.map((f) => f.name).join(", ")}
          </p>
        )}
      </section>

      {/* Mensaje de error */}
      {errorMsg && (
        <div
          role="alert"
          className="mb-6 rounded-xl border border-red-300 dark:border-red-900/70 bg-red-50 dark:bg-red-950/50 px-4 py-3 text-sm text-red-800 dark:text-red-200"
        >
          {errorMsg}
        </div>
      )}

      {/* Estado vacío */}
      {predictions.length === 0 && !loading && (
        <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-white/60 dark:bg-neutral-900/40 px-6 py-14 text-center">
          <p className="text-sm font-medium text-neutral-700 dark:text-neutral-300">
            Aún no hay predicciones en el grid
          </p>
          <p className="mt-1 text-xs text-neutral-500 dark:text-neutral-400">
            Carga una o más imágenes, elige el modo de ejecución y haz clic en analizar.
          </p>
        </div>
      )}

      {/* Grid progresivo de predicciones */}
      {predictions.length > 0 && (
        <section
          aria-label="Resultados de predicción"
          className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4"
        >
          {predictions.map((item) => {
            const isMalignant = item.prob_maligno >= 0.5;
            const malignantPct = (item.prob_maligno * 100).toFixed(2);
            const benignPct = (item.prob_benigno * 100).toFixed(2);

            return (
              <article
                key={item.id}
                onClick={() => setSelectedCard(item)}
                className={`group flex cursor-pointer flex-col overflow-hidden rounded-xl border bg-white dark:bg-neutral-900 shadow-2xs transition duration-150 hover:-translate-y-0.5 hover:shadow-md ${
                  isMalignant
                    ? "border-red-300 dark:border-red-800/80 hover:border-red-400"
                    : "border-emerald-300 dark:border-emerald-800/80 hover:border-emerald-400"
                }`}
              >
                <div className="relative h-40 w-full overflow-hidden bg-neutral-100 dark:bg-neutral-800">
                  {item.previewUrl ? (
                    // eslint-disable-next-line @next/next/no-img-element
                    <img
                      src={item.previewUrl}
                      alt={item.filename}
                      className="h-full w-full object-cover transition duration-200 group-hover:scale-[1.02]"
                    />
                  ) : (
                    <div className="flex h-full items-center justify-center text-xs text-neutral-400">
                      Sin vista previa
                    </div>
                  )}

                  <span
                    className={`absolute top-2.5 right-2.5 rounded-full px-2.5 py-0.5 text-[11px] font-semibold backdrop-blur-xs ${
                      isMalignant
                        ? "bg-red-600/90 text-white"
                        : "bg-emerald-600/90 text-white"
                    }`}
                  >
                    {isMalignant ? "Maligno" : "Benigno"}
                  </span>
                </div>

                <div className="flex flex-1 flex-col justify-between p-3.5">
                  <div>
                    <p
                      title={item.filename}
                      className="truncate text-sm font-semibold text-neutral-900 dark:text-neutral-100"
                    >
                      {item.filename}
                    </p>
                    <p
                      className={`mt-1 text-xs font-semibold ${
                        isMalignant
                          ? "text-red-600 dark:text-red-400"
                          : "text-emerald-700 dark:text-emerald-400"
                      }`}
                    >
                      {item.diagnostico}
                    </p>

                    <div className="mt-3 space-y-1.5 text-xs font-(family-name:--font-geist-mono)">
                      <div className="flex justify-between text-neutral-700 dark:text-neutral-300">
                        <span>Maligno</span>
                        <span className="font-semibold">{malignantPct}%</span>
                      </div>
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-neutral-200 dark:bg-neutral-800">
                        <div
                          className={`h-full rounded-full ${
                            isMalignant ? "bg-red-500" : "bg-amber-500"
                          }`}
                          style={{ width: `${Math.min(100, Math.max(0, item.prob_maligno * 100))}%` }}
                        />
                      </div>

                      <div className="flex justify-between pt-0.5 text-neutral-600 dark:text-neutral-400">
                        <span>Benigno</span>
                        <span>{benignPct}%</span>
                      </div>
                    </div>
                  </div>

                  <div className="mt-3.5 flex items-center justify-between border-t border-neutral-100 dark:border-neutral-800 pt-2.5 text-[11px] text-neutral-500 dark:text-neutral-400">
                    <span className="font-mono">{item.tiempo_ms} ms</span>
                    <span className="font-medium text-neutral-700 dark:text-neutral-300 group-hover:underline">
                      Ver detalle →
                    </span>
                  </div>
                </div>
              </article>
            );
          })}
        </section>
      )}

      {/* Modal de reporte */}
      {showReportModal && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="report-modal-title"
          onClick={() => setShowReportModal(false)}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-[2px]"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-2xl rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 shadow-2xl"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 id="report-modal-title" className="text-lg font-semibold">
                  Reporte de tiempos del lote
                </h2>
                <p className="mt-0.5 text-xs text-neutral-500 dark:text-neutral-400">
                  {isBufferReport
                    ? "Métricas de buffer (write, read, cleanup y total) y resumen del lote."
                    : "Estadísticos de tiempo por imagen (promedio, mediana, mínimo y máximo) y resumen del lote."}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setShowReportModal(false)}
                aria-label="Cerrar reporte"
                className="rounded-lg p-1.5 text-neutral-500 hover:bg-neutral-100 dark:hover:bg-neutral-800 hover:text-neutral-900 dark:hover:text-neutral-100"
              >
                ✕
              </button>
            </div>

            {/* Datos comunes */}
            <div className="mt-4">
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
                Resumen del lote
              </h3>
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 font-(family-name:--font-geist-mono)">
                <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                  <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                    Modo ejecución
                  </div>
                  <div className="mt-1 truncate text-xs font-semibold">
                    {isBufferReport ? `buffer-${bufferStorage}` : "no-buffer"}
                  </div>
                </div>

                <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                  <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                    Total imágenes
                  </div>
                  <div className="mt-1 text-sm font-semibold">{predictions.length}</div>
                </div>

                <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                  <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                    Peso total
                  </div>
                  <div className="mt-1 text-sm font-semibold">{totalWeightMb} MB</div>
                </div>

                <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                  <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                    Tiempo total lote
                  </div>
                  <div className="mt-1 text-sm font-semibold">
                    {batchTotalMs !== null ? `${batchTotalMs} ms` : "N/A"}
                  </div>
                </div>
              </div>
            </div>

            {/* Métricas específicas según modo */}
            <div className="mt-4">
              <h3 className="mb-2 text-xs font-semibold uppercase tracking-wider text-neutral-500 dark:text-neutral-400">
                {isBufferReport
                  ? "Métricas del buffer"
                  : "Estadísticos de tiempo por imagen"}
              </h3>

              {isBufferReport ? (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 font-(family-name:--font-geist-mono)">
                  <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                    <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                      Write
                    </div>
                    <div className="mt-1 text-sm font-semibold">
                      {bufferMetrics ? `${bufferMetrics.write} ms` : "N/A"}
                    </div>
                  </div>

                  <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                    <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                      Read
                    </div>
                    <div className="mt-1 text-sm font-semibold">
                      {bufferMetrics ? `${bufferMetrics.read} ms` : "N/A"}
                    </div>
                  </div>

                  <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                    <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                      Cleanup
                    </div>
                    <div className="mt-1 text-sm font-semibold">
                      {bufferMetrics ? `${bufferMetrics.cleanup} ms` : "N/A"}
                    </div>
                  </div>

                  <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                    <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                      Total buffer
                    </div>
                    <div className="mt-1 text-sm font-semibold">
                      {bufferMetrics ? `${bufferMetrics.total} ms` : "N/A"}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 font-(family-name:--font-geist-mono)">
                  <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                    <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                      Promedio
                    </div>
                    <div className="mt-1 text-sm font-semibold">
                      {imageStats.mean !== null ? `${imageStats.mean} ms` : "N/A"}
                    </div>
                  </div>

                  <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                    <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                      Mediana
                    </div>
                    <div className="mt-1 text-sm font-semibold">
                      {imageStats.median !== null ? `${imageStats.median} ms` : "N/A"}
                    </div>
                  </div>

                  <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                    <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                      Mínimo
                    </div>
                    <div className="mt-1 text-sm font-semibold">
                      {imageStats.min !== null ? `${imageStats.min} ms` : "N/A"}
                    </div>
                  </div>

                  <div className="rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-50 dark:bg-neutral-950 p-3">
                    <div className="font-sans text-[11px] text-neutral-500 dark:text-neutral-400">
                      Máximo
                    </div>
                    <div className="mt-1 text-sm font-semibold">
                      {imageStats.max !== null ? `${imageStats.max} ms` : "N/A"}
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Vista previa del reporte en texto */}
            <pre className="mt-4 max-h-60 overflow-auto rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-950 p-3.5 font-(family-name:--font-geist-mono) text-xs text-neutral-200">
              {reportData.textReport}
            </pre>

            <div className="mt-5 flex flex-wrap items-center justify-between gap-2">
              <div className="flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={handleCopyReport}
                  className="rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 px-3.5 py-2 text-xs font-semibold text-neutral-800 dark:text-neutral-100 transition hover:bg-neutral-100 dark:hover:bg-neutral-700"
                >
                  {copiedReport ? "¡Copiado!" : "Copiar texto"}
                </button>
                <button
                  type="button"
                  onClick={() => handleDownloadReport("txt")}
                  className="rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 px-3.5 py-2 text-xs font-semibold text-neutral-800 dark:text-neutral-100 transition hover:bg-neutral-100 dark:hover:bg-neutral-700"
                >
                  Descargar .TXT
                </button>
                <button
                  type="button"
                  onClick={() => handleDownloadReport("json")}
                  className="rounded-lg border border-neutral-300 dark:border-neutral-700 bg-white dark:bg-neutral-800 px-3.5 py-2 text-xs font-semibold text-neutral-800 dark:text-neutral-100 transition hover:bg-neutral-100 dark:hover:bg-neutral-700"
                >
                  Descargar .JSON
                </button>
              </div>

              <button
                type="button"
                onClick={() => setShowReportModal(false)}
                className="rounded-lg bg-neutral-900 dark:bg-neutral-100 px-4 py-2 text-xs font-semibold text-white dark:text-neutral-950 transition hover:bg-neutral-800 dark:hover:bg-neutral-200"
              >
                Cerrar
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Modal de detalle */}
      {selectedCard && (
        <div
          role="dialog"
          aria-modal="true"
          aria-labelledby="detail-modal-title"
          onClick={() => setSelectedCard(null)}
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4 backdrop-blur-[2px]"
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="w-full max-w-lg rounded-2xl border border-neutral-200 dark:border-neutral-800 bg-white dark:bg-neutral-900 p-6 shadow-2xl"
          >
            <div className="flex items-start justify-between gap-4">
              <div>
                <h2 id="detail-modal-title" className="text-lg font-semibold">
                  Detalle de predicción
                </h2>
                <p className="mt-0.5 break-all text-xs text-neutral-500 dark:text-neutral-400">
                  {selectedCard.filename}
                </p>
              </div>
              <button
                type="button"
                onClick={() => setSelectedCard(null)}
                aria-label="Cerrar modal"
                className="rounded-lg p-1.5 text-neutral-500 hover:bg-neutral-100 dark:hover:bg-neutral-800 hover:text-neutral-900 dark:hover:text-neutral-100"
              >
                ✕
              </button>
            </div>

            {selectedCard.previewUrl && (
              <div className="mt-4 overflow-hidden rounded-xl border border-neutral-200 dark:border-neutral-800 bg-neutral-100 dark:bg-neutral-950 p-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={selectedCard.previewUrl}
                  alt={selectedCard.filename}
                  className="mx-auto max-h-60 w-auto rounded-lg object-contain"
                />
              </div>
            )}

            <dl className="mt-4 divide-y divide-neutral-100 dark:divide-neutral-800 text-sm">
              <div className="flex items-center justify-between py-2.5">
                <dt className="text-neutral-500 dark:text-neutral-400">Diagnóstico</dt>
                <dd
                  className={`font-semibold ${
                    selectedCard.prob_maligno >= 0.5
                      ? "text-red-600 dark:text-red-400"
                      : "text-emerald-700 dark:text-emerald-400"
                  }`}
                >
                  {selectedCard.diagnostico}
                </dd>
              </div>

              <div className="flex items-center justify-between py-2.5 font-(family-name:--font-geist-mono)">
                <dt className="font-sans text-neutral-500 dark:text-neutral-400">
                  Prob. Melanoma Maligno
                </dt>
                <dd className="font-semibold text-red-600 dark:text-red-400">
                  {(selectedCard.prob_maligno * 100).toFixed(4)}%
                </dd>
              </div>

              <div className="flex items-center justify-between py-2.5 font-(family-name:--font-geist-mono)">
                <dt className="font-sans text-neutral-500 dark:text-neutral-400">
                  Prob. Melanoma Benigno
                </dt>
                <dd className="font-semibold text-emerald-700 dark:text-emerald-400">
                  {(selectedCard.prob_benigno * 100).toFixed(4)}%
                </dd>
              </div>

              <div className="flex items-center justify-between py-2.5">
                <dt className="text-neutral-500 dark:text-neutral-400">Modo de ejecución</dt>
                <dd className="rounded bg-neutral-100 dark:bg-neutral-800 px-2 py-0.5 font-mono text-xs">
                  {selectedCard.mode}
                </dd>
              </div>

              <div className="flex items-center justify-between py-2.5">
                <dt className="text-neutral-500 dark:text-neutral-400">Tiempo de proceso</dt>
                <dd className="font-mono text-xs font-medium">{selectedCard.tiempo_ms} ms</dd>
              </div>
            </dl>

            <div className="mt-5 flex justify-end">
              <button
                type="button"
                onClick={() => setSelectedCard(null)}
                className="rounded-lg bg-neutral-900 dark:bg-neutral-100 px-4 py-2 text-sm font-medium text-white dark:text-neutral-950 transition hover:bg-neutral-800 dark:hover:bg-neutral-200"
              >
                Cerrar
              </button>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}