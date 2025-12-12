import { useState, useRef, useCallback } from "react";

export interface MetricsData {
  totalTime: number;
  inferenceTime: number;
  latency: number;
  timestamp: string;
  messageLength: number;
}

export const useDevMetrics = (isEnabled: boolean) => {
  const [metrics, setMetrics] = useState<MetricsData[]>([]);
  const startRef = useRef<number | null>(null);

  const startMonitoring = useCallback(() => {
    if (!isEnabled) return;
    startRef.current = performance.now();
  }, [isEnabled]);

  const completeMonitoring = useCallback(
    (responseText: string, inferenceTime?: number) => {
      if (!isEnabled || !startRef.current) return;

      const end = performance.now();
      const total = end - startRef.current;
      const latency = Math.max(0, total - (inferenceTime ?? 0));

      const newMetric: MetricsData = {
        totalTime: total,
        inferenceTime: inferenceTime ?? 0,
        latency,
        timestamp: new Date().toISOString(),
        messageLength: responseText.length,
      };

      setMetrics((prev) => [newMetric, ...prev].slice(0, 50));
      startRef.current = null;
      return newMetric;
    },
    [isEnabled]
  );

  const clearMetrics = useCallback(() => setMetrics([]), []);

  return { metrics, startMonitoring, completeMonitoring, clearMetrics };
};