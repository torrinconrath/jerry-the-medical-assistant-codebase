import React from "react";
import type { MetricsData } from "../hooks/useDevMetrics";
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faTrash, faClock, faCog, faWifi, faDatabase } from '@fortawesome/free-solid-svg-icons';

interface DevMetricsProps {
  isEnabled: boolean;
  metrics: MetricsData[];
  clearMetrics: () => void;
}

const DevMetrics: React.FC<DevMetricsProps> = ({ isEnabled, metrics, clearMetrics }) => {
  if (!isEnabled) return null;

  const avg = (arr: number[]) =>
    arr.length ? Math.round(arr.reduce((a, b) => a + b, 0) / arr.length) : 0;

  const avgTotal = avg(metrics.map((m) => m.totalTime));
  const avgInference = avg(metrics.map((m) => m.inferenceTime));
  const avgLatency = avg(metrics.map((m) => m.latency));

  return (
    <div className="dev-metrics-panel">
      <div className="metrics-header">
        <h3>Dev Metrics</h3>
        <button className="metrics-button clear-button" onClick={clearMetrics}>
          <FontAwesomeIcon icon={faTrash} /> Clear
        </button>
      </div>

      <div className="metrics-summary">
        <div><FontAwesomeIcon icon={faClock} /> Avg Total Time: {avgTotal} ms</div>
        <div><FontAwesomeIcon icon={faCog} /> Avg Inference: {avgInference} ms</div>
        <div><FontAwesomeIcon icon={faWifi} /> Avg Latency: {avgLatency} ms</div>
        <div><FontAwesomeIcon icon={faDatabase} /> Total Samples: {metrics.length}</div>
      </div>

      <div className="metrics-details">
        <h4>Recent</h4>
        <ul>
          {metrics.slice(0, 10).map((m, i) => (
            <li key={i}>
              <div>
                [{new Date(m.timestamp).toLocaleTimeString()}] 
              </div>
              <span><FontAwesomeIcon icon={faClock} /> {m.totalTime.toFixed(0)}ms  </span>   
              <span><FontAwesomeIcon icon={faCog} />{" "} {m.inferenceTime.toFixed(0)}ms  </span>   
              <span><FontAwesomeIcon icon={faWifi} /> {m.latency.toFixed(0)}ms  </span> 
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default DevMetrics;
