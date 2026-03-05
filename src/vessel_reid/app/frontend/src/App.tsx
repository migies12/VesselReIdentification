import { useEffect, useRef, useState } from "react";
import { fetchEvents, inferEvent } from "./api";
import type { VesselEvent, InferenceResult } from "./types";
import "./index.css";

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toUTCString().replace(" GMT", " UTC");
}

function formatCoord(val: number | null): string {
  if (val === null) return "—";
  return val.toFixed(4);
}

export default function App() {
  const [events, setEvents] = useState<VesselEvent[]>([]);
  const [cursor, setCursor] = useState(0);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingEvents, setLoadingEvents] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const cache = useRef<Record<string, InferenceResult>>({});

  useEffect(() => {
    fetchEvents()
      .then((data) => {
        setEvents(data);
        setLoadingEvents(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoadingEvents(false);
      });
  }, []);

  useEffect(() => {
    if (events.length === 0) return;
    const event = events[cursor];
    if (cache.current[event.event_id]) {
      setResult(cache.current[event.event_id]);
      return;
    }
    setResult(null);
    setError(null);
    setLoading(true);
    inferEvent(event.event_id)
      .then((data) => {
        cache.current[event.event_id] = data;
        setResult(data);
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  }, [events, cursor]);

  const currentEvent = events[cursor] ?? null;
  const top3 = result?.all_results.slice(0, 3) ?? [];

  return (
    <div className="app">
      <header className="app-header">
        <h1>Vessel Re-Identification</h1>
        <div className="nav-controls">
          <button
            onClick={() => setCursor((c) => Math.max(0, c - 1))}
            disabled={cursor === 0 || loadingEvents || loading}
          >
            ← Prev
          </button>
          <span className="nav-counter">
            {events.length > 0 ? `${cursor + 1} / ${events.length}` : "—"}
          </span>
          <button
            onClick={() => setCursor((c) => Math.min(events.length - 1, c + 1))}
            disabled={cursor >= events.length - 1 || loadingEvents || loading}
          >
            Next →
          </button>
        </div>
      </header>

      {loadingEvents && <div className="status-msg">Loading events…</div>}
      {error && <div className="status-msg error">{error}</div>}

      {!loadingEvents && currentEvent && (
        <>
          <div className="main-panel">
            <div className="query-image-box">
              {loading && <div className="image-placeholder">Running inference…</div>}
              {!loading && result && (
                <img
                  src={`data:image/jpeg;base64,${result.query_image}`}
                  alt="Satellite detection"
                  className="query-image"
                />
              )}
              {!loading && !result && !error && (
                <div className="image-placeholder">No image</div>
              )}
            </div>

            <div className="event-details">
              <h2>Event Details</h2>
              <table className="details-table">
                <tbody>
                  <tr><th>MMSI</th><td>{currentEvent.mmsi ?? "—"}</td></tr>
                  <tr><th>Vessel Name</th><td>{currentEvent.vessel_name ?? "—"}</td></tr>
                  <tr><th>Type</th><td>{currentEvent.vessel_type ?? "—"}</td></tr>
                  <tr><th>Country</th><td>{currentEvent.country_code ?? "—"}</td></tr>
                  <tr>
                    <th>Length</th>
                    <td>{currentEvent.estimated_length != null ? `${currentEvent.estimated_length} m` : "—"}</td>
                  </tr>
                  <tr>
                    <th>Heading</th>
                    <td>{currentEvent.orientation != null ? `${currentEvent.orientation}°` : "—"}</td>
                  </tr>
                  <tr>
                    <th>Detection Score</th>
                    <td>{currentEvent.detection_score != null ? currentEvent.detection_score.toFixed(3) : "—"}</td>
                  </tr>
                  <tr><th>Time</th><td>{formatTime(currentEvent.time)}</td></tr>
                  <tr>
                    <th>Location</th>
                    <td>{formatCoord(currentEvent.lat)}, {formatCoord(currentEvent.lon)}</td>
                  </tr>
                  {result && (
                    <tr>
                      <th>Re-ID</th>
                      <td className={result.matched ? "tag-match" : "tag-no-match"}>
                        {result.matched ? "Match found" : "No match"}
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>

          <div className="matches-section">
            <h2>Top Matches</h2>
            {loading && <div className="status-msg">Running inference…</div>}
            {!loading && top3.length === 0 && !error && (
              <div className="status-msg">No results yet</div>
            )}
            <div className="matches-grid">
              {top3.map((match, i) => (
                <div key={i} className="match-card">
                  <div className="match-image-box">
                    <img
                      src={`/gallery-image/${match.image_path}`}
                      alt={`Match ${i + 1}`}
                      className="match-image"
                      onError={(e) => {
                        (e.currentTarget as HTMLImageElement).src = "";
                        e.currentTarget.parentElement!.classList.add("no-image");
                      }}
                    />
                  </div>
                  <div className="match-info">
                    <div className="match-score">{(match.score * 100).toFixed(1)}%</div>
                    <div className="match-detail"><span>MMSI</span>{match.boat_id}</div>
                    {match.length_m != null && (
                      <div className="match-detail"><span>Length</span>{match.length_m.toFixed(0)} m</div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
