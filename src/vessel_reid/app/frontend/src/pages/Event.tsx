import { useState } from "react";
import { Link } from "react-router-dom";

import MatchModal from "../components/MatchModal";
import "../styles/index.css";
import { getEventByUrl, inferEvent } from "../utils/api";
import { type VesselEvent, type InferenceResult, type GalleryMatch } from "../utils/types";

function formatTime(iso: string | null): string {
  if (!iso) return "—";
  return new Date(iso).toUTCString().replace(" GMT", " UTC");
}

function formatCoord(val: number | null | undefined): string {
  if (val == null) return "—";
  return val.toFixed(4);
}

export default function Event(
) {
  const [event, setEvent] = useState<VesselEvent>();
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [eventLoading, setEventLoading] = useState(false);
  const [inferenceLoading, setInferenceLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedMatch, setSelectedMatch] = useState<GalleryMatch | null>(null);

  const top3 = result?.all_results?.slice(0, 3) ?? [];

  function handleInfer() {
    if (!event || eventLoading) return;
    setError(null);
    setInferenceLoading(true);
    inferEvent(event.event_id)
      .then((data) => {
        setResult(data);
        setInferenceLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setInferenceLoading(false);
      });
  }

  function handleSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const input = formData.get("eventId")?.toString().trim();

    if (!input) return;

    setEventLoading(true);
    setError(null);
    setResult(null);

    const fetchPromise = getEventByUrl(input) ;

    fetchPromise
      .then((data) => {
        setEvent(data);
        setEventLoading(false);
        (e.target as HTMLFormElement).reset();
      })
      .catch((err) => {
        setError(err.message || "Failed to load event.");
        setEventLoading(false);
      });
  }

  return (
    <div className="app">
        <header className="app-header">
            <Link to="/" className="nav-controls button">←</Link>
            <h1>Vessel Re-Identification</h1>
            <div className="nav-controls">
                <button
                className="btn-reid"
                onClick={handleInfer}
                disabled={eventLoading || !event || inferenceLoading}
                >
                {inferenceLoading ? "Running…" : "Run Re-ID"}
                </button>
            </div>
        </header>
        <form onSubmit={handleSubmit} className="admin-form-group">
            <input
                name="eventId"
                type="text"
                className="admin-input"
                placeholder="Paste Event URL..."
                disabled={eventLoading}
            />
            <button type="submit" className="btn-reid">
                {eventLoading ? "Adding..." : "Add Event"}
            </button>
        </form>

      {eventLoading && <div className="status-msg">Loading event…</div>}
      {!eventLoading && event && (
        <>
          <div className="main-panel">
            <div className="query-image-box">
              {event.image_url ? (
                <img
                  src={event.image_url}
                  alt="Satellite detection"
                  className="query-image"
                  onError={(e) => {
                    e.currentTarget.style.display = "none";
                    e.currentTarget.parentElement!.classList.add("no-image");
                  }}
                />
              ) : (
                <div className="image-placeholder">No image available</div>
              )}
            </div>

            <div className="event-details">
              <h2>Event Details</h2>
              <table className="details-table">
                <tbody>
                  <tr><th>MMSI</th><td>{event.mmsi ?? "—"}</td></tr>
                  <tr><th>Vessel Name</th><td>{event.vessel_name ?? "—"}</td></tr>
                  <tr><th>Type</th><td>{event.vessel_type ?? "—"}</td></tr>
                  <tr><th>Country</th><td>{event.country_code ?? "—"}</td></tr>
                  <tr>
                    <th>Length</th>
                    <td>{event.estimated_length != null ? `${event.estimated_length} m` : "—"}</td>
                  </tr>
                  <tr>
                    <th>Heading</th>
                    <td>{event.heading != null ? `${event.heading}°` : "—"}</td>
                  </tr>
                  <tr>
                    <th>Detection Score</th>
                    <td>{event.detection_score != null ? event.detection_score.toFixed(3) : "—"}</td>
                  </tr>
                  <tr><th>Time</th><td>{formatTime(event.time)}</td></tr>
                  <tr>
                    <th>Location</th>
                    <td>{formatCoord(event.lat)}, {formatCoord(event.lon)}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>

          <div className="matches-section">
            <h2>Top Matches</h2>
            {inferenceLoading && <div className="status-msg">Running inference…</div>}
            {error && <div className="status-msg error">{error}</div>}
            {!inferenceLoading && !result && !error && (
              <div className="status-msg">Press "Run Re-ID" to see matches</div>
            )}
            {!inferenceLoading && result && top3.length === 0 && (
              <div className="status-msg">No matches found above threshold</div>
            )}
            <div className="matches-grid">
              {top3.map((match, i) => (
                <div 
                  key={i} 
                  className="match-card"
                  onClick={() => setSelectedMatch(match)}
                >
                  <div className="match-image-box">
                    <img
                      src={match.image_url}
                      alt={`Match ${i + 1}`}
                      className="match-image"
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
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

      <MatchModal
        match={selectedMatch}
        event={event || null}
        onClose={() => setSelectedMatch(null)}
      />
    </div>
  );
}
