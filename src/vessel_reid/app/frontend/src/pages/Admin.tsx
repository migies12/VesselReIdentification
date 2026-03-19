import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { addDemoEvent, fetchDemoEvents, removeDemoEvent } from "../utils/api";
import "../styles/Admin.css";

export default function Admin() {
  const [eventId, setEventId] = useState("");
  const [demoIds, setDemoIds] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState<{ type: "success" | "error"; msg: string } | null>(null);

  useEffect(() => {
    loadIds();
  }, []);

  const loadIds = async () => {
    try {
      const events = await fetchDemoEvents();
      setDemoIds(events.map(e => e.event_id));
    } catch (err) {
      console.error("Failed to load demo IDs", err);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!eventId.trim() || loading) return;
    setLoading(true);
    setStatus(null);

    try {
      await addDemoEvent(eventId.trim());
      setStatus({ type: "success", msg: `Added ${eventId}` });
      setEventId("");
      await loadIds();
    } catch (err) {
      setStatus({ type: "error", msg: String(err) });
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (id: string) => {
    const confirmed = window.confirm(`Are you sure you want to remove event: ${id}?`);
    if (!confirmed) return;

    try {
      await removeDemoEvent(id);
      await loadIds(); // Refresh list
    } catch (err) {
      alert("Failed to remove event");
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '15px' }}>
          <Link to="/" className="nav-controls button">←</Link>
          <h1>Admin: Demo Management</h1>
        </div>
      </header>

      <main className="main-panel">
        <div className="event-details full-width-panel">
          <h2>Add New Demo Event</h2>
          <form onSubmit={handleSubmit} className="admin-form-group">
            <input
              type="text"
              className="admin-input"
              placeholder="Paste Event ID..."
              value={eventId}
              onChange={(e) => setEventId(e.target.value)}
              disabled={loading}
            />
            <button type="submit" className="btn-reid" disabled={loading || !eventId.trim()}>
              {loading ? "Adding..." : "Add Event"}
            </button>
          </form>
          {status && <div className={`status-msg ${status.type === "error" ? "error" : ""}`}>{status.msg}</div>}
        </div>

        <div className="event-details full-width-panel" style={{ marginTop: '20px' }}>
          <h2>Current Demo Events ({demoIds.length})</h2>
          <div className="admin-list">
            {demoIds.length === 0 && <p className="admin-help-text">No demo events stored.</p>}
            {demoIds.map((id) => (
              <div key={id} className="admin-list-item">
                <span className="admin-list-id">{id}</span>
                <button 
                  className="nav-controls button btn-danger" 
                  onClick={() => handleRemove(id)}
                >
                  Remove
                </button>
              </div>
            ))}
          </div>
        </div>
      </main>
    </div>
  );
}