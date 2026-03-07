import { type GalleryMatch } from "../utils/types";
import "../styles/MatchModal.css";


export default function MatchModal({ 
  match, 
  onClose 
}: { 
  match: GalleryMatch | null; 
  onClose: () => void 
}) {
  if (!match) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>&times;</button>
        
        <h2>Match Details</h2>
        <div className="modal-body">
          <div className="map-placeholder">
            Map View for MMSI: {match.boat_id}
          </div>
          
          <div className="modal-info-grid">
             <p><strong>Confidence:</strong> {(match.score * 100).toFixed(1)}%</p>
             <p><strong>Vessel Length:</strong> {match.length_m}m</p>
          </div>
        </div>
      </div>
    </div>
  );
}