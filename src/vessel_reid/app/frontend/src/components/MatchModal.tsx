import { Map, Marker, ZoomControl } from "pigeon-maps"
import { VesselEvent, type GalleryMatch } from "../utils/types";
import "../styles/MatchModal.css";

interface MatchModalProps {
  match: GalleryMatch | null;
  event: VesselEvent | null;
  onClose: () => void;
}

export default function MatchModal({ match, event, onClose }: MatchModalProps) {
  if (!match || !event) return null;

  const mapCenter: [number, number] = [event.lat || 0, event.lon || 0];

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>&times;</button>
        
        <h2>Match Details</h2>
        <div className="modal-body">
          <div className="map-wrapper">
            <Map
              height={400}
              defaultCenter={mapCenter}
              defaultZoom={5}
            >
              <ZoomControl />
              <Marker
                width={40}
                anchor={[event.lat || 0, event.lon || 0]}
                color="hsl(210, 100%, 50%)"
              />
              <Marker
                width={40}
                anchor={[match.coords[0], match.coords[1]]}
                color="hsl(350, 100%, 50%)"
              />
            </Map>
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