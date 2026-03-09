import { Map, Marker, ZoomControl } from "pigeon-maps"
import { VesselEvent, type GalleryMatch } from "../utils/types";
import "../styles/MatchModal.css";

const mapProvider = (x: number, y: number, z: number) => {
  return `https://tiles.stadiamaps.com/tiles/osm_bright/${z}/${x}/${y}.png`;
}

interface MatchModalProps {
  match: GalleryMatch | null;
  event: VesselEvent | null;
  onClose: () => void;
}

export default function MatchModal({ match, event, onClose }: MatchModalProps) {
  if (!match || !event) return null;

  const marker1: [number, number] = [event.lat || 0, event.lon || 0];
  const marker2: [number, number] = [match.coords[0], match.coords[1]];
  const mapCenter: [number, number] = [(marker1[0] + marker2[0]) / 2, (marker1[1] + marker2[1]) / 2];
  const distance = Math.sqrt(Math.pow(marker1[0] + marker2[0], 2) + Math.pow(marker1[1] + marker2[1], 2));
  const zoom = distance < 0.1 ? 12 : distance < 2 ? 5 : 2;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close" onClick={onClose}>&times;</button>
        
        <h2>Match Details</h2>
        <div className="modal-body">
          <div className="map-wrapper">
            <Map
              height={400}
              provider={mapProvider}
              defaultCenter={mapCenter}
              defaultZoom={zoom}
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