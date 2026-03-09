import { Map, Marker, ZoomControl } from "pigeon-maps"
import { getZoom, getHaversineDistance, getMapCentre, getTimeDifference } from "../utils/map_helper.ts";
import { VesselEvent, type GalleryMatch } from "../utils/types";
import "../styles/MatchModal.css";

const KM_TO_KNOTS = 1.852;

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
  const mapCentre: [number, number] = getMapCentre(marker1, marker2);
  const zoom: number = getZoom(marker1, marker2);
  const distance: number = getHaversineDistance(marker1, marker2);
  const timeDifference: number = getTimeDifference(match.time, event.time);
  const avgSpeed: number = timeDifference > 0 ? distance / timeDifference / KM_TO_KNOTS : 0;

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
              defaultCenter={mapCentre}
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
             <p><strong>Vessel Length:</strong> {match.length_m} m</p>
             <p><strong>Distance:</strong> {Math.round(distance)} km</p>
             <p><strong>Elapsed Time:</strong> {Math.round(timeDifference)} hrs</p>
             <p><strong>Implied Speed:</strong> {avgSpeed.toFixed(1)} kts</p>
          </div>
        </div>
      </div>
    </div>
  );
}