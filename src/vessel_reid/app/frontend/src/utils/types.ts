export interface VesselEvent {
  event_id: string;
  event_type: string | null;
  mmsi: string | null;
  vessel_name: string | null;
  vessel_type: string | null;
  country_code: string | null;
  image_url: string | null;
  estimated_length: number | null;
  heading: number | null;
  detection_score: number | null;
  lat: number | null;
  lon: number | null;
  time: string;
}

export interface GalleryMatch {
  score: number;
  boat_id: string;
  image_path: string;
  image_url: string;
  length_m: number;
  coords: [number, number];
  time: string;
}

export interface InferenceResult {
  event: VesselEvent;
  query_image: string;
  matched: boolean;
  top_match: GalleryMatch | null;
  all_results: GalleryMatch[];
}
