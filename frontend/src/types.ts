export interface VesselEvent {
  event_id: string;
  event_type: string | null;
  mmsi: string | null;
  vessel_name: string | null;
  vessel_type: string | null;
  country_code: string | null;
  image_url: string | null;
  estimated_length: number | null;
  orientation: number | null;
  detection_score: number | null;
  lat: number | null;
  lon: number | null;
  time: string | null;
}

export interface GalleryMatch {
  score: number;
  boat_id: string;
  image_path: string;
  length_m: number | null;
}

export interface InferenceResult {
  event: VesselEvent;
  query_image: string;
  matched: boolean;
  top_match: GalleryMatch | null;
  all_results: GalleryMatch[];
}
