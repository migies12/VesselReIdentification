import type { VesselEvent, InferenceResult } from "./types";

export async function fetchEvents(): Promise<VesselEvent[]> {
  const res = await fetch("/events");
  if (!res.ok) throw new Error(`Failed to fetch events: ${res.statusText}`);
  return res.json();
}

export async function inferEvent(eventId: string): Promise<InferenceResult> {
  const res = await fetch(`/events/${eventId}/infer`, { method: "POST" });
  if (!res.ok) throw new Error(`Inference failed: ${res.statusText}`);
  return res.json();
}
