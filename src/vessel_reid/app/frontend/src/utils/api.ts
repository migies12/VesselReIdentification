import type { VesselEvent, InferenceResult } from "./types";

export async function fetchEvents(): Promise<VesselEvent[]> {
  const res = await fetch("/api/events");
  if (!res.ok) throw new Error(`Failed to fetch events: ${res.statusText}`);
  return res.json();
}

export async function inferEvent(eventId: string): Promise<InferenceResult> {
  const res = await fetch(`/api/events/${eventId}/infer`, { method: "POST" });
  if (!res.ok) throw new Error(`Inference failed: ${res.statusText}`);
  return res.json();
}

export async function addDemoEvent(eventId: string): Promise<void> {
  const res = await fetch(`/api/events/demo/add/${eventId}`, {
    method: "POST"
  });
  if (!res.ok) throw new Error(`Failed to add demo event: ${res.statusText}`);
}

export async function removeDemoEvent(eventId: string): Promise<void> {
  const res = await fetch(`/api/events/demo/remove/${encodeURIComponent(eventId)}`, {
    method: "DELETE"
  });
  if (!res.ok) throw new Error(`Failed to remove event: ${res.statusText}`);
}

export async function fetchDemoEvents(): Promise<VesselEvent[]> {
  const res = await fetch("/api/events/demo");
  if (!res.ok) throw new Error(`Failed to fetch demo events: ${res.statusText}`);
  return res.json();
}
