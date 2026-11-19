import type { GpuFrame } from "./types";

export type Segment =
  | { kind: "weights"; cells: number }
  | {
      kind: "cache";
      cells: number;
      id: string;
      bytes: number;
      rows?: number;
      features?: number;
      tenant?: string;
      alt: boolean;
    }
  | { kind: "small"; cells: number; count: number; bytes: number }
  | { kind: "free"; cells: number };

/**
 * Lay one GPU's memory out as cells of `bytesPerCell`: the model weights, each resident cache of at least one cell,
 * the small caches together, then free cells. Cells never exceed the card's nominal memory.
 */
export function layout(
  gpu: GpuFrame,
  bytesPerCell: number,
  weightsBytes: number,
): { total: number; segments: Segment[] } {
  const total = Math.round(gpu.hbm_bytes / bytesPerCell);
  const segments: Segment[] = [];
  let used = Math.min(total, Math.ceil(weightsBytes / bytesPerCell));
  segments.push({ kind: "weights", cells: used });
  let smallBytes = 0;
  let smallCount = 0;
  let alt = false;
  for (const c of gpu.caches) {
    const cells = Math.round(c.bytes / bytesPerCell);
    if (cells < 1) {
      smallBytes += c.bytes;
      smallCount += 1;
      continue;
    }
    const take = Math.min(cells, total - used);
    if (take <= 0) break;
    segments.push({
      kind: "cache",
      cells: take,
      id: c.id,
      bytes: c.bytes,
      rows: c.rows,
      features: c.features,
      tenant: c.tenant,
      alt,
    });
    alt = !alt;
    used += take;
  }
  if (smallCount > 0) {
    const cells = Math.min(Math.max(1, Math.round(smallBytes / bytesPerCell)), total - used);
    if (cells > 0) {
      segments.push({ kind: "small", cells, count: smallCount, bytes: smallBytes });
      used += cells;
    }
  }
  if (total - used > 0) segments.push({ kind: "free", cells: total - used });
  return { total, segments };
}

export function residentBytes(gpu: GpuFrame): number {
  return gpu.caches.reduce((a, c) => a + c.bytes, 0);
}

export function clock(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}`;
}

export function gb(bytes: number): string {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(1)} GB` : `${Math.round(bytes / 1e6)} MB`;
}
