import { describe, expect, it } from "vitest";
import { clock, gb, layout } from "./strip";
import type { GpuFrame } from "./types";

const gpu = (caches: GpuFrame["caches"], hbm = 24e9): GpuFrame => ({
  name: "l4-0",
  kind: "l4",
  hbm_bytes: hbm,
  true_capacity: 21e9,
  capacity_estimate: 22e9,
  caches,
  host_caches: 0,
  host_bytes: 0,
  queued: 0,
  busy: false,
});

describe("layout", () => {
  it("fills weights, caches, small caches and free cells to the card's size", () => {
    const { total, segments } = layout(
      gpu([
        { id: "a", bytes: 5e9 },
        { id: "b", bytes: 3e8 },
        { id: "c", bytes: 4e8 },
      ]),
      1e9,
      2.2e9,
    );
    expect(total).toBe(24);
    expect(segments.map((s) => s.kind)).toEqual(["weights", "cache", "small", "free"]);
    expect(segments.reduce((a, s) => a + s.cells, 0)).toBe(24);
  });

  it("never draws more cells than the card has", () => {
    const { segments } = layout(gpu([{ id: "a", bytes: 30e9 }]), 1e9, 2.2e9);
    expect(segments.reduce((a, s) => a + s.cells, 0)).toBe(24);
  });

  it("alternates the shade of neighbouring caches", () => {
    const { segments } = layout(
      gpu([
        { id: "a", bytes: 3e9 },
        { id: "b", bytes: 3e9 },
      ]),
      1e9,
      2.2e9,
    );
    const caches = segments.filter((s) => s.kind === "cache");
    expect(caches.map((s) => (s.kind === "cache" ? s.alt : null))).toEqual([false, true]);
  });
});

describe("formatting", () => {
  it("formats clock time and sizes", () => {
    expect(clock(3 * 3600 + 25 * 60)).toBe("03:25");
    expect(gb(2.25e9)).toBe("2.3 GB");
    expect(gb(4e8)).toBe("400 MB");
  });
});
