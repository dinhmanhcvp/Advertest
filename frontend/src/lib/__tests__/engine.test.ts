import { afterEach, describe, expect, it, vi } from "vitest";
import { loadWorkspace } from "../api";
import { boxIntegrity, predictBox } from "../geometry";
import { MOCK_SLICES, MOCK_WORKSPACE, TOTAL_HARD_NEGATIVES } from "../mockData";
import { asTag, imuPath, landmarksFor, TAG_ORDER } from "../taxonomy";

describe("taxonomy", () => {
  it("maps unknown tags to a canonical one", () => {
    expect(asTag("error_fisheye")).toBe("error_fisheye");
    expect(TAG_ORDER).toContain(asTag("something_else"));
  });

  it("derives five landmarks inside the face box", () => {
    const box: [number, number, number, number] = [10, 20, 50, 60];
    const pts = landmarksFor(box);
    expect(pts).toHaveLength(5);
    for (const [x, y] of pts) {
      expect(x).toBeGreaterThan(box[0]);
      expect(x).toBeLessThan(box[0] + box[2]);
      expect(y).toBeGreaterThan(box[1]);
      expect(y).toBeLessThan(box[1] + box[3]);
    }
  });

  it("IMU path is non-linear and grows with severity", () => {
    const low = imuPath(3, 1, 0);
    const high = imuPath(3, 5, 0);
    const span = (p: [number, number][]) => Math.max(...p.map(([, y]) => y)) - Math.min(...p.map(([, y]) => y));
    expect(span(high)).toBeGreaterThan(span(low));
    expect(span(low)).toBeGreaterThan(0);
  });
});

describe("geometry", () => {
  it("extrapolates a box with constant velocity", () => {
    expect(predictBox([48, 34, 40, 52], [55, 33, 40, 52])).toEqual([62, 32, 40, 52]);
  });

  it("flags boxes pushed out of frame by distortion", () => {
    expect(boxIntegrity([58, 24, 44, 58], 0)).toBeCloseTo(1);
    expect(boxIntegrity([150, 100, 40, 40], 1)).toBeLessThan(0.5);
  });
});

describe("mock store", () => {
  it("failure taxonomy sums to the total and matches the spec split", () => {
    expect(MOCK_SLICES.reduce((a, s) => a + s.count, 0)).toBe(TOTAL_HARD_NEGATIVES);
    expect(MOCK_SLICES.map((s) => +(s.pct * 100).toFixed(1))).toEqual([42.5, 31, 18.5, 8]);
  });

  it("passes the blind evaluation gate", () => {
    const { base_model: b, advertest_model: a, gate } = MOCK_WORKSPACE.report;
    expect(b.base_map - a.base_map).toBeLessThanOrEqual(gate.max_base_drop + 1e-9);
    expect(a.corrupted_map - b.corrupted_map).toBeGreaterThanOrEqual(gate.min_corrupted_gain);
  });
});

describe("loadWorkspace", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("falls back to local data when the API is unreachable", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new Error("offline")));
    const data = await loadWorkspace(true);
    expect(data.source).toBe("local");
    expect(data.cases.length).toBeGreaterThan(0);
  });

  it("reads the measured artifacts as static files when the API is down", async () => {
    const audit = [
      { id: "wider_0001", iou: 0.1, confidence: 0.2, audit_trail: { insight_source: "error_fisheye", applied_chain: ["OpticalDistortion"], severity_level: 3, justification: "x" } },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        const ok = url.startsWith("/demo/audit_trail");
        return { ok, status: ok ? 200 : 404, json: async () => audit };
      }),
    );
    const data = await loadWorkspace(true);
    expect(data.source).toBe("artifact");
    expect(data.hardNegatives[0].id).toBe("wider_0001");
  });

  it("uses API artifacts when available", async () => {
    const audit = [
      { id: "ego4d_001", iou: 0.1, confidence: 0.2, audit_trail: { insight_source: "error_fisheye", applied_chain: ["OpticalDistortion"], severity_level: 3, justification: "x" } },
    ];
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => ({
        ok: url.includes("audit-trail"),
        status: url.includes("audit-trail") ? 200 : 404,
        json: async () => audit,
      })),
    );
    const data = await loadWorkspace(true);
    expect(data.source).toBe("api");
    expect(data.hardNegatives[0].id).toBe("ego4d_001");
    expect(data.samples[0].id).toBe("atk_ego4d_001");
  });
});
