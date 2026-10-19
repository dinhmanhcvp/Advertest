import { MOCK_CASES, MOCK_REPORT, MOCK_SLICES, MOCK_HARD_NEGATIVES, MOCK_SAMPLES, MOCK_TRACK, makeSample } from "./mockData";
import { asTag, landmarksFor, TAGS, TAG_ORDER } from "./taxonomy";
import type {
  CureCase,
  EvalReport,
  FailureSlice,
  HardNegative,
  StudioSample,
  WorkspaceData,
} from "./types";

interface AuditRecord {
  id: string;
  iou?: number;
  confidence?: number;
  audit_trail: { insight_source: string; applied_chain: string[]; severity_level: number; justification: string };
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 4000);
  try {
    const res = await fetch(path, { ...init, signal: ctrl.signal, headers: { "Content-Type": "application/json" } });
    if (!res.ok) throw new Error(`${path} → HTTP ${res.status}`);
    return (await res.json()) as T;
  } finally {
    clearTimeout(timer);
  }
}

/** Try the API path; if it is unreachable, read the same artifact as a static file (served from /demo). */
async function artifact<T>(apiPath: string, staticFile: string): Promise<{ data: T; from: "api" | "artifact" } | null> {
  try {
    return { data: await http<T>(apiPath), from: "api" };
  } catch {
    try {
      return { data: await http<T>(`/demo/${staticFile}`), from: "artifact" };
    } catch {
      return null;
    }
  }
}

const isAudit = (v: unknown): v is AuditRecord[] =>
  Array.isArray(v) && v.length > 0 && typeof v[0] === "object" && v[0] !== null && "audit_trail" in v[0];

const isCases = (v: unknown): v is CureCase[] =>
  Array.isArray(v) && v.length > 0 && typeof v[0]?.model_v1?.confidence === "number" && Array.isArray(v[0]?.model_v1?.bbox);

const isReport = (v: unknown): v is EvalReport => {
  const r = v as EvalReport | null;
  return !!r && typeof r.base_model?.base_map === "number" && typeof r.advertest_model?.corrupted_map === "number";
};

const isSynth = (r: AuditRecord) => r.id.startsWith("wider_adv_");

function fromAudit(all: AuditRecord[]): { slices: FailureSlice[]; negatives: HardNegative[]; samples: StudioSample[] } {
  // The artifact lists measured hard negatives first and the synthesized adversarial samples after them.
  const hard = all.filter((r) => !isSynth(r));
  const synth = all.filter(isSynth);
  const records = hard.length ? hard : all;
  const counts = new Map<string, number>();
  const negatives: HardNegative[] = records.map((r) => {
    const tag = asTag(r.audit_trail.insight_source);
    counts.set(tag, (counts.get(tag) ?? 0) + 1);
    return {
      id: r.id,
      mount: isSynth(r) || r.id.startsWith("wider_") ? "WIDER FACE · simulated" : TAGS[tag].mount,
      tag,
      conf: r.confidence ?? 0.2,
      iou: r.iou ?? 0.2,
    };
  });
  const slices: FailureSlice[] = TAG_ORDER.map((tag) => ({
    tag,
    label: TAGS[tag].label,
    count: counts.get(tag) ?? 0,
    pct: (counts.get(tag) ?? 0) / records.length,
  }));
  negatives.sort((a, b) => a.iou - b.iou);

  // Studio samples: first record of each tag, then fill — keeps the sample list diverse.
  const pool = synth.length ? synth : records;
  const picked: AuditRecord[] = [];
  for (const tag of TAG_ORDER) {
    const hit = pool.find((r) => asTag(r.audit_trail.insight_source) === tag);
    if (hit) picked.push(hit);
  }
  for (const r of pool) if (picked.length < 6 && !picked.includes(r)) picked.push(r);

  const samples = picked.map((r, i) => {
    const tag = asTag(r.audit_trail.insight_source);
    const base = MOCK_SAMPLES.find((s) => s.tag === tag) ?? MOCK_SAMPLES[0];
    const hz = 2 + ((i * 7) % 30) / 10;
    const s = makeSample(`atk_${r.id}`, tag, {
      box: base.box,
      hz,
      severity: r.audit_trail.severity_level,
      pan: tag === "error_motion_blur" ? 8 + i * 4 : 0,
    });
    s.audit = { ...r.audit_trail };
    s.landmarks = landmarksFor(s.box);
    return s;
  });
  return { slices, negatives, samples };
}

/**
 * Load everything the workspace needs.
 * - `demo = true`  → pre-computed artifacts (`GET /api/v1/demo/*`, `GET /api/v1/eval-report`)
 * - `demo = false` → live-capable endpoints (`POST /api/v1/triage/analyze`, `POST /api/v1/attack/generate`)
 * Any failure falls back, per part, to the local store; `source` is "api" only if the audit data came from the backend.
 */
export async function loadWorkspace(demo: boolean): Promise<WorkspaceData> {
  let audit: AuditRecord[] | null = null;
  let auditFrom: "api" | "artifact" = "api";
  if (!demo) {
    // Live-capable path first; any failure falls through to the artifacts below.
    try {
      const triage = await http<{ worst_samples: { id: string }[] }>("/api/v1/triage/analyze", {
        method: "POST",
        body: JSON.stringify({}),
      });
      const ids = triage.worst_samples.slice(0, 6).map((s) => s.id);
      const gen = await http<{ audit_trail: unknown }>("/api/v1/attack/generate", {
        method: "POST",
        body: JSON.stringify({ sample_ids: ids }),
      });
      if (isAudit(gen.audit_trail)) audit = gen.audit_trail;
    } catch {
      audit = null;
    }
  }
  if (!audit) {
    const r = await artifact<unknown>("/api/v1/demo/audit-trail", "audit_trail.json");
    if (r && isAudit(r.data)) {
      audit = r.data;
      auditFrom = r.from;
    }
  }

  const [cases, report] = await Promise.all([
    artifact<unknown>("/api/v1/demo/proof-of-cure", "proof_of_cure.json"),
    artifact<unknown>("/api/v1/eval-report", "eval_report.json"),
  ]);

  const fromApi = audit ? fromAudit(audit) : null;
  return {
    source: fromApi ? auditFrom : "local",
    slices: fromApi?.slices ?? MOCK_SLICES,
    hardNegatives: fromApi?.negatives ?? MOCK_HARD_NEGATIVES,
    samples: fromApi?.samples ?? MOCK_SAMPLES,
    cases: cases && isCases(cases.data) ? cases.data : MOCK_CASES,
    report: report && isReport(report.data) ? { ...MOCK_REPORT, ...report.data } : MOCK_REPORT,
    track: MOCK_TRACK,
  };
}
