// Closed-loop engine: evaluation → triage → insight extraction → attack-combo
// recommendation → attack execution → retrain DB → retrain → RL reward.
//
// The engine is a deterministic, seeded *simulation* of a perception model's
// response to degradations.  It exposes plain functions with serialisable
// inputs/outputs so it can later be swapped for real backend calls
// (Yolov7Evaluator / AttackEngine / Ray trainer) without touching the UI.

import {
  ATTACK_BY_ID,
  ATTACKS,
  DATASET_BY_ID,
  INSIGHT_TAGS,
  MODEL_BY_ID,
  synergyOf,
} from "./catalog";
import { clamp, gauss, hashStr, makeRng, pick, weightedPick } from "./rng";

const PENALTY_K = 0.74; // condition intensity → score penalty
const ATTACK_K = 0.6; // attack at sev 5 / aff 1 → added condition intensity
const ERR_MISSED = 0.25;
const ERR_CORRECT = 0.5;

// ─────────────────────────────── Samples ───────────────────────────────────

const sampleCache = new Map();

function datasetPrefix(dataset) {
  return dataset.id.split("-")[0].toUpperCase().slice(0, 4);
}

export function generateSamples(datasetId) {
  if (sampleCache.has(datasetId)) return sampleCache.get(datasetId);
  const dataset = DATASET_BY_ID[datasetId];
  const prefix = datasetPrefix(dataset);
  const mixEntries = Object.entries(dataset.mix);
  const samples = [];
  for (let i = 0; i < dataset.size; i += 1) {
    const seed = hashStr(`${dataset.id}:${i}`);
    const rng = makeRng(seed);
    const conds = {};
    for (const tag of INSIGHT_TAGS) conds[tag] = Math.pow(rng(), 3) * 0.12;
    const primary = weightedPick(rng, mixEntries);
    conds[primary] = clamp(0.08 + 0.92 * Math.pow(rng(), 1.35 / dataset.pressure));
    const cls = pick(rng, dataset.classes);
    const small = cls === "small-face" || cls === "cyclist" || cls === "pedestrian";
    const w = dataset.scene === "portrait" ? (small ? 0.16 : 0.26) + rng() * 0.06 : (small ? 0.12 : 0.24) + rng() * 0.1;
    const h = dataset.scene === "portrait" ? w * 1.28 * 1.6 : w * (cls === "pedestrian" || cls === "cyclist" ? 1.9 : 0.78) * 1.6;
    const gt = {
      cx: dataset.scene === "portrait" ? 0.5 + (rng() - 0.5) * 0.36 : 0.2 + rng() * 0.6,
      cy: dataset.scene === "portrait" ? 0.46 + (rng() - 0.5) * 0.12 : 0.58 + rng() * 0.1,
      w: Math.min(w, 0.4),
      h: Math.min(h, 0.7),
    };
    samples.push({
      id: `${prefix}-${String(i + 1).padStart(4, "0")}`,
      index: i,
      seed,
      datasetId,
      scene: dataset.scene,
      cls,
      conds,
      primaryIntrinsic: primary,
      gt,
      eps: gauss(rng) * 0.045,
      epsConf: gauss(rng) * 0.05,
    });
  }
  sampleCache.set(datasetId, samples);
  return samples;
}

// ─────────────────────────────── Scoring ───────────────────────────────────

const noFixes = { fixes: {}, baseDelta: 0 };
export const freshModelState = () => ({ version: 1, fixes: {}, baseDelta: 0, history: [] });

function sensEff(model, state, tag) {
  return model.sens[tag] * (1 - (state.fixes?.[tag] || 0));
}

export function scoreConds(conds, model, state, eps = 0) {
  let penalty = 0;
  for (const tag of INSIGHT_TAGS) penalty += conds[tag] * sensEff(model, state, tag);
  return clamp(model.base + (state.baseDelta || 0) - PENALTY_K * penalty + eps);
}

export function attribution(conds, model, state) {
  return INSIGHT_TAGS.map((tag) => ({ tag, contrib: conds[tag] * sensEff(model, state, tag) })).sort(
    (a, b) => b.contrib - a.contrib,
  );
}

export const errorType = (score) => (score >= ERR_CORRECT ? "ok" : score >= ERR_MISSED ? "mislocalized" : "missed");

// ───────────────────────────── Evaluation ──────────────────────────────────

function averagePrecision(scored) {
  const ranked = [...scored].sort((a, b) => b.conf - a.conf);
  let tp = 0;
  let apSum = 0;
  ranked.forEach((s, idx) => {
    if (s.correct) {
      tp += 1;
      apSum += tp / (idx + 1);
    }
  });
  return scored.length ? apSum / scored.length : 0;
}

export function evaluate(modelId, datasetId, modelState = noFixes) {
  const model = MODEL_BY_ID[modelId];
  const dataset = DATASET_BY_ID[datasetId];
  const samples = generateSamples(datasetId);
  const scored = samples.map((s) => {
    const score = scoreConds(s.conds, model, modelState, s.eps);
    const attr = attribution(s.conds, model, modelState);
    return {
      ...s,
      score,
      conf: clamp(score + s.epsConf),
      correct: score >= ERR_CORRECT,
      err: errorType(score),
      attrTag: attr[0].tag,
      attrMargin: attr[0].contrib - attr[1].contrib,
      attr,
    };
  });
  const n = scored.length;
  const ok = scored.filter((s) => s.correct).length;
  const missed = scored.filter((s) => s.err === "missed").length;
  const mis = scored.filter((s) => s.err === "mislocalized").length;

  const byClass = dataset.classes.map((cls) => {
    const group = scored.filter((s) => s.cls === cls);
    return { cls, n: group.length, acc: group.filter((s) => s.correct).length / Math.max(1, group.length) };
  });
  const byTag = INSIGHT_TAGS.map((tag) => {
    const group = scored.filter((s) => s.attrTag === tag);
    return { tag, n: group.length, acc: group.filter((s) => s.correct).length / Math.max(1, group.length) };
  });
  const bins = Array.from({ length: 20 }, (_, i) => ({ lo: i / 20, hi: (i + 1) / 20, count: 0 }));
  for (const s of scored) bins[Math.min(19, Math.floor(s.score * 20))].count += 1;

  return {
    modelId,
    datasetId,
    n,
    scored,
    metrics: {
      accuracy: ok / n,
      map50: averagePrecision(scored),
      meanIou: scored.reduce((a, s) => a + s.score, 0) / n,
      missRate: missed / n,
      mislocRate: mis / n,
      ok,
      missed,
      mis,
    },
    byClass,
    byTag,
    bins,
  };
}

// ───────────────────────────── Triage & insights ───────────────────────────

export function triage(evalResult, k) {
  return [...evalResult.scored].sort((a, b) => a.score - b.score || a.index - b.index).slice(0, k);
}

const LEVELS = [
  { min: 0.45, key: "critical", label: "Nghiêm trọng" },
  { min: 0.25, key: "high", label: "Cao" },
  { min: 0.12, key: "medium", label: "Trung bình" },
  { min: 0, key: "low", label: "Thấp" },
];

export function extractInsights(worst, modelId = null) {
  const total = worst.length || 1;
  const groups = new Map();
  for (const s of worst) {
    const tag = s.attrTag;
    if (!groups.has(tag)) groups.set(tag, []);
    groups.get(tag).push(s);
  }
  const insights = [];
  for (const [tag, group] of groups) {
    const share = group.length / total;
    const avgScore = group.reduce((a, s) => a + s.score, 0) / group.length;
    const avgIntensity = group.reduce((a, s) => a + s.conds[tag], 0) / group.length;
    const confidence = clamp(0.55 + (group.reduce((a, s) => a + s.attrMargin, 0) / group.length) * 1.6, 0.5, 0.99);
    const secondaryCount = {};
    for (const s of group) {
      const second = s.attr[1].tag;
      secondaryCount[second] = (secondaryCount[second] || 0) + 1;
    }
    const secondary = Object.entries(secondaryCount)
      .map(([t, c]) => ({ tag: t, share: c / group.length }))
      .sort((a, b) => b.share - a.share)
      .slice(0, 2);
    const impact = share * (1 - avgScore);
    const level = LEVELS.find((l) => impact >= l.min * 0.6) || LEVELS[LEVELS.length - 1];
    insights.push({
      tag,
      count: group.length,
      share,
      avgScore,
      avgIntensity,
      confidence,
      secondary,
      impact,
      level,
      sampleIds: group.map((s) => s.id),
      modelId,
    });
  }
  return insights.sort((a, b) => b.impact - a.impact);
}

// ───────────────────────────── Attack mechanics ────────────────────────────

export const comboSignature = (attacks) =>
  [...attacks]
    .map((a) => a.id)
    .sort()
    .join("+");

/** Added condition intensity per failure tag produced by a combo of attacks. */
export function comboAddedIntensity(attacks) {
  const add = {};
  for (const tag of INSIGHT_TAGS) add[tag] = 0;
  for (const a of attacks) {
    const def = ATTACK_BY_ID[a.id];
    for (const [tag, aff] of Object.entries(def.aff)) add[tag] += aff * (a.severity / 5) * ATTACK_K;
  }
  let synergy = 0;
  for (let i = 0; i < attacks.length; i += 1) {
    for (let j = i + 1; j < attacks.length; j += 1) {
      synergy += synergyOf(attacks[i].id, attacks[j].id) * (Math.min(attacks[i].severity, attacks[j].severity) / 5);
    }
  }
  const mult = 1 + synergy;
  for (const tag of INSIGHT_TAGS) add[tag] = Math.min(1.1, add[tag] * mult);
  return add;
}

const FIDELITY_COST = { cutout: 0.14, fisheye: 0.08, grid_distortion: 0.1, color_shift: 0.04, fog: 0.06 };

export function comboFidelity(attacks) {
  let loss = 0;
  for (const a of attacks) loss += (FIDELITY_COST[a.id] ?? 0.09) * Math.pow(a.severity / 5, 1.3);
  return clamp(1 - loss);
}

export function attackedConds(conds, attacks) {
  const add = comboAddedIntensity(attacks);
  const out = {};
  for (const tag of INSIGHT_TAGS) out[tag] = clamp(conds[tag] + add[tag], 0, 1.35);
  return out;
}

export function runAttack(sample, modelId, modelState, attacks) {
  const model = MODEL_BY_ID[modelId];
  const sig = comboSignature(attacks);
  const rng = makeRng(`${sample.seed}:${sig}`);
  const jitter = gauss(rng) * 0.02;
  const beforeScore = scoreConds(sample.conds, model, modelState, sample.eps);
  const conds2 = attackedConds(sample.conds, attacks);
  const afterScore = Math.min(beforeScore, scoreConds(conds2, model, modelState, sample.eps + jitter));
  const beforeConf = clamp(beforeScore + sample.epsConf);
  const afterConf = clamp(afterScore + sample.epsConf * 0.6);
  const drop = beforeScore - afterScore;
  const relDrop = beforeScore > 0 ? drop / beforeScore : 0;
  return {
    sampleId: sample.id,
    beforeScore,
    afterScore,
    beforeConf,
    afterConf,
    drop,
    relDrop,
    beforeErr: errorType(beforeScore),
    afterErr: errorType(afterScore),
    flipped: beforeScore >= ERR_CORRECT && afterScore < ERR_CORRECT,
    newlyMissed: beforeScore >= ERR_MISSED && afterScore < ERR_MISSED,
    effective: drop >= 0.08,
    fidelity: comboFidelity(attacks),
    conds2,
  };
}

export function summarizeAttack(results) {
  const n = results.length || 1;
  const mean = (f) => results.reduce((a, r) => a + f(r), 0) / n;
  const asrCount = results.filter((r) => r.relDrop >= 0.4 || r.newlyMissed).length;
  return {
    n: results.length,
    scoreBefore: mean((r) => r.beforeScore),
    scoreAfter: mean((r) => r.afterScore),
    confBefore: mean((r) => r.beforeConf),
    confAfter: mean((r) => r.afterConf),
    detBefore: results.filter((r) => r.beforeScore >= ERR_CORRECT).length / n,
    detAfter: results.filter((r) => r.afterScore >= ERR_CORRECT).length / n,
    missBefore: results.filter((r) => r.beforeScore < ERR_MISSED).length / n,
    missAfter: results.filter((r) => r.afterScore < ERR_MISSED).length / n,
    asr: asrCount / n,
    meanDrop: mean((r) => r.drop),
    meanFidelity: mean((r) => r.fidelity),
  };
}

// ───────────────────────────── RL memory (contextual bandit) ───────────────

export const memoryEntryQ = (entry) => {
  const dropAvg = entry.sumDrop / Math.max(1, entry.n);
  if (!entry.nReward) return dropAvg * 0.6;
  return 0.4 * dropAvg + 0.6 * (entry.sumReward / entry.nReward);
};

export function emptyMemory() {
  return {};
}

export function recordMemory(memory, tag, attacks, result) {
  const sig = comboSignature(attacks);
  const next = { ...memory, [tag]: { ...(memory[tag] || {}) } };
  const prev = next[tag][sig] || {
    sig,
    tag,
    attacks: attacks.map((a) => ({ id: a.id, severity: a.severity })),
    n: 0,
    sumDrop: 0,
    sumFlip: 0,
    sumFid: 0,
    nReward: 0,
    sumReward: 0,
    history: [],
    rewardHistory: [],
  };
  next[tag][sig] = {
    ...prev,
    attacks: attacks.map((a) => ({ id: a.id, severity: a.severity })),
    n: prev.n + 1,
    sumDrop: prev.sumDrop + result.drop,
    sumFlip: prev.sumFlip + (result.flipped || result.newlyMissed ? 1 : 0),
    sumFid: prev.sumFid + result.fidelity,
    history: [...prev.history, result.drop].slice(-40),
  };
  return next;
}

export function applyRewards(memory, rewardsByTag, dbEntries) {
  const next = {};
  for (const [tag, entries] of Object.entries(memory)) next[tag] = { ...entries };
  for (const [tag, reward] of Object.entries(rewardsByTag)) {
    const tagEntries = dbEntries.filter((e) => e.tag === tag);
    if (!tagEntries.length || !next[tag]) continue;
    const counts = {};
    for (const e of tagEntries) counts[e.sig] = (counts[e.sig] || 0) + 1;
    for (const [sig, c] of Object.entries(counts)) {
      const prev = next[tag][sig];
      if (!prev) continue;
      const attributed = reward * (0.5 + (0.5 * c) / tagEntries.length);
      next[tag][sig] = {
        ...prev,
        nReward: prev.nReward + 1,
        sumReward: prev.sumReward + attributed,
        rewardHistory: [...(prev.rewardHistory || []), attributed].slice(-20),
      };
    }
  }
  return next;
}

// ───────────────────────────── Combo recommendation ────────────────────────

const AGGRESSION = { mild: 0.58, moderate: 0.82, aggressive: 1 };
export const AGGRESSION_LEVELS = [
  { id: "mild", label: "Nhẹ", hint: "Giữ ảnh gần thực tế" },
  { id: "moderate", label: "Vừa", hint: "Cân bằng độ khó & độ thực" },
  { id: "aggressive", label: "Mạnh", hint: "Tối đa suy giảm" },
];

function relevance(attackId, insight) {
  const aff = ATTACK_BY_ID[attackId].aff;
  const second = insight.secondary?.[0]?.tag;
  return (aff[insight.tag] || 0) + (second ? 0.45 * (aff[second] || 0) : 0);
}

function combinations(pool, minSize, maxSize) {
  const out = [];
  const walk = (start, acc) => {
    if (acc.length >= minSize) out.push([...acc]);
    if (acc.length === maxSize) return;
    for (let i = start; i < pool.length; i += 1) {
      acc.push(pool[i]);
      walk(i + 1, acc);
      acc.pop();
    }
  };
  walk(0, []);
  return out;
}

function assignSeverity(ids, insight, aggression) {
  const rels = ids.map((id) => relevance(id, insight));
  const max = Math.max(...rels, 0.01);
  return ids.map((id, i) => ({
    id,
    severity: clamp(Math.round(1 + 4 * (rels[i] / max) * aggression), 1, 5),
  }));
}

function dryRunDrop(samples, modelId, modelState, attacks) {
  if (!samples.length) return 0;
  let sum = 0;
  for (const s of samples) sum += runAttack(s, modelId, modelState, attacks).drop;
  return sum / samples.length;
}

/** Live estimate for a (possibly user-edited) combo against an insight's samples. */
export function predictCombo(samples, modelId, modelState, attacks) {
  return {
    drop: dryRunDrop(samples, modelId, modelState, attacks),
    fidelity: comboFidelity(attacks),
  };
}

const jaccard = (a, b) => {
  const A = new Set(a.map((x) => x.id));
  const B = new Set(b.map((x) => x.id));
  let inter = 0;
  for (const x of A) if (B.has(x)) inter += 1;
  return inter / (A.size + B.size - inter);
};

/**
 * Propose ≥2-attack combinations for one insight.  Candidates come from the
 * affinity heuristic and from RL memory (previously successful recipes);
 * the final ranking mixes predicted degradation, label fidelity and the
 * learned Q-value with a UCB exploration bonus, plus one forced explorer.
 */
export function recommendCombos({ insight, samples, modelId, modelState, memory, aggression = "moderate", seedSalt = 0 }) {
  const agg = AGGRESSION[aggression] ?? AGGRESSION.moderate;
  const memEntries = memory?.[insight.tag] ? Object.values(memory[insight.tag]) : [];
  const totalN = memEntries.reduce((a, e) => a + e.n, 0);

  const pool = [...ATTACKS]
    .map((a) => ({ id: a.id, rel: relevance(a.id, insight) }))
    .sort((a, b) => b.rel - a.rel)
    .slice(0, 6)
    .map((a) => a.id);

  const candidates = new Map();
  const register = (attacks, source) => {
    const sig = comboSignature(attacks);
    if (candidates.has(sig)) return;
    const drop = dryRunDrop(samples, modelId, modelState, attacks);
    const fidelity = comboFidelity(attacks);
    const mem = memory?.[insight.tag]?.[sig] || null;
    const q = mem ? memoryEntryQ(mem) : 0;
    const ucb = 0.1 * Math.sqrt(Math.log(totalN + 2) / ((mem?.n || 0) + 1));
    const score = 0.68 * clamp(drop / 0.55) + 0.22 * fidelity + 0.4 * q + ucb;
    candidates.set(sig, { sig, attacks, drop, fidelity, mem, q, score, source });
  };

  for (const ids of combinations(pool, 2, 4)) register(assignSeverity(ids, insight, agg), "heuristic");
  for (const entry of memEntries) {
    if (entry.attacks.length >= 2) register(entry.attacks, "rl");
  }
  // learned recipes keep their provenance even if the heuristic produced the same set
  for (const c of candidates.values()) if (c.mem && c.mem.n > 0) c.source = "rl";

  const ranked = [...candidates.values()].sort((a, b) => b.score - a.score);
  const chosen = [];
  for (const c of ranked) {
    if (chosen.every((x) => jaccard(x.attacks, c.attacks) <= 0.67)) chosen.push(c);
    if (chosen.length === 2) break;
  }

  // forced explorer keeps the bandit learning
  const rng = makeRng(`${insight.tag}:${totalN}:${seedSalt}`);
  const size = 2 + Math.floor(rng() * 3);
  const explorerIds = [...pool].sort(() => rng() - 0.5).slice(0, size);
  const explorer = assignSeverity(explorerIds, insight, agg);
  let explorerCand = ranked.find((c) => comboSignature(c.attacks) === comboSignature(explorer));
  if (!explorerCand) {
    register(explorer, "explore");
    explorerCand = candidates.get(comboSignature(explorer));
  }
  if (chosen.every((x) => jaccard(x.attacks, explorerCand.attacks) <= 0.8)) {
    chosen.push({ ...explorerCand, source: explorerCand.mem?.n ? "rl" : "explore" });
  } else {
    const next = ranked.find((c) => chosen.every((x) => jaccard(x.attacks, c.attacks) <= 0.67));
    if (next) chosen.push(next);
  }

  const maxDrop = Math.max(...chosen.map((c) => c.drop));
  const maxFid = Math.max(...chosen.map((c) => c.fidelity));
  return chosen.map((c, i) => ({
    id: `${insight.tag}-${i}`,
    tag: insight.tag,
    sig: c.sig,
    attacks: c.attacks,
    predictedDrop: c.drop,
    fidelity: c.fidelity,
    source: c.source,
    mem: c.mem,
    role: c.drop === maxDrop ? "Max impact" : c.fidelity === maxFid ? "High fidelity" : "Balanced",
    recommended: i === 0,
  }));
}

// ───────────────────────────── Stress evaluation ───────────────────────────

/**
 * Attack every sample with the plan matching its dominant failure tag and
 * report accuracy — used for a like-for-like before/after-retrain comparison.
 */
export function stressEvaluate(evalResult, modelId, modelState, planByTag) {
  const model = MODEL_BY_ID[modelId];
  const perTag = {};
  let okAfter = 0;
  let attacked = 0;
  for (const s of evalResult.scored) {
    const attacks = planByTag[s.attrTag];
    const sample = evalResult.scored[s.index];
    let after = s.score;
    if (attacks?.length) {
      const conds2 = attackedConds(sample.conds, attacks);
      after = Math.min(s.score, scoreConds(conds2, model, modelState, sample.eps));
      attacked += 1;
    }
    const slot = (perTag[s.attrTag] ||= { n: 0, okBefore: 0, okAfter: 0, sumAfter: 0 });
    slot.n += 1;
    slot.okBefore += s.correct ? 1 : 0;
    slot.okAfter += after >= ERR_CORRECT ? 1 : 0;
    slot.sumAfter += after;
    okAfter += after >= ERR_CORRECT ? 1 : 0;
  }
  const tags = Object.entries(perTag).map(([tag, v]) => ({
    tag,
    n: v.n,
    accBefore: v.okBefore / v.n,
    accAfter: v.okAfter / v.n,
    meanAfter: v.sumAfter / v.n,
  }));
  return { accuracy: okAfter / evalResult.n, attacked, perTag: tags };
}

// ───────────────────────────── Retraining ──────────────────────────────────

export const DEFAULT_RETRAIN_CFG = { epochs: 12, advRatio: 0.5 };

export function planFromDb(dbEntries) {
  // Most-used recipe per tag becomes the stress plan for evaluation.
  const byTag = {};
  for (const e of dbEntries) {
    byTag[e.tag] ||= {};
    byTag[e.tag][e.sig] ||= { count: 0, attacks: e.combo };
    byTag[e.tag][e.sig].count += 1;
  }
  const plan = {};
  for (const [tag, sigs] of Object.entries(byTag)) {
    plan[tag] = Object.values(sigs).sort((a, b) => b.count - a.count)[0].attacks;
  }
  return plan;
}

export function simulateRetrain({ modelId, datasetId, modelState, dbEntries, cfg = DEFAULT_RETRAIN_CFG, plan }) {
  const model = MODEL_BY_ID[modelId];
  const beforeEval = evaluate(modelId, datasetId, modelState);
  const beforeStress = stressEvaluate(beforeEval, modelId, modelState, plan);

  const coverage = {};
  const counts = {};
  for (const e of dbEntries) {
    coverage[e.tag] = (coverage[e.tag] || 0) + 0.6 + 0.8 * clamp(e.drop);
    counts[e.tag] = (counts[e.tag] || 0) + 1;
  }
  const mixQ = clamp(1 - 1.6 * Math.pow(cfg.advRatio - 0.45, 2), 0.55, 1);
  const epochF = 1 - Math.exp(-cfg.epochs / 6);
  const fixes = { ...(modelState.fixes || {}) };
  const perTagFix = [];
  for (const tag of INSIGHT_TAGS) {
    const before = fixes[tag] || 0;
    const cov = coverage[tag] || 0;
    const gain = 0.82 * (1 - Math.exp(-cov / 16)) * mixQ * epochF;
    const after = clamp(before + (1 - before) * gain, 0, 0.92);
    fixes[tag] = after;
    perTagFix.push({ tag, entries: counts[tag] || 0, fixBefore: before, fixAfter: after });
  }
  const forgetting = Math.max(0, cfg.advRatio - 0.6) * 0.05;
  const baseDelta = (modelState.baseDelta || 0) + 0.012 * mixQ * epochF - forgetting;
  const newState = {
    version: (modelState.version || 1) + 1,
    fixes,
    baseDelta,
    history: modelState.history || [],
  };

  const afterEval = evaluate(modelId, datasetId, newState);
  const afterStress = stressEvaluate(afterEval, modelId, newState, plan);

  const stressTagBefore = Object.fromEntries(beforeStress.perTag.map((t) => [t.tag, t]));
  const stressTagAfter = Object.fromEntries(afterStress.perTag.map((t) => [t.tag, t]));
  const perTag = perTagFix
    .filter((t) => t.entries > 0 || stressTagBefore[t.tag])
    .map((t) => ({
      ...t,
      stressBefore: stressTagBefore[t.tag]?.accAfter ?? null,
      stressAfter: stressTagAfter[t.tag]?.accAfter ?? null,
      cleanBefore: beforeEval.byTag.find((x) => x.tag === t.tag)?.acc ?? null,
      cleanAfter: afterEval.byTag.find((x) => x.tag === t.tag)?.acc ?? null,
    }));

  const rewards = {};
  for (const t of perTag) {
    if (t.entries > 0 && t.stressBefore != null && t.stressAfter != null) {
      rewards[t.tag] = clamp((t.stressAfter - t.stressBefore) / 0.5, -0.5, 1);
    }
  }

  // Animated training curves — converge onto the true evaluated end-points.
  const rng = makeRng(`${modelId}:${datasetId}:${newState.version}`);
  const E = cfg.epochs;
  const curve = [];
  const m0 = beforeEval.metrics.map50;
  const m1 = afterEval.metrics.map50;
  const s0 = beforeStress.accuracy;
  const s1 = afterStress.accuracy;
  for (let e = 0; e <= E; e += 1) {
    const t = e === 0 ? 0 : (1 - Math.exp((-3 * e) / E)) / (1 - Math.exp(-3));
    const noise = e === 0 || e === E ? 0 : gauss(rng) * 0.006;
    curve.push({
      epoch: e,
      loss: e === 0 ? 1.42 : 0.24 + 1.18 * Math.exp((-3 * e) / E) + gauss(rng) * 0.012,
      map50: m0 + (m1 - m0) * t + noise,
      stress: s0 + (s1 - s0) * t + noise * 1.4,
    });
  }

  return {
    newState,
    before: { eval: beforeEval, stress: beforeStress },
    after: { eval: afterEval, stress: afterStress },
    perTag,
    rewards,
    curve,
    cfg,
    dbSize: dbEntries.length,
    mixQuality: mixQ,
    forgetting,
  };
}
