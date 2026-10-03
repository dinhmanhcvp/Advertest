import { describe, expect, it } from "vitest";
import { DATASETS, MODELS } from "../catalog";
import {
  applyRewards,
  evaluate,
  extractInsights,
  freshModelState,
  planFromDb,
  recommendCombos,
  recordMemory,
  runAttack,
  simulateRetrain,
  summarizeAttack,
  triage,
} from "../engine";

const model = MODELS[0].id;
const dataset = DATASETS[0].id;

describe("closed-loop engine", () => {
  it("is deterministic and yields plausible baseline accuracy", () => {
    const a = evaluate(model, dataset, freshModelState());
    const b = evaluate(model, dataset, freshModelState());
    expect(a.metrics.accuracy).toBe(b.metrics.accuracy);
    for (const m of MODELS) {
      for (const d of DATASETS) {
        const r = evaluate(m.id, d.id, freshModelState());
        console.log(m.id, d.id, r.metrics.accuracy.toFixed(3), r.metrics.map50.toFixed(3));
        expect(r.metrics.accuracy).toBeGreaterThan(0.45);
        expect(r.metrics.accuracy).toBeLessThan(0.97);
      }
    }
  });

  it("proposes multi-attack combos (>= 2) and attacks reduce score", () => {
    const state = freshModelState();
    const ev = evaluate(model, dataset, state);
    const worst = triage(ev, 16);
    const insights = extractInsights(worst);
    expect(insights.length).toBeGreaterThan(0);
    for (const insight of insights) {
      const group = worst.filter((s) => insight.sampleIds.includes(s.id));
      const combos = recommendCombos({ insight, samples: group, modelId: model, modelState: state, memory: {} });
      expect(combos.length).toBeGreaterThanOrEqual(3);
      for (const c of combos) {
        expect(c.attacks.length).toBeGreaterThanOrEqual(2);
        const results = group.map((s) => runAttack(s, model, state, c.attacks));
        const sum = summarizeAttack(results);
        expect(sum.scoreAfter).toBeLessThanOrEqual(sum.scoreBefore);
      }
    }
  });

  it("retraining improves stress accuracy and feeds RL rewards", () => {
    let state = freshModelState();
    const ev = evaluate(model, dataset, state);
    const worst = triage(ev, 40);
    const insights = extractInsights(worst);
    let memory = {};
    const db = [];
    for (const insight of insights) {
      const group = worst.filter((s) => insight.sampleIds.includes(s.id));
      const [combo] = recommendCombos({ insight, samples: group, modelId: model, modelState: state, memory });
      for (const s of group) {
        const r = runAttack(s, model, state, combo.attacks);
        memory = recordMemory(memory, insight.tag, combo.attacks, r);
        db.push({ id: s.id, tag: insight.tag, sig: combo.sig, combo: combo.attacks, drop: r.drop });
      }
    }
    const plan = planFromDb(db);
    const out = simulateRetrain({ modelId: model, datasetId: dataset, modelState: state, dbEntries: db, plan });
    console.log(
      "stress",
      out.before.stress.accuracy.toFixed(3),
      "->",
      out.after.stress.accuracy.toFixed(3),
      "clean",
      out.before.eval.metrics.accuracy.toFixed(3),
      "->",
      out.after.eval.metrics.accuracy.toFixed(3),
    );
    expect(out.after.stress.accuracy).toBeGreaterThan(out.before.stress.accuracy);
    expect(out.after.eval.metrics.accuracy).toBeGreaterThanOrEqual(out.before.eval.metrics.accuracy);
    const next = applyRewards(memory, out.rewards, db);
    const anyRewarded = Object.values(next).some((t) => Object.values(t).some((e) => e.nReward > 0));
    expect(anyRewarded).toBe(true);
    state = out.newState;
    expect(state.version).toBe(2);
  });
});
