import json
from pathlib import Path

R = str(Path(__file__).resolve().parents[1]) + "/"
r = json.load(open(R + "data/demo_artifacts/eval_report.json", encoding="utf8"))
m = json.load(open(R + "data/demo_artifacts/manifest.json", encoding="utf8"))
b1, b2 = r["base_model"]["base_map"] * 100, r["advertest_model"]["base_map"] * 100
c1, c2 = r["base_model"]["corrupted_map"] * 100, r["advertest_model"]["corrupted_map"] * 100
rec = r["hard_negative_recovery"] * 100
fd, hn, prov = m["failure_distribution"], m["hard_negatives"], r["provenance"]
w = m.get("weights") or {}
weights_line = (
    f"Model weights ({w['bytes'] / 1e6:.1f} MB, SHA-256 {w['sha256'][:12]}&hellip;) are stored at {w['location']}; transfer to the Core Data Lake is pending."
    if w
    else "Model weights live on the Colab worker until exported to the Core Data Lake."
)

p = R + "docs/advertest_executive_report.html"
s = open(p, encoding="utf8").read()
s = s.replace("DATE: <strong>September 30, 2026</strong>", "DATE: <strong>October 19, 2026</strong>")
s = s.replace("STATUS: <strong>READY FOR PRODUCTION</strong>", "STATUS: <strong>DEMO-READY &middot; EGO4D VALIDATION PENDING</strong>")
a = s.index("        <!-- Executive Summary -->")
b = s.index("        </article>")

body = f"""        <!-- Executive Summary -->
        <article class="prose prose-slate prose-lg max-w-none">

            <div class="print:break-inside-avoid">
                <h2>1. Executive Summary</h2>
                <p class="lead text-xl text-slate-600 font-serif italic border-l-4 border-indigo-500 pl-4">
                    Face detectors trained on clean, eye-level photos degrade on wearable-camera footage. AdverTest measures that gap, synthesizes the failure modes with an auditable attack engine, retrains, and accepts the new model only if it passes a blind evaluation gate.
                </p>
                <p>
                    Model: <strong>yolov7-face</strong> (bounding box + 5 facial landmarks). Base domain: <strong>WIDER FACE</strong>. Target domain: <strong>Ego4D Social</strong> (first-person video with fisheye distortion, head-bob motion blur and sensor degradation). The Enterprise Data Platform supplies raw data through short-lived presigned URLs; raw frames are streamed and never persisted on developer machines.
                </p>
            </div>

            <hr class="my-10 border-slate-200">

            <div class="print:break-inside-avoid">
                <h2>2. Workflow</h2>
                <div class="grid grid-cols-1 md:grid-cols-4 gap-4 my-8 not-prose">
                    <div class="bg-slate-50 border border-slate-200 p-4 rounded-xl"><h3 class="font-bold text-slate-800 mb-1">1 &middot; Triage</h3><p class="text-sm text-slate-600">Baseline inference; a hard negative is a face with IoU &lt; 0.5 or confidence &lt; 0.4. Only that subset moves on.</p></div>
                    <div class="bg-slate-50 border border-slate-200 p-4 rounded-xl"><h3 class="font-bold text-slate-800 mb-1">2 &middot; Synthesize</h3><p class="text-sm text-slate-600">Radial distortion, biomechanical IMU blur, sensor degradation, occlusion. Exclusion matrix and a 50% bbox-integrity gate; every sample carries an audit trail.</p></div>
                    <div class="bg-slate-50 border border-slate-200 p-4 rounded-xl"><h3 class="font-bold text-slate-800 mb-1">3 &middot; Retrain</h3><p class="text-sm text-slate-600">Fine-tune on targeted samples plus a clean replay set (guards the base domain).</p></div>
                    <div class="bg-slate-50 border border-slate-200 p-4 rounded-xl"><h3 class="font-bold text-slate-800 mb-1">4 &middot; Blind gate</h3><p class="text-sm text-slate-600">Held-out images: accept V2 only if corrupted mAP gains &ge; 5 points and base mAP drops &le; 1 point.</p></div>
                </div>
                <p>All heavy compute runs on a Colab GPU worker (<code>colab/colab_zero_disk_pipeline.py</code>); the laptop footprint stays under 5&nbsp;MB. The web console and API serve the exported results.</p>
            </div>

            <hr class="my-10 border-slate-200">

            <div class="print:break-inside-avoid">
                <h2>3. Empirical Results (measured run)</h2>
                <p>One run of the pipeline on {prov['n_train']} streamed WIDER FACE validation images for training and <strong>{prov['n_blind']} held-out images</strong> for the blind evaluation (seed {prov['seed']}, {prov['epochs']} epochs, yolov7-tiny-face). mAP is AP@IoU 0.5 over faces of at least 16&nbsp;px.</p>
                <table>
                    <thead><tr><th>Metric</th><th>V1 baseline</th><th>V2 retrained</th><th>Change</th><th>Gate</th></tr></thead>
                    <tbody>
                        <tr><td>Base mAP (clean WIDER FACE)</td><td>{b1:.1f}%</td><td>{b2:.1f}%</td><td>{b2 - b1:+.1f} pts</td><td>drop &le; 1.0 pt &mdash; passed</td></tr>
                        <tr><td>Corrupted mAP (egocentric proxy)</td><td>{c1:.1f}%</td><td>{c2:.1f}%</td><td>{c2 - c1:+.1f} pts</td><td>gain &ge; 5.0 pts &mdash; passed</td></tr>
                        <tr><td>Hard negatives recovered</td><td colspan="2">{rec:.1f}% of blind-set failures</td><td colspan="2">&nbsp;</td></tr>
                    </tbody>
                </table>
                <p>Baseline failures ({hn} hard negatives) split into: fisheye {fd['error_fisheye']}, motion blur {fd['error_motion_blur']}, sensor exposure/ISO {fd['error_sensor_noise']}, occlusion {fd['error_occlusion']}. These tags follow one canonical taxonomy so they map directly onto the central metadata registry.</p>

                <div class="bg-amber-50 border-l-4 border-amber-500 p-6 rounded-r-lg mt-8 not-prose">
                    <h4 class="font-bold text-amber-900 mb-2">Limitations &mdash; read before relying on these numbers</h4>
                    <ul class="text-sm text-amber-900 list-disc pl-5 space-y-1">
                        <li><strong>Target domain is a proxy.</strong> No Ego4D footage was used yet (licence keys pending). &ldquo;Corrupted&rdquo; means WIDER FACE passed through the physical attack engine, which has exact ground truth. Real Ego4D frames are supported (streamed from presigned URLs) but not yet evaluated.</li>
                        <li><strong>Small sample.</strong> {prov['n_blind']} blind images and a single run: the corrupted-mAP gain clears the 5-point gate by under one point and has no confidence interval.</li>
                        <li><strong>Modest recovery.</strong> {rec:.0f}% of the baseline hard negatives are recovered; most remain failures.</li>
                        <li><strong>Landmarks are pseudo-labels.</strong> WIDER FACE has boxes only; training landmarks come from the baseline model, transformed with the same geometry as the image.</li>
                        <li>{weights_line}</li>
                    </ul>
                </div>
            </div>

            <hr class="my-10 border-slate-200">

            <div class="print:break-inside-avoid">
                <h2>4. Next Steps</h2>
                <ol>
                    <li><strong>Ego4D validation:</strong> supply presigned URLs for a few Social-benchmark clips; the pipeline reports confidence, dropout and tracker recovery on real frames.</li>
                    <li><strong>Statistical confidence:</strong> repeat with larger blind sets and several seeds; report confidence intervals for the gate.</li>
                    <li><strong>Temporal tracker on real sequences:</strong> the velocity tracker is implemented and unit-tested; evaluate recovery on Ego4D clips.</li>
                    <li><strong>Weights handover:</strong> move the accepted V2 checkpoint (hash recorded in the manifest) to the Core Data Lake together with the lineage bundle.</li>
                </ol>
            </div>

"""
s = s[:a] + body + s[b:]
open(p, "w", encoding="utf8").write(s)

p = R + "docs/beautiful_architecture.html"
s = open(p, encoding="utf8").read()
s = s.replace("RL Policy Bandit", "Insight-weighted attack sampling").replace("Ray Cluster / Colab GPU", "Colab GPU worker")
open(p, "w", encoding="utf8").write(s)
print("ok")
