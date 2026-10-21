"""Offline tests for the pure logic of colab/colab_zero_disk_pipeline.py.

No weights, datasets or torch are needed: every test uses small in-memory arrays.
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "colab_pipeline", Path(__file__).resolve().parents[1] / "colab" / "colab_zero_disk_pipeline.py"
)
cp = importlib.util.module_from_spec(_spec)
sys.modules["colab_pipeline"] = cp  # dataclasses resolves string annotations via sys.modules
_spec.loader.exec_module(cp)


def test_iou_and_ap():
    assert cp.iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert cp.iou_xyxy((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    gts = [[(0, 0, 10, 10), (50, 50, 60, 60)]]
    perfect = [[cp.Det((0, 0, 10, 10), 0.9), cp.Det((50, 50, 60, 60), 0.8)]]
    half = [[cp.Det((0, 0, 10, 10), 0.9)]]
    assert cp.average_precision(perfect, gts) == pytest.approx(1.0)
    assert cp.average_precision(half, gts) == pytest.approx(0.5)
    assert cp.average_precision([[]], gts) == 0.0


def test_triage_rule():
    gt = [(0, 0, 10, 10)]
    assert not cp.triage_image(gt, [cp.Det((0, 0, 10, 10), 0.9)])["hard"]
    assert cp.triage_image(gt, [cp.Det((0, 0, 10, 10), 0.3)])["hard"]  # weak confidence
    assert cp.triage_image(gt, [cp.Det((40, 40, 50, 50), 0.9)])["hard"]  # missed
    assert cp.triage_image(gt, [])["hard"]


def test_distortion_points_follow_pixels():
    h, w = 240, 320
    img = np.zeros((h, w, 3), np.uint8)
    pt = (240, 70)
    img[pt[1] - 1 : pt[1] + 2, pt[0] - 1 : pt[0] + 2] = 255
    k1, k2 = 0.3, 0.1
    out = cp.distort_image(img, k1, k2)
    ys, xs = np.nonzero(out[..., 0] > 100)
    assert len(xs) > 0
    px, py = cp.distort_points([pt], w, h, k1, k2)[0]
    assert abs(xs.mean() - px) < 3 and abs(ys.mean() - py) < 3


def test_integrity_discards_pushed_out_boxes():
    h, w = 240, 320
    rng = np.random.default_rng(0)
    img = np.full((h, w, 3), 128, np.uint8)
    inner = cp.synthesize(img, [(130, 90, 190, 150)], [None], "error_fisheye", 3, rng)
    assert not inner.discarded and inner.integrity > 0.9
    assert cp.inside_fraction((310, 100, 350, 140), w, h) < 0.5


def test_imu_kernel_is_normalised_and_nonlinear():
    k = cp.imu_kernel(4, 3.0, 10.0)
    assert k.sum() == pytest.approx(1.0, abs=1e-4)
    ys, _ = np.nonzero(k > 1e-4)
    assert ys.max() - ys.min() >= 3  # not a straight horizontal line


def test_exclusion_matrix():
    assert not cp.exclusion_ok(["flare", "low_light"])
    assert cp.exclusion_ok(["tone_curve", "flare"])
    assert cp.exclusion_ok(["low_light", "high_iso"])


@pytest.mark.parametrize("tag", cp.CANONICAL_TAGS)
def test_every_tag_changes_the_image(tag):
    rng = np.random.default_rng(1)
    img = np.random.default_rng(2).integers(60, 200, (240, 320, 3), dtype=np.uint8)
    att = cp.synthesize(img, [(120, 80, 200, 170)], [np.array([[140, 100], [180, 100], [160, 125], [145, 150], [175, 150]])], tag, 4, rng)
    assert att.chain == cp.TAG_CHAIN[tag]
    assert att.image.shape == img.shape and not np.array_equal(att.image, img)
    assert att.lmk[0] is not None and len(att.boxes) == 1


def test_classify_failure_from_statistics():
    rng = np.random.default_rng(3)
    sharp = rng.integers(0, 255, (100, 100, 3), dtype=np.uint8)
    assert cp.classify_failure(sharp, (10, 10, 90, 90)) in cp.CANONICAL_TAGS
    flat = np.full((100, 100, 3), 120, np.uint8)
    assert cp.classify_failure(flat, (10, 10, 90, 90)) == "error_motion_blur"
    bright = np.full((100, 100, 3), 250, np.uint8)
    assert cp.classify_failure(bright, (10, 10, 90, 90)) == "error_sensor_noise"
    assert cp.classify_failure(flat, (10, 10, 90, 90), {"blur": 2}) == "error_motion_blur"


def test_tracker_recovers_dropout_with_velocity():
    seq = [[cp.Det((10 + 5 * i, 20, 50 + 5 * i, 80), 0.9)] for i in range(5)]
    seq[2] = []  # detector lost the face on frame t
    out = cp.track_sequence(seq)
    assert out[2]["recovered"] and out[2]["tracked"]
    assert abs(out[2]["box"][0] - 20) < 2.5  # linear extrapolation of the x1 velocity
    assert not out[3]["recovered"]


def test_blind_gate():
    assert cp.blind_gate(0.914, 0.911, 0.482, 0.768, 0.01, 0.05)["accepted"]
    assert not cp.blind_gate(0.914, 0.85, 0.482, 0.9, 0.01, 0.05)["accepted"]  # forgot the base domain
    assert not cp.blind_gate(0.914, 0.914, 0.482, 0.49, 0.01, 0.05)["accepted"]  # no real gain


def test_reference_frame_mapping():
    assert cp.to_ref((0, 0, 640, 480), 640, 480) == [0.0, 0.0, 160.0, 120.0]
    x, y, w, h = cp.to_ref((0, 0, 100, 100), 400, 100)  # wide image gets letter-boxed vertically
    assert y > 0 and w == pytest.approx(40.0)


def test_presigned_url_ttl():
    assert cp.url_seconds_remaining("https://x/y?Expires=1") < 0
    soon = int(time.time()) + 600
    assert cp.url_seconds_remaining(f"https://x/y?Expires={soon}") > 0
    assert cp.url_seconds_remaining("https://x/y") is None


def test_wider_dir_loader_parses_official_annotation_format(tmp_path):
    import cv2

    img_dir = tmp_path / "WIDER_val" / "images" / "0--Parade"
    img_dir.mkdir(parents=True)
    split = tmp_path / "wider_face_split"
    split.mkdir()
    cv2.imwrite(str(img_dir / "a.jpg"), np.full((120, 160, 3), 90, np.uint8))
    cv2.imwrite(str(img_dir / "b.jpg"), np.full((120, 160, 3), 90, np.uint8))
    # x y w h blur expression illumination invalid occlusion pose
    (split / "wider_face_val_bbx_gt.txt").write_text(
        "0--Parade/a.jpg\n3\n10 10 40 40 2 0 1 0 0 0\n60 60 8 8 0 0 0 0 0 0\n80 20 30 30 0 0 0 1 0 0\n"
        "0--Parade/b.jpg\n0\n0 0 0 0 0 0 0 0 0 0\n"
    )
    out = list(cp._wider_from_dir(tmp_path, 5))
    assert len(out) == 1  # b.jpg has no faces; tiny and invalid faces in a.jpg are dropped
    img, boxes, attrs = out[0]
    assert boxes == [(10, 10, 50, 50)]
    assert attrs[0] == {"blur": 2, "illumination": 1, "occlusion": 0}


def test_decode_yolov7_face_rows_back_to_image_pixels():
    # image 320x240 letterboxed into 640: scale 2, pad (0, 80)
    s, px, py = 2.0, 0.0, 80.0
    row = [100, 180, 300, 380, 0.9, 0] + [x for i in range(5) for x in (120 + 20 * i, 200 + 10 * i, 0.99)]
    d = cp.decode_detections(np.array([row], np.float32), s, px, py, 320, 240)[0]
    assert d.box == (50.0, 50.0, 150.0, 150.0) and d.score == pytest.approx(0.9)
    assert d.lmk.shape == (5, 2)
    assert tuple(d.lmk[0]) == (60.0, 60.0) and tuple(d.lmk[4]) == (100.0, 80.0)
    with pytest.raises(RuntimeError):
        cp.decode_detections(np.zeros((1, 16), np.float32), s, px, py, 320, 240)


def test_label_rows_match_yolov7_face_loader_contract():
    lm = np.array([[30, 30], [50, 30], [40, 40], [32, 50], [48, 50]], float)
    rows = cp.yolo_label_rows([(20, 20, 60, 60), (200, 0, 260, 10)], [lm, None], 100, 100)
    vals = [float(v) for v in rows[0].split()]
    assert len(vals) == 5 + 3 * 5  # class, box, 5 x (x, y, visibility)
    coords = vals[1:5] + [v for k, v in enumerate(vals[5:]) if k % 3 != 2]  # x/y columns; visibility is exempt
    assert all(v >= 0 for v in vals) and all(v <= 1 for v in coords)  # the loader rejects negatives / >1
    assert vals[7] == 2.0 and vals[5] == pytest.approx(0.3)
    assert len(rows) == 1  # second box lies outside the 100x100 frame -> dropped
    missing = cp.yolo_label_rows([(20, 20, 60, 60)], [None], 100, 100)[0].split()
    assert missing[5:] == ["0.000000"] * 15 or all(float(v) == 0 for v in missing[5:])
    dup = cp.yolo_label_rows([(20, 20, 60, 60), (20, 20, 60, 60)], [None, None], 100, 100)
    assert len(dup) == 1  # duplicate rows are asserted against by the loader


def test_run_logged_streams_output_and_reports_the_tail(capsys):
    import sys as _sys

    cp.run_logged([_sys.executable, "-c", "print('hello from child')"])
    assert "hello from child" in capsys.readouterr().out
    with pytest.raises(cp.CommandFailed) as ei:
        cp.run_logged([_sys.executable, "-c", "import sys; print('boom detail'); sys.exit(3)"])
    assert "boom detail" in str(ei.value) and "exited with 3" in str(ei.value)


def test_run_logged_passes_extra_env():
    import sys as _sys

    with pytest.raises(cp.CommandFailed) as ei:
        cp.run_logged([_sys.executable, "-c", "import os,sys; print(os.environ['X_FLAG']); sys.exit(1)"], env={"X_FLAG": "on"})
    assert "on" in str(ei.value)


def test_numpy_compat_shim_restores_removed_aliases(tmp_path, capsys):
    import sys as _sys

    script = tmp_path / "train.py"
    script.write_text(
        "import numpy as np, sys\n"
        "print('argv', sys.argv[1:])\n"
        "print('floor', np.floor(np.arange(5) / 2).astype(np.int).tolist(), np.float(1.5), np.bool(1))\n"
    )
    cp.run_logged([_sys.executable, "-c", cp.NUMPY_COMPAT_SHIM, "train.py", "--epochs", "3"], cwd=tmp_path)
    out = capsys.readouterr().out
    assert "floor [0, 0, 1, 1, 2] 1.5 True" in out and "argv ['--epochs', '3']" in out


def test_shim_blocks_wandb_and_stubs_repo_validation(tmp_path, capsys):
    import sys as _sys

    (tmp_path / "train.py").write_text(
        "try:\n    import wandb\n    print('wandb importable')\nexcept ImportError:\n    print('wandb blocked')\n"
        "import test\n"
        "res, maps, times = test.test({}, batch_size=2)\n"
        "print('val', len(res), len(maps), len(times))\n"
    )
    cp.run_logged([_sys.executable, "-c", cp.NUMPY_COMPAT_SHIM, "train.py"], cwd=tmp_path)
    out = capsys.readouterr().out
    assert "wandb blocked" in out and "val 7 1 3" in out


def test_crop_around_zooms_on_the_face_and_keeps_boxes_inside():
    img = np.zeros((600, 800, 3), np.uint8)
    gb = (400.0, 300.0, 420.0, 325.0)  # a tiny crowd-scene face
    crop, x0, y0 = cp.crop_around(img, gb)
    h, w = crop.shape[:2]
    sb = cp.shift_box(gb, x0, y0)
    assert w <= 130 and h <= 100 and w >= 90  # zoomed window, not the whole 800x600 frame
    assert 0 <= sb[0] < sb[2] <= w and 0 <= sb[1] < sb[3] <= h
    ref = cp.to_ref(sb, w, h)
    assert ref[2] > 8  # the face is no longer a few reference units wide


def test_export_weights_records_hash_and_size(tmp_path):
    import hashlib

    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"fake-checkpoint" * 100)
    info = cp.export_weights(ckpt, tmp_path, accepted=True)
    assert info["sha256"] == hashlib.sha256(ckpt.read_bytes()).hexdigest()
    assert info["bytes"] == 1500 and info["accepted_by_gate"] is True
    assert "worker-only" in info["location"]  # outside Colab and without a PUT URL nothing is uploaded


# ---- mentor Q&A coverage: reproducibility, leakage, FPPI, ghost boxes, landmark geometry -------------------------


def test_any_synthesized_sample_is_reproducible_from_its_record():
    img = np.random.default_rng(5).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    boxes = [(120.0, 80.0, 200.0, 170.0)]
    lm = [np.array([[140, 100], [180, 100], [160, 125], [145, 150], [175, 150]], float)]
    tag, sev = cp.draw_attack(7, ("adv", 3))
    a = cp.apply_attack(img, boxes, lm, 7, ("adv", 3), tag, sev)
    record = {"reproduce": {"seed": 7, "key": ["adv", 3], "tag": tag, "severity": sev}}
    b = cp.reproduce_sample(img, boxes, lm, record)
    assert np.array_equal(a.image, b.image) and a.boxes == b.boxes and a.params == b.params
    other = cp.apply_attack(img, boxes, lm, 7, ("adv", 4), tag, sev)
    assert other.params != a.params or not np.array_equal(other.image, a.image)  # different key => different draw


def test_draw_attack_is_deterministic_and_respects_weights():
    assert cp.draw_attack(1, ("k", 1)) == cp.draw_attack(1, ("k", 1))
    only_blur = [0, 1, 0, 0]
    assert all(cp.draw_attack(1, ("k", i), weights=only_blur)[0] == "error_motion_blur" for i in range(20))


def test_leak_guard_finds_near_duplicates_only():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    near = np.clip(a.astype(int) + rng.integers(-3, 3, a.shape), 0, 255).astype(np.uint8)  # same scene, sensor noise
    far = rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
    assert cp.find_leaks([near, far], [a]) == {0}


def test_recall_at_fppi_is_an_operating_point_metric():
    gts = [[(0, 0, 10, 10)], [(0, 0, 10, 10)]]
    dets = [
        [cp.Det((0, 0, 10, 10), 0.9)],
        [cp.Det((50, 50, 60, 60), 0.8), cp.Det((0, 0, 10, 10), 0.4)],  # a confident false positive outranks the hit
    ]
    out = cp.recall_at_fppi(dets, gts, targets=(0.0, 0.4, 0.5))
    assert out["recall@fppi=0.0"] == pytest.approx(0.5)  # only the first hit before any false positive
    assert out["recall@fppi=0.4"] == pytest.approx(0.5)  # the second hit costs 0.5 FP/image: over budget
    assert out["recall@fppi=0.5"] == pytest.approx(1.0)  # one FP over two images is allowed at 0.5


def test_tracker_does_not_chase_a_ghost():
    seq = [[cp.Det((10 + 8 * i, 20, 50 + 8 * i, 80), 0.9)] for i in range(4)] + [[] for _ in range(8)]
    out = cp.track_sequence(seq)
    coasted = [o for o in out if o["recovered"]]
    assert 0 < len(coasted) <= 4  # max_age: the track terminates, it does not coast forever
    assert not out[-1]["tracked"]
    steps = [coasted[i + 1]["box"][0] - coasted[i]["box"][0] for i in range(len(coasted) - 1)]
    assert all(steps[i + 1] < steps[i] for i in range(len(steps) - 1))  # velocity damping: motion decays
    tr = cp.VelocityTracker(frame_wh=(100, 100))
    for i in range(3):
        tr.update([cp.Det((60 + 15 * i, 20, 90 + 15 * i, 60), 0.9)])
    assert (tr.update([])[0] is None) or tr.box is not None  # leaving the frame terminates the track


def test_landmarks_follow_the_exact_radial_model_not_bbox_interpolation():
    """Evidence for the keypoint-integrity answer: at severity 5 a face near the edge is warped non-linearly."""
    w, h = 640, 480
    k1, k2 = 0.10 * 5, 0.04 * 5  # what synthesize() uses for error_fisheye at severity 5
    box = (480.0, 120.0, 600.0, 260.0)  # right edge of a chest-mount frame
    lm = np.array([[510, 160], [570, 160], [540, 195], [515, 230], [565, 230]], float)
    exact = cp.distort_points(lm, w, h, k1, k2)
    nb = cp.box_after_distortion(box, w, h, k1, k2)
    # naive alternative: keep each landmark at the same relative position inside the transformed bbox
    rel = (lm - [box[0], box[1]]) / [box[2] - box[0], box[3] - box[1]]
    naive = [nb[0], nb[1]] + rel * [nb[2] - nb[0], nb[3] - nb[1]]
    err = np.linalg.norm(exact - naive, axis=1)
    assert err.max() > 3.0, f"bbox interpolation would already be accurate ({err.max():.1f}px); the exact map matters"
    # and the exact map agrees with where the warped pixels really go
    img = np.zeros((h, w, 3), np.uint8)
    x, y = int(lm[0][0]), int(lm[0][1])
    img[y - 1 : y + 2, x - 1 : x + 2] = 255
    out = cp.distort_image(img, k1, k2)
    ys, xs = np.nonzero(out[..., 0] > 100)
    assert abs(xs.mean() - exact[0][0]) < 3 and abs(ys.mean() - exact[0][1]) < 3


# ---- control experiment, bootstrap CIs and a full dry run of run_pipeline() ---------------------------------------


def test_bootstrap_ci_separates_a_real_gain_from_noise():
    gts = [[(0, 0, 10, 10)] for _ in range(40)]
    miss = [[] for _ in range(40)]
    hit = [[cp.Det((0, 0, 10, 10), 0.9)] for _ in range(40)]
    real = cp.bootstrap_diff_ci(miss, hit, gts, n_boot=50, seed=1)
    assert real["lo"] > 0.9 and real["mean"] == pytest.approx(1.0)
    same = cp.bootstrap_diff_ci(hit, hit, gts, n_boot=50, seed=1)
    assert same["lo"] <= 0 <= same["hi"] and same["mean"] == 0.0
    assert cp.bootstrap_diff_ci(miss, hit, gts, n_boot=50, seed=1) == real  # deterministic


class _FakeDetector:
    """Finds the bright 'face' square; confidence = its contrast. 'v2'/'ctrl' checkpoints get a small boost."""

    def __init__(self, weights, repo, img_size=640, conf=0.25, nms_iou=0.45):
        self.boost = 0.15 if "v2" in str(weights) else 0.05 if "ctrl" in str(weights) else 0.0

    def predict(self, bgr):
        import cv2

        g = cv2.GaussianBlur(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (15, 15), 0)
        _, mx, _, loc = cv2.minMaxLoc(g)
        x, y = loc
        x1, y1, x2, y2 = max(0, x - 20), max(0, y - 20), min(bgr.shape[1], x + 20), min(bgr.shape[0], y + 20)
        contrast = float(g[y1:y2, x1:x2].mean() - g.mean())
        score = float(min(1.0, max(0.0, contrast / 120.0 + self.boost)))
        lm = np.array([[x - 8, y - 6], [x + 8, y - 6], [x, y], [x - 6, y + 8], [x + 6, y + 8]], float)
        return [cp.Det((float(x1), float(y1), float(x2), float(y2)), score, lm)]


def _fake_stream(n):
    rng = np.random.default_rng(11)
    for _ in range(n):
        img = rng.integers(40, 80, (240, 320, 3), dtype=np.uint8)
        x, y = int(rng.integers(60, 240)), int(rng.integers(50, 170))
        img[y - 20 : y + 20, x - 20 : x + 20] = 255
        yield img, [(float(x - 20), float(y - 20), float(x + 20), float(y + 20))], [{}]


def test_full_pipeline_dry_run_produces_every_artifact_and_reproducible_samples(tmp_path, monkeypatch):
    import json
    import zipfile

    import cv2

    wd = tmp_path / "wd"
    (wd / "yolov7_face").mkdir(parents=True)
    (wd / "yolov7-tiny-face.pt").write_bytes(b"weights")
    for k, v in dict(workdir=str(wd), n_pool=44, n_blind=14, epochs=1, n_boot=20, corruption_repeats=2, run_control=True).items():
        monkeypatch.setitem(cp.CFG, k, v)
    monkeypatch.setattr(cp, "stream_wider", lambda n: _fake_stream(n))
    monkeypatch.setattr(cp, "FaceDetector", _FakeDetector)
    trained = []

    def fake_finetune(repo, weights, data_yaml, name, data_sig=None):
        assert Path(data_yaml).exists() and data_sig  # the training-set fingerprint is always supplied
        trained.append(name)
        p = tmp_path / f"{name}.pt"
        p.write_bytes(name.encode())
        return p

    monkeypatch.setattr(cp, "finetune", fake_finetune)
    monkeypatch.setitem(cp.CFG, "ego4d_urls", ["https://s3.example/clips/a.mp4?X-Amz-Signature=TOPSECRET"])
    monkeypatch.setattr(
        cp, "stream_ego4d", lambda urls, n, stride: iter([(urls[0], [img for img, _, _ in _fake_stream(6)])])
    )
    zpath = cp.run_pipeline()

    assert trained == ["advertest_v2", "advertest_ctrl"]
    out = zipfile.ZipFile(zpath)
    assert sorted(out.namelist()) == ["audit_trail.json", "eval_report.json", "manifest.json", "proof_of_cure.json"]
    rep = json.loads(out.read("eval_report.json"))
    assert rep["control_model"]["note"] and "v2_minus_control" in rep["stats"]["corrupted"]
    assert "gain_statistically_supported" in rep["gate"] and "beats_clean_only_control" in rep["gate"]
    assert len(rep["corruption_repeats"]["draws"]) == 2
    assert set(rep["recall_at_fppi"]) == {"clean", "corrupted"}
    ego = rep["provenance"]["ego4d_real"]
    assert ego["frames"] == 6 and ego["per_clip"][0]["clip"] == "a.mp4" and "dropout_control" in ego
    assert "proxy" in rep["provenance"]["target_domain"]  # mAP stays on the proxy domain even with real frames
    assert all("TOPSECRET" not in out.read(n).decode() for n in out.namelist())  # no signature is ever exported
    assert rep["provenance"]["v1_sha256"] and "leaks_removed" in rep["provenance"]
    manifest = json.loads(out.read("manifest.json"))
    assert manifest["weights"]["sha256"] and manifest["weights"]["bytes"] > 0

    audit = json.loads(out.read("audit_trail.json"))
    adv = [r for r in audit if r["id"].startswith("wider_adv_")]
    assert adv and all("reproduce" in r and r["audit_trail"]["parameters"] for r in audit if "reproduce" in r)
    # rebuild the first synthesized sample from its record + source image: must equal what was written to disk
    rec = adv[0]
    src_img, gts, _ = list(_fake_stream(44))[rec["reproduce"]["source"]["index"]]
    rebuilt = cp.reproduce_sample(src_img, gts, [None] * len(gts), rec)
    ok, buf = cv2.imencode(".jpg", rebuilt.image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    on_disk = cv2.imread(str(wd / "ds" / "images" / "train" / f"adv_{int(rec['id'].split('_')[-1]):04d}.jpg"))
    assert np.array_equal(cv2.imdecode(buf, cv2.IMREAD_COLOR), on_disk)
    assert rebuilt.params == rec["audit_trail"]["parameters"]


# ---- Ego4D: presigning from Colab Secrets, statistics only, no signature or frame leaves the worker --------------


def test_pick_keys_is_deterministic_and_spread():
    keys = [f"public/v1/clips/{i:03d}.mp4" for i in range(40)] + ["public/v1/clips/readme.txt"]
    a = cp.pick_keys(keys, 4)
    assert a == cp.pick_keys(list(reversed(keys)), 4) and len(a) == 4
    assert a[0].endswith("000.mp4") and a[-1] > a[0] and all(k.endswith(".mp4") for k in a)
    assert cp.pick_keys(keys[:2], 5) == sorted(keys[:2])


def test_presigning_uses_the_client_and_never_logs_urls(capsys):
    class FakeS3:
        def list_objects_v2(self, Bucket, Prefix, MaxKeys):  # noqa: N803  (mirrors the boto3 signature)
            return {"Contents": [{"Key": f"{Prefix}clip{i}.mp4"} for i in range(6)]}

        def generate_presigned_url(self, op, Params, ExpiresIn):  # noqa: N803
            return f"https://s3.example/{Params['Key']}?X-Amz-Signature=TOPSECRET&Expires={ExpiresIn}"

    urls = cp.ego4d_presigned_urls(3, expires=600, client=FakeS3())
    assert len(urls) == 3 and all("TOPSECRET" in u for u in urls)
    assert "TOPSECRET" not in capsys.readouterr().out  # only object names are logged


def test_aws_credentials_error_explains_where_keys_go(monkeypatch):
    monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
    monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
    with pytest.raises(RuntimeError, match="Colab Secrets"):
        cp.aws_credentials()


def test_ego_stats_counts_dropouts_fixed_by_v2():
    face = [cp.Det((10, 10, 40, 40), 0.9)]
    clip = dict(
        name="a.mp4", wh=(320, 240),
        v1=[face, [], [], face],      # two dropouts
        v2=[face, face, [], face],    # V2 fixes one of them
        control=[face, [], [], face],
    )
    st = cp.ego_stats([clip])
    assert st["frames"] == 4 and st["dropout_v1"] == 0.5 and st["dropout_v2"] == 0.25
    assert st["v1_dropouts_fixed_by_v2"] == 0.5
    assert st["per_clip"][0]["clip"] == "a.mp4" and st["dropout_control"] == 0.5
    assert cp.ego_stats([]) is None


# ---- resilience: reuse a trained checkpoint, and never lose a run to an optional stage ---------------------------


def _fake_repo(tmp_path):
    repo = tmp_path / "repo"
    (repo / "cfg").mkdir(parents=True)
    (repo / "data").mkdir()
    (repo / "cfg" / "yolov7-tiny-face.yaml").write_text("x")
    (repo / "data" / "hyp.scratch.tiny.yaml").write_text("x")
    return str(repo)


def test_finetune_reuses_a_matching_checkpoint_and_retrains_when_data_changes(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    calls = []

    def fake_run(cmd, cwd=None, env=None, **kw):
        calls.append(cmd)
        w = Path(repo, "runs", "train", "m", "weights")
        w.mkdir(parents=True, exist_ok=True)
        (w / "best.pt").write_bytes(b"ckpt")

    monkeypatch.setattr(cp, "run_logged", fake_run)
    a = cp.finetune(repo, "w.pt", "d.yaml", "m", data_sig="sigA")
    b = cp.finetune(repo, "w.pt", "d.yaml", "m", data_sig="sigA")
    assert a == b and len(calls) == 1  # second call reused the stamped checkpoint
    cp.finetune(repo, "w.pt", "d.yaml", "m", data_sig="sigB")
    assert len(calls) == 2  # different training data => retrain, never a stale model


def test_unstamped_checkpoint_needs_an_explicit_opt_in(tmp_path, monkeypatch):
    repo = _fake_repo(tmp_path)
    w = Path(repo, "runs", "train", "m", "weights")
    w.mkdir(parents=True)
    (w / "last.pt").write_bytes(b"old")
    calls = []
    monkeypatch.setattr(cp, "run_logged", lambda *a, **k: (calls.append(1), (w / "best.pt").write_bytes(b"new")))
    monkeypatch.setenv("REUSE_CHECKPOINTS", "1")
    assert cp.finetune(repo, "w.pt", "d.yaml", "m", data_sig="x").name == "last.pt" and not calls
    monkeypatch.delenv("REUSE_CHECKPOINTS")
    cp.finetune(repo, "w.pt", "d.yaml", "m", data_sig="x")
    assert calls  # without the opt-in an unstamped checkpoint is not trusted


def test_files_signature_changes_with_content_size(tmp_path):
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    a.write_bytes(b"1")
    b.write_bytes(b"22")
    s1 = cp.files_signature([a, b])
    assert s1 == cp.files_signature([b, a])
    b.write_bytes(b"333")
    assert cp.files_signature([a, b]) != s1


def test_a_failing_ego4d_stage_keeps_the_run_and_hides_the_signature(tmp_path, monkeypatch):
    import json
    import zipfile

    wd = tmp_path / "wd"
    (wd / "yolov7_face").mkdir(parents=True)
    (wd / "yolov7-tiny-face.pt").write_bytes(b"weights")
    for k, v in dict(workdir=str(wd), n_pool=44, n_blind=14, epochs=1, n_boot=10, corruption_repeats=1, run_control=False).items():
        monkeypatch.setitem(cp.CFG, k, v)
    monkeypatch.setattr(cp, "stream_wider", lambda n: _fake_stream(n))
    monkeypatch.setattr(cp, "FaceDetector", _FakeDetector)

    def fake_finetune(repo, w, y, name, data_sig=None):
        p = tmp_path / "v2.pt"
        p.write_bytes(b"x")
        return p

    monkeypatch.setattr(cp, "finetune", fake_finetune)
    monkeypatch.setitem(cp.CFG, "ego4d_urls", [])
    monkeypatch.setitem(cp.CFG, "ego4d_auto", True)

    def boom(*a, **k):
        raise ModuleNotFoundError("No module named 'boto3' while opening https://s3.x/clip.mp4?X-Amz-Signature=TOPSECRET")

    monkeypatch.setattr(cp, "ego4d_presigned_urls", boom)
    zpath = cp.run_pipeline()  # must not raise
    out = zipfile.ZipFile(zpath)
    rep = json.loads(out.read("eval_report.json"))
    assert rep["provenance"]["ego4d_real"] is None
    assert "ModuleNotFoundError" in rep["provenance"]["ego4d_error"]
    assert all("TOPSECRET" not in out.read(n).decode() for n in out.namelist())
