"""AdverTest Master Orchestrator (CLI).

This script serves as the central command line interface for the entire
AdverTest pipeline. It ties together the inference engine, Label Studio
connector, physics-based attack engine, dataset compiler, and evaluator.

Usage:
    python main.py run-inference --weights <path> --data <path> --project-id <id>
    python main.py generate-attacks --project-id <id> --input <path> --output <path>
    python main.py compile-dataset --project-id <id> --clean <path> --adv <path> --output <path>
    python main.py evaluate --old-weights <path> --new-weights <path> --clean-val <path> --corr-val <path>
"""

import argparse
import logging
import sys
from pathlib import Path

# Configure global logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-15s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("advertest.main")


def cmd_run_inference(args: argparse.Namespace) -> None:
    """Run YOLOv7 inference and upload hard negatives to Label Studio."""
    from advertest.core.inference_engine import Yolov7Evaluator
    from advertest.label_studio.ls_connector import LSConnector
    
    logger.info("Starting Phase 1: Inference & Hard Negative Extraction")
    
    evaluator = Yolov7Evaluator(weights=args.weights)
    hard_negs = evaluator.extract_hard_negatives(
        image_dir=args.data,
        output_dir="data/temp/hard_negatives",
        iou_threshold=args.iou_threshold,
        conf_threshold=args.conf_threshold
    )
    
    if not hard_negs:
        logger.info("No hard negatives found. Pipeline stops here for this batch.")
        return
        
    logger.info("Connecting to Label Studio to upload %d Hard Negatives...", len(hard_negs))
    try:
        ls = LSConnector()
        ls.upload_hard_negatives(project_id=args.project_id, hard_negatives=hard_negs)
        logger.info("Upload complete. Awaiting Human-In-The-Loop Error Analysis.")
    except Exception as e:
        logger.error("Label Studio upload failed: %s", e)
        sys.exit(1)


def cmd_generate_attacks(args: argparse.Namespace) -> None:
    """Fetch Error Analysis insights, run AttackEngine, and push Attack Pools."""
    from advertest.label_studio.ls_connector import LSConnector
    from advertest.attacks.engine import AttackEngine, BBoxIntegrityChecker
    from advertest.core.inference_engine import YOLOBox
    import cv2
    import shutil
    
    logger.info("Starting Phase 2: Generating Physics-based Adversarial Attacks")
    
    # 1. Fetch Insights
    try:
        ls = LSConnector()
        distribution = ls.fetch_insights(project_id=args.project_id)
        logger.info("Fetched Error Distribution: %s", distribution)
    except Exception as e:
        logger.error("Label Studio fetch failed: %s", e)
        sys.exit(1)
        
    # 2. Setup Engine
    engine = AttackEngine()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_exts = {".jpg", ".jpeg", ".png", ".webp"}
    images = [p for p in input_dir.iterdir() if p.suffix.lower() in image_exts]
    
    logger.info("Generating adversarial examples for %d clean images...", len(images))
    success_count = 0
    
    for img_path in images:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
            
        label_path = img_path.with_suffix(".txt")
        boxes = []
        if label_path.exists():
            for line in label_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append(YOLOBox(
                        class_id=int(parts[0]),
                        x_center=float(parts[1]),
                        y_center=float(parts[2]),
                        width=float(parts[3]),
                        height=float(parts[4])
                    ))
        
        # Simple probability routing (could be extracted to InsightRouter)
        import random
        # Choose attack based on distribution weights, default to Fisheye if clean
        tags = list(distribution.keys())
        probs = list(distribution.values())
        if sum(probs) == 0:
            chosen_tag = "error_fisheye"
        else:
            chosen_tag = random.choices(tags, weights=probs, k=1)[0]
            
        if chosen_tag == "clean":
            continue
            
        # Map tag to attack strategy
        attack_map = {
            "error_fisheye": "fisheye",
            "error_blur": "blur",
            "error_overexposure": "exposure"
        }
        strategy = attack_map.get(chosen_tag, "fisheye")
        severity = random.randint(3, 5) # Hard mode
        
        # Execute Attack
        result = engine.apply(img_bgr, boxes, strategy, severity)
        
        if result.discarded:
            logger.warning("Image %s: Transformation rejected by BBoxIntegrityChecker.", img_path.name)
            continue
            
        # Save output
        out_img_path = output_dir / f"{img_path.stem}_{chosen_tag}_s{severity}.jpg"
        out_lbl_path = output_dir / f"{img_path.stem}_{chosen_tag}_s{severity}.txt"
        
        cv2.imwrite(str(out_img_path), result.image)
        with open(out_lbl_path, "w") as f:
            for b in result.boxes:
                f.write(b.to_label_str() + "\n")
                
        success_count += 1
        
    logger.info("Successfully generated %d robust adversarial samples.", success_count)
    logger.info("Next Step: Import %s into Label Studio for HITL Review.", output_dir)


def cmd_compile_dataset(args: argparse.Namespace) -> None:
    """Merge approved adversarial pools with the clean baseline."""
    from advertest.core.dataset_compiler import DatasetCompiler
    from advertest.label_studio.ls_connector import LSConnector
    
    logger.info("Starting Phase 3: Dataset Compilation & Versioning")
    
    try:
        ls = LSConnector()
        compiler = DatasetCompiler(ls_connector=ls)
        
        stats = compiler.merge_approved_pools(
            project_id=args.project_id,
            clean_dataset_dir=args.clean,
            adversarial_source_dir=args.adv,
            output_dir=args.output
        )
        
        compiler.generate_data_card(output_dir=args.output, compilation_stats=stats)
        
        logger.info("Dataset compilation complete. Total samples: %d", stats["total_count"])
    except Exception as e:
        logger.error("Dataset compilation failed: %s", e)
        sys.exit(1)


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Run the Blind Evaluation Gate."""
    from advertest.core.evaluator import RobustnessEvaluator
    
    logger.info("Starting Phase 4: Blind Evaluation Gate")
    
    try:
        evaluator = RobustnessEvaluator(
            old_weights_path=args.old_weights,
            new_weights_path=args.new_weights
        )
        
        results = evaluator.calculate_map_drop(
            clean_val_dir=args.clean_val,
            corrupted_val_dir=args.corr_val
        )
        
        if results["status"] == "FAIL":
            logger.error("Deployment BLOCKED. Metrics did not pass the robustness threshold.")
            sys.exit(1)
        else:
            logger.info("Deployment APPROVED. Model meets all robustness criteria.")
            sys.exit(0)
            
    except Exception as e:
        logger.error("Evaluation Gate failed due to internal error: %s", e)
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AdverTest Pipeline Master Orchestrator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Pipeline phase to execute")

    # 1. Run Inference
    p_inf = subparsers.add_parser("run-inference", help="Run inference and upload hard negatives")
    p_inf.add_argument("--weights", type=str, required=True, help="Path to YOLOv7 .pt weights")
    p_inf.add_argument("--data", type=str, required=True, help="Directory containing target images")
    p_inf.add_argument("--project-id", type=int, required=True, help="Label Studio Error Analysis Project ID")
    p_inf.add_argument("--iou-threshold", type=float, default=0.5, help="IoU cutoff for hard negatives")
    p_inf.add_argument("--conf-threshold", type=float, default=0.4, help="Conf cutoff for hard negatives")
    p_inf.set_defaults(func=cmd_run_inference)

    # 2. Generate Attacks
    p_att = subparsers.add_parser("generate-attacks", help="Generate physics attacks based on LS insights")
    p_att.add_argument("--project-id", type=int, required=True, help="Label Studio Error Analysis Project ID")
    p_att.add_argument("--input", type=str, required=True, help="Path to clean training images to attack")
    p_att.add_argument("--output", type=str, default="data/pools/generated", help="Output directory for Attack Pools")
    p_att.set_defaults(func=cmd_generate_attacks)

    # 3. Compile Dataset
    p_com = subparsers.add_parser("compile-dataset", help="Merge approved attacks with clean baseline")
    p_com.add_argument("--project-id", type=int, required=True, help="Label Studio Review Project ID")
    p_com.add_argument("--clean", type=str, required=True, help="Path to clean WIDER FACE dataset root")
    p_com.add_argument("--adv", type=str, required=True, help="Path to locally generated adversarial images")
    p_com.add_argument("--output", type=str, default="data/dataset_v2", help="Output directory for new dataset")
    p_com.set_defaults(func=cmd_compile_dataset)

    # 4. Evaluate (Blind Gate)
    p_eval = subparsers.add_parser("evaluate", help="Run Blind Evaluation Gate")
    p_eval.add_argument("--old-weights", type=str, required=True, help="Original model weights (baseline)")
    p_eval.add_argument("--new-weights", type=str, required=True, help="Retrained model weights (candidate)")
    p_eval.add_argument("--clean-val", type=str, required=True, help="Clean Ego4D validation set directory")
    p_eval.add_argument("--corr-val", type=str, required=True, help="Corrupted Ego4D validation set directory")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        logger.info("Pipeline terminated by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
