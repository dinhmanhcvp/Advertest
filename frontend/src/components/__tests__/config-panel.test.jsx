import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import ConfigPanel from "@/components/ConfigPanel.jsx";

afterEach(() => {
  cleanup();
});

describe("ConfigPanel", () => {
  it("separates the base model family from its base checkpoint", () => {
    const actions = {
      setSelectedDataset: vi.fn(),
      setMode: vi.fn(),
      setModelFamily: vi.fn(),
      setSelectedModelVersion: vi.fn(),
      toggleAttack: vi.fn(),
      updateAttackSeverity: vi.fn(),
      handleRun: vi.fn(),
    };
    render(
      <ConfigPanel
        datasets={[{ name: "synthetic_shapes", anonymized: true }]}
        attacks={[{ name: "gaussian_noise", group: "A", cost_class: "LIGHT" }]}
        modes={[
          { id: "detection2d", title: "2D Object Detection", runnable: true },
          {
            id: "segmentation",
            title: "Instance Segmentation",
            runnable: false,
            status: "ready",
            blocked_reason: "WAITING_FOR_ARTIFACTS",
          },
        ]}
        modelFamilies={[{ id: "yolov7-face", display_name: "YOLOv7-Face", runnable: true }]}
        baseCheckpoints={[
          { id: "yolov7-face-base", model_name: "yolov7-face", model_family_id: "yolov7-face", task: "detection2d", runnable: true },
        ]}
        recipePresets={[{ preset_id: "weather_robustness", name: "Weather Robustness" }]}
        mode="detection2d"
        selectedModelFamily="yolov7-face"
        selectedModelVersion="yolov7-face-base"
        selectedDataset="synthetic_shapes"
        selectedAttacks={["gaussian_noise"]}
        recipe={{ steps: [{ position: 0, attack_name: "gaussian_noise", severity: 3 }] }}
        isRunning={false}
        actions={actions}
      />,
    );

    fireEvent.change(screen.getByLabelText("Base Checkpoint"), { target: { value: "yolov7-face-base" } });
    expect(actions.setSelectedModelVersion).toHaveBeenCalledWith("yolov7-face-base");
    expect(screen.getByRole("option", { name: "Instance Segmentation" })).not.toBeDisabled();
  });

  it("does not surface a defence checkpoint in the attack selector", () => {
    render(
      <ConfigPanel
        datasets={[]}
        attacks={[]}
        modes={[]}
        modelFamilies={[{ id: "yolov7-face", display_name: "YOLOv7-Face", runnable: true }]}
        baseCheckpoints={[
          {
            id: "yolov7-face-base",
            model_name: "yolov7-face",
            task: "detection2d",
            model_family_id: "yolov7-face",
            runnable: true,
          },
        ]}
        mode="detection2d"
        selectedModelFamily="yolov7-face"
        selectedModelVersion="yolov7-face-base"
        selectedDataset=""
        selectedAttacks={[]}
        recipe={{ steps: [] }}
        isRunning={false}
        actions={{}}
      />,
    );

    expect(screen.getByRole("option", { name: /yolov7-face.*base/i })).toBeInTheDocument();
    expect(screen.queryByRole("option", { name: /widerface.*b0/i })).toBeNull();
  });

  it("marks attacks incompatible with the selected model as unavailable", () => {
    render(
      <ConfigPanel
        datasets={[{ name: "synthetic_shapes", anonymized: true }]}
        attacks={[
          {
            name: "dag",
            group: "D",
            cost_class: "HEAVY",
            available: false,
            reason: "missing_model_capability:dense_proposals",
          },
        ]}
        modes={[{ id: "detection2d", title: "2D Object Detection", runnable: true }]}
        modelFamilies={[{ id: "yolov7-face", display_name: "YOLOv7-Face", runnable: true }]}
        baseCheckpoints={[
          { id: "yolov7-face-base", model_name: "yolov7-face", model_family_id: "yolov7-face", task: "detection2d", runnable: true },
        ]}
        mode="detection2d"
        selectedModelFamily="yolov7-face"
        selectedModelVersion="yolov7-face-base"
        selectedDataset="synthetic_shapes"
        selectedAttacks={[]}
        recipe={{ steps: [] }}
        isRunning={false}
        actions={{}}
      />,
    );

    const dag = screen.getByRole("button", { name: /dag/i });
    expect(dag).toBeDisabled();
    expect(dag).toHaveAttribute("title", "missing_model_capability:dense_proposals");
  });

  it("separates attack masked_faceds by threat-model class without hiding incompatible methods", () => {
    render(
      <ConfigPanel
        datasets={[]}
        attacks={[
          { name: "fgsm", threat_model: "white_box", attack_type: "gradient", available: true },
          {
            name: "square",
            threat_model: "black_box",
            attack_type: "query",
            available: false,
            reason: "missing capability",
          },
          { name: "fog", threat_model: "model_agnostic", attack_type: "corruption", available: true },
        ]}
        modes={[]}
        modelFamilies={[]}
        baseCheckpoints={[]}
        mode="detection2d"
        selectedModelFamily=""
        selectedModelVersion=""
        selectedDataset=""
        selectedAttacks={[]}
        recipe={{ steps: [] }}
        isRunning={false}
        actions={{}}
      />,
    );

    expect(screen.getByText("White-box")).toBeVisible();
    expect(screen.getByText("Black-box / query")).toBeVisible();
    expect(screen.getByText("Real-world / corruption / weather / sensor")).toBeVisible();
    expect(screen.getByRole("button", { name: /square/i })).toBeDisabled();
  });

  it("supports manual recipe attack selection and severity", () => {
    const actions = {
      setSelectedDataset: vi.fn(),
      setMode: vi.fn(),
      setModelFamily: vi.fn(),
      setSelectedModelVersion: vi.fn(),
      toggleAttack: vi.fn(),
      updateAttackSeverity: vi.fn(),
      handleRun: vi.fn(),
    };

    render(
      <ConfigPanel
        datasets={[{ name: "synthetic_shapes", anonymized: true }]}
        attacks={[{ name: "gaussian_noise", group: "A", cost_class: "LIGHT" }]}
        modes={[{ id: "detection2d", title: "YOLO", runnable: true }]}
        modelFamilies={[{ id: "yolov7-face", display_name: "YOLOv7-Face", runnable: true }]}
        baseCheckpoints={[
          { id: "yolov7-face-base", model_name: "yolov7-face", model_family_id: "yolov7-face", task: "detection2d", runnable: true },
        ]}
        mode="detection2d"
        selectedModelFamily="yolov7-face"
        selectedModelVersion="yolov7-face-base"
        selectedDataset="synthetic_shapes"
        selectedAttacks={["gaussian_noise"]}
        recipe={{ steps: [{ position: 0, attack_name: "gaussian_noise", severity: 3 }] }}
        isRunning={false}
        actions={actions}
      />,
    );

    // Click Real-world tab and toggle attack
    fireEvent.click(screen.getByText("Real-world / corruption / weather / sensor"));
    fireEvent.click(screen.getByRole("button", { name: /gaussian noise/i }));
    expect(actions.toggleAttack).toHaveBeenCalledWith("gaussian_noise");
  });

  it("toggles the advanced settings drawer", () => {
    render(
      <ConfigPanel
        datasets={[{ name: "synthetic_shapes", anonymized: true }]}
        attacks={[]}
        modes={[{ id: "detection2d", title: "YOLO", runnable: true }]}
        modelFamilies={[{ id: "yolov7-face", display_name: "YOLOv7-Face", runnable: true }]}
        baseCheckpoints={[
          { id: "yolov7-face-base", model_name: "yolov7-face", model_family_id: "yolov7-face", task: "detection2d", runnable: true },
        ]}
        mode="detection2d"
        selectedModelFamily="yolov7-face"
        selectedModelVersion="yolov7-face-base"
        selectedDataset="synthetic_shapes"
        selectedAttacks={[]}
        recipe={{ steps: [] }}
        isRunning={false}
        actions={{}}
      />,
    );

    expect(screen.queryByLabelText("Seed")).toBeNull();
    fireEvent.click(screen.getByText(/Show Advanced Settings Drawer/));
    expect(screen.getByLabelText("Seed")).toHaveValue(42);
    expect(screen.getByLabelText("Sample Limit")).toHaveValue(8);
  });

  it("changes only the severity of the selected recipe step", () => {
    const updateAttackSeverity = vi.fn();
    render(
      <ConfigPanel
        datasets={[]}
        attacks={[]}
        modes={[]}
        modelVersions={[]}
        mode="detection2d"
        selectedDataset=""
        selectedModelVersion=""
        selectedAttacks={["brightness", "fgsm"]}
        recipe={{
          steps: [
            { position: 0, attack_name: "brightness", severity: 1 },
            { position: 1, attack_name: "fgsm", severity: 4 },
          ],
        }}
        isRunning={false}
        actions={{ updateAttackSeverity }}
      />,
    );

    fireEvent.change(screen.getByLabelText("brightness severity"), { target: { value: "2" } });

    expect(updateAttackSeverity).toHaveBeenCalledWith(0, 2);
    expect(updateAttackSeverity).not.toHaveBeenCalledWith(1, expect.anything());
  });
});
