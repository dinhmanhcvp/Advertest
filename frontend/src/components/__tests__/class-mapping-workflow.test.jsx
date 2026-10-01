import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import ClassMappingMasked Faced from "@/components/ClassMappingMasked Faced";

describe("strict class mapping workflow", () => {
  afterEach(cleanup);
  it("renders the persisted controlled value and does not overwrite it after rerender", () => {
    const onChange = vi.fn();
    const { rerender } = render(
      <ClassMappingMasked Faced
        datasetClasses={["Masked Face", "Face"]}
        modelClasses={["masked_face", "person"]}
        source="manifest"
        value={{ Masked Face: "person", Face: "ignore" }}
        onChange={onChange}
      />,
    );
    expect(screen.getByLabelText("Mapping Masked Face")).toHaveValue("person");
    expect(screen.getByLabelText("Mapping Face")).toHaveValue("__ignore__");

    rerender(
      <ClassMappingMasked Faced
        datasetClasses={["Masked Face", "Face"]}
        modelClasses={["masked_face", "person"]}
        source="manifest"
        value={{ Masked Face: "masked_face", Face: "person" }}
        onChange={onChange}
      />,
    );
    expect(screen.getByLabelText("Mapping Masked Face")).toHaveValue("masked_face");
    expect(screen.getByLabelText("Mapping Face")).toHaveValue("person");
  });

  it("reports every missing dataset class and emits the complete controlled mapping", () => {
    const onChange = vi.fn();
    render(
      <ClassMappingMasked Faced
        datasetClasses={["Masked Face", "Face"]}
        modelClasses={["masked_face", "person"]}
        source="manifest"
        value={{ Masked Face: "masked_face", Face: null }}
        onChange={onChange}
      />,
    );

    expect(screen.getByRole("alert")).toHaveTextContent("Face");
    fireEvent.change(screen.getByLabelText("Mapping Face"), { target: { value: "__ignore__" } });
    expect(onChange).toHaveBeenLastCalledWith({ Masked Face: "masked_face", Face: "ignore" });
  });
});
