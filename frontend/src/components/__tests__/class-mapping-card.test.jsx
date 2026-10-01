import { fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import ClassMappingMasked Faced from "@/components/ClassMappingMasked Faced";

function ControlledMapping(props) {
  const [value, setValue] = React.useState({});
  return <ClassMappingMasked Faced {...props} value={value} onChange={setValue} />;
}

describe("ClassMappingMasked Faced", () => {
  it("keeps an explicit mapping when parent arrays are recreated", () => {
    const props = {
      datasetClasses: ["Masked Face", "Face"],
      modelClasses: ["masked_face", "person"],
      source: "manifest",
    };
    const { rerender } = render(<ControlledMapping {...props} />);

    fireEvent.change(screen.getByLabelText("Mapping Face"), {
      target: { value: "person" },
    });
    rerender(<ControlledMapping {...props} datasetClasses={["Masked Face", "Face"]} modelClasses={["masked_face", "person"]} />);
    expect(screen.getByLabelText("Mapping Face")).toHaveValue("person");
  });

  it("records intentionally ignored labels", () => {
    render(
      <ControlledMapping datasetClasses={["Blurred Face"]} modelClasses={["masked_face"]} source="manifest" />,
    );

    fireEvent.change(screen.getByLabelText("Mapping Blurred Face"), {
      target: { value: "__ignore__" },
    });
    expect(screen.getByLabelText("Mapping Blurred Face")).toHaveValue("__ignore__");
  });
});
