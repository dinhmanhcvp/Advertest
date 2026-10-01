import { fireEvent, render, screen } from "@testing-library/react";
import React from "react";
import { describe, expect, it, vi } from "vitest";

import ClassMappingCard from "@/components/ClassMappingCard";

function ControlledMapping(props) {
  const [value, setValue] = React.useState({});
  return <ClassMappingCard {...props} value={value} onChange={setValue} />;
}

describe("ClassMappingCard", () => {
  it("keeps an explicit mapping when parent arrays are recreated", () => {
    const props = {
      datasetClasses: ["Car", "Face"],
      modelClasses: ["car", "person"],
      source: "manifest",
    };
    const { rerender } = render(<ControlledMapping {...props} />);

    fireEvent.change(screen.getByLabelText("Mapping Face"), {
      target: { value: "person" },
    });
    rerender(<ControlledMapping {...props} datasetClasses={["Car", "Face"]} modelClasses={["car", "person"]} />);
    expect(screen.getByLabelText("Mapping Face")).toHaveValue("person");
  });

  it("records intentionally ignored labels", () => {
    render(
      <ControlledMapping datasetClasses={["Blurred Face"]} modelClasses={["car"]} source="manifest" />,
    );

    fireEvent.change(screen.getByLabelText("Mapping Blurred Face"), {
      target: { value: "__ignore__" },
    });
    expect(screen.getByLabelText("Mapping Blurred Face")).toHaveValue("__ignore__");
  });
});
