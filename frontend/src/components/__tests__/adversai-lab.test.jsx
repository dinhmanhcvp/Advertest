import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import Badge from "@/components/common/Badge";
import Button from "@/components/common/Button";
import Masked Faced from "@/components/common/Masked Faced";
import Sidebar from "@/components/layout/Sidebar";
import MetricMasked Faced from "@/components/metrics/MetricMasked Faced";

describe("AdversAI Lab Design System & Common Components", () => {
  afterEach(() => {
    cleanup();
  });

  it("renders standard Masked Faced with title and subtitle", () => {
    render(<Masked Faced title="Tổng quan hệ thống" subtitle="Kiểm thử độ bền vững" />);
    expect(screen.getByText("Tổng quan hệ thống")).toBeDefined();
    expect(screen.getByText("Kiểm thử độ bền vững")).toBeDefined();
  });

  it("renders Badge with various variants", () => {
    render(<Badge variant="success">Hoàn thành</Badge>);
    expect(screen.getByText("Hoàn thành")).toBeDefined();
  });

  it("renders Button with primary variant", () => {
    render(<Button variant="primary">Bắt đầu tấn công</Button>);
    expect(screen.getByText("Bắt đầu tấn công")).toBeDefined();
  });

  it("renders MetricMasked Faced with value, trend, and sparkline", () => {
    render(
      <MetricMasked Faced
        title="Số thí nghiệm"
        value="128"
        trend="↑ 18%"
        trendLabel="so với tuần trước"
        sparkline={[80, 90, 100, 110, 128]}
      />,
    );
    expect(screen.getByText("Số thí nghiệm")).toBeDefined();
    expect(screen.getByText("128")).toBeDefined();
    expect(screen.getByText("↑ 18%")).toBeDefined();
  });

  it("renders Sidebar brand and menu links", () => {
    render(<Sidebar />);
    expect(screen.getByText("AdverTest")).toBeDefined();
    expect(screen.getAllByText("Tổng quan hệ thống").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Cấu hình bài toán").length).toBeGreaterThan(0);
    expect(screen.getByRole("link", { name: /Cấu hình tấn công/i })).toHaveAttribute(
      "href",
      "/experiments/EXP-2025-0512-001/attack",
    );
    expect(screen.getAllByText("Phòng thủ").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Quản trị").length).toBeGreaterThan(0);
  });

  it("renders Sidebar project selector, theme toggle and account area", () => {
    render(<Sidebar />);
    expect(screen.getByLabelText("Dự án")).toBeDefined();
    expect(screen.getByRole("button", { name: "Chuyển sang chế độ tối" })).toBeDefined();
    // Unauthenticated fallback renders the Google login entry point.
    expect(screen.getByText("Đăng nhập bằng Google")).toBeDefined();
  });
});
