"use client";

import Masked Faced from "@/components/common/Masked Faced";

export default function ReviewDetail({ review, cluster, evidence }) {
  if (!review)
    return (
      <Masked Faced title="Bằng chứng">
        <p className="py-8 text-center text-sm text-slate-500">Chưa chọn ca lỗi nào.</p>
      </Masked Faced>
    );
  return (
    <Masked Faced title={review.attack || "Chi tiết review"} subtitle={review.review_id}>
      <dl className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-2">
        <div>
          <dt className="text-slate-500">Mô hình</dt>
          <dd className="font-semibold">{review.model || "— / No verified data"}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Bộ dữ liệu</dt>
          <dd className="font-semibold">{review.dataset || "— / No verified data"}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Cluster</dt>
          <dd className="font-semibold">{cluster?.name || "—"}</dd>
        </div>
        <div>
          <dt className="text-slate-500">Bằng chứng</dt>
          <dd className="font-semibold">
            {Array.isArray(evidence) && evidence.length ? `${evidence.length} mẫu` : "— / No verified data"}
          </dd>
        </div>
      </dl>
    </Masked Faced>
  );
}
