import uuid
from datetime import datetime, timezone
from typing import Optional, Any
from sqlalchemy import String, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import declarative_base, Mapped, mapped_column

Base = declarative_base()

class RetrainingJob(Base):
    """Stores the overarching Retraining cycles on Ray Cluster."""
    __tablename__ = "retraining_jobs"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version: Mapped[str] = mapped_column(String, index=True)
    status: Mapped[str] = mapped_column(String, default="PENDING") # PENDING, RUNNING, SUCCESS, FAILED
    
    # Evaluation Metrics post-training
    base_map_before: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    base_map_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    corrupted_map_before: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    corrupted_map_after: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    # RL Reward (Calculated as Corrupted_mAP improvement - Base_mAP drop penalty)
    rl_reward: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class AttackPoolItem(Base):
    """
    Stores individual generated adversarial frames.
    Links to S3 URIs and stores VLM diagnostics and Attack params in JSONB.
    """
    __tablename__ = "attack_pool"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id: Mapped[Optional[uuid.UUID]] = mapped_column(UUID(as_uuid=True), ForeignKey("retraining_jobs.id"), nullable=True)
    
    # S3 or Cloudflare R2 URI (NO BINARY DATA IN DB!)
    image_uri: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    
    # Insight from VLM / Fast Heuristics (State for RL)
    insight_source: Mapped[str] = mapped_column(String, index=True) # e.g., 'error_blur', 'error_fisheye'
    
    # The Action chosen by RL and applied by AttackEngine
    applied_chain: Mapped[Any] = mapped_column(JSONB, nullable=False) # e.g., [{"type": "MotionBlur", "severity": 3}]
    severity_level: Mapped[float] = mapped_column(Float, nullable=False)
    
    # Did this specific sample help? (Attributed reward after retraining job completes)
    attributed_reward: Mapped[float] = mapped_column(Float, default=0.0)
    
    created_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(timezone.utc))
