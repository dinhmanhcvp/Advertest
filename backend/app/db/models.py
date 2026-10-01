import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class RetrainingJob(Base):
    """Stores the overarching Retraining cycles on Ray Cluster."""
    __tablename__ = "retraining_jobs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version = Column(String, index=True)
    status = Column(String, default="PENDING") # PENDING, RUNNING, SUCCESS, FAILED
    
    # Evaluation Metrics post-training
    base_map_before = Column(Float, nullable=True)
    base_map_after = Column(Float, nullable=True)
    corrupted_map_before = Column(Float, nullable=True)
    corrupted_map_after = Column(Float, nullable=True)
    
    # RL Reward (Calculated as Corrupted_mAP improvement - Base_mAP drop penalty)
    rl_reward = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)


class AttackPoolItem(Base):
    """
    Stores individual generated adversarial frames.
    Links to S3 URIs and stores VLM diagnostics and Attack params in JSONB.
    """
    __tablename__ = "attack_pool"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("retraining_jobs.id"), nullable=True)
    
    # S3 or Cloudflare R2 URI (NO BINARY DATA IN DB!)
    image_uri = Column(String, nullable=False, unique=True)
    
    # Insight from VLM / Fast Heuristics (State for RL)
    insight_source = Column(String, index=True) # e.g., 'error_blur', 'error_fisheye'
    
    # The Action chosen by RL and applied by AttackEngine
    applied_chain = Column(JSONB, nullable=False) # e.g., [{"type": "MotionBlur", "severity": 3}]
    severity_level = Column(Float, nullable=False)
    
    # Did this specific sample help? (Attributed reward after retraining job completes)
    attributed_reward = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
