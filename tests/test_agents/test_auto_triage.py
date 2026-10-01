"""Unit tests for the AutoTriageAgent."""

from src.agents.auto_triage import AutoTriageAgent
from src.core.egocentric_types import AttackPool, AttackPoolEntry


def test_triage_pool_severity_filtering():
    """Verify that lower severity has a higher approval rate than higher severity."""
    agent = AutoTriageAgent(rng_seed=42)
    
    # Create a pool with 100 entries of Severity 1, and 100 entries of Severity 5
    entries = []
    for i in range(100):
        entries.append(AttackPoolEntry(
            entry_id=f"sev1_{i}",
            source_sample_id="img",
            attack_name="fisheye",
            severity=1,
            error_category="pii_fisheye"
        ))
        entries.append(AttackPoolEntry(
            entry_id=f"sev5_{i}",
            source_sample_id="img",
            attack_name="fisheye",
            severity=5,
            error_category="pii_fisheye"
        ))
        
    pool = AttackPool(
        pool_id="test_pool",
        error_category="pii_fisheye",
        attack_name="fisheye",
        severity_range=(1, 5),
        entries=tuple(entries)
    )
    
    approved_pool, rejected_entries = agent.triage_pool(pool)
    
    # Analyze the results
    sev1_approved = sum(1 for e in approved_pool.entries if e.severity == 1)
    sev5_approved = sum(1 for e in approved_pool.entries if e.severity == 5)
    
    # We expect almost all (if not all) Severity 1 to be approved, 
    # and almost all Severity 5 to be rejected.
    assert sev1_approved > 80, f"Expected high approval for Sev 1, got {sev1_approved}"
    assert sev5_approved < 20, f"Expected low approval for Sev 5, got {sev5_approved}"
    
    # Ensure no entries were lost
    assert len(approved_pool.entries) + len(rejected_entries) == 200
