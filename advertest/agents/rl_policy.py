import logging
import numpy as np
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import Dict, Any, List, Tuple

# Assuming we have access to the DB models
# from backend.app.db.models import AttackPoolItem

logger = logging.getLogger(__name__)

class RLAttackPolicy:
    """
    Contextual Multi-Armed Bandit for self-evolving attack policies.
    State (Context): The VLM insight (e.g., 'error_blur')
    Action (Arm): The specific Attack Chain & Severity chosen (e.g., 'MotionBlur_Sev3')
    Reward: The Attributed mAP improvement after retraining.
    """
    
    def __init__(self, db_session: Session = None):
        self.db = db_session
        self.epsilon = 0.2 # Epsilon-greedy exploration rate
        
        # Hardcoded fallback mappings if DB is cold (Cold Start)
        self.cold_start_policy = {
            "error_blur": [("KineticBlur", 2), ("DefocusBlur", 3)],
            "error_fisheye": [("FisheyeDistortion", 3)],
            "error_overexposure": [("SunFlare", 2), ("RandomGamma", 4)]
        }

    def _get_q_values_from_db(self, state: str) -> Dict[str, float]:
        """
        Calculates Q-values (expected reward) for each action under a specific state
        using historical data from the PostgreSQL DB.
        """
        if not self.db:
            return {}
            
        # Equivalent SQL: 
        # SELECT applied_chain, AVG(attributed_reward) as q_val 
        # FROM attack_pool WHERE insight_source = :state GROUP BY applied_chain
        
        # NOTE: Mocking the SQLAlchemy query execution for this architecture design
        # results = self.db.query(
        #     AttackPoolItem.applied_chain,
        #     func.avg(AttackPoolItem.attributed_reward).label('q_value')
        # ).filter(AttackPoolItem.insight_source == state)\
        #  .group_by(AttackPoolItem.applied_chain).all()
        
        # return {str(r.applied_chain): float(r.q_value) for r in results}
        return {} # Fallback for mock

    def select_action(self, insight_state: str) -> Tuple[str, int]:
        """
        Epsilon-greedy action selection.
        With probability epsilon, explore a random valid attack.
        With probability 1-epsilon, exploit the best attack from the DB history.
        """
        q_values = self._get_q_values_from_db(insight_state)
        
        # Exploration OR Cold Start (no history)
        if np.random.rand() < self.epsilon or not q_values:
            logger.info(f"[RL Policy] EXPLORING for state: {insight_state}")
            options = self.cold_start_policy.get(insight_state, [("GaussianNoise", 2)])
            idx = np.random.choice(len(options))
            return options[idx]
            
        # Exploitation (Choose the attack that yielded highest mAP improvement historically)
        best_action_str = max(q_values.items(), key=lambda x: x[1])[0]
        logger.info(f"[RL Policy] EXPLOITING (Q={q_values[best_action_str]:.2f}) for state: {insight_state}")
        
        # Parse the JSON string back to (AttackName, Severity)
        # Simplified parser for demonstration
        attack_name = best_action_str.split("_")[0]
        severity = int(best_action_str.split("_Sev")[-1]) if "_Sev" in best_action_str else 3
        return (attack_name, severity)

    def attribute_rewards(self, job_id: str, corrupted_map_improvement: float):
        """
        Called after Ray Cluster finishes a Training Job.
        Distributes the global reward (mAP improvement) to all the specific attack 
        actions that were in the pool for this job.
        """
        if not self.db:
            return
            
        # If the mAP improved, we give a positive reward. 
        # If it dropped, we give a negative reward (penalty) so the Bandit avoids it next time.
        reward = corrupted_map_improvement * 100.0 # Scale to percentage
        
        # SQL UPDATE attack_pool SET attributed_reward = :reward WHERE job_id = :job_id
        # self.db.query(AttackPoolItem).filter(AttackPoolItem.job_id == job_id).update(
        #     {"attributed_reward": reward}
        # )
        # self.db.commit()
        logger.info(f"[RL Policy] Attributed Reward {reward:+.2f} to Attack Pool generated in Job {job_id}")
