import random
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class ExperienceReplayBuffer:
    """
    Combats Catastrophic Forgetting during Continuous Training.
    Instead of merging new adversarial samples completely, it samples a mixed batch:
    - 20% Original Clean Baseline Data
    - 30% Historical Adversarial Data (Past iterations)
    - 50% New Adversarial Data (Current streaming iteration)
    """

    def __init__(self, max_history_size: int = 50000):
        self.clean_baseline: List[Dict[str, Any]] = []
        self.historical_adversarial: List[Dict[str, Any]] = []
        self.max_history_size = max_history_size

    def seed_baseline(self, clean_data: List[Dict[str, Any]]):
        """Loads the initial golden dataset."""
        self.clean_baseline = clean_data
        logger.info(f"Replay Buffer seeded with {len(clean_data)} clean samples.")

    def compile_training_batch(self, new_adversarial_data: List[Dict[str, Any]], target_size: int = 10000) -> List[Dict[str, Any]]:
        """
        Creates the mixed dataset payload for the Ray DDP cluster.
        """
        num_new = int(target_size * 0.50)
        num_history = int(target_size * 0.30)
        num_clean = target_size - num_new - num_history # Remaining 20%

        # Sample New
        sampled_new = random.sample(new_adversarial_data, min(num_new, len(new_adversarial_data)))
        
        # Sample History
        sampled_history = random.sample(self.historical_adversarial, min(num_history, len(self.historical_adversarial))) if self.historical_adversarial else []
        
        # Sample Clean
        sampled_clean = random.sample(self.clean_baseline, min(num_clean, len(self.clean_baseline))) if self.clean_baseline else []

        mixed_batch = sampled_new + sampled_history + sampled_clean
        random.shuffle(mixed_batch)

        # Update historical buffer with new data
        self.historical_adversarial.extend(new_adversarial_data)
        if len(self.historical_adversarial) > self.max_history_size:
            # Keep latest
            self.historical_adversarial = self.historical_adversarial[-self.max_history_size:]

        logger.info(f"Compiled CT Batch: {len(sampled_new)} New | {len(sampled_history)} Hist | {len(sampled_clean)} Clean")
        return mixed_batch
