"""Kafka Ingestion Stream for Continuous Training.

Listens to an enterprise Kafka topic ('raw_ego4d_streams'), decodes 
incoming video frame payloads in real-time, and routes them instantly 
into the AdverTest Inference & Filter pipeline.

Requirements:
    pip install confluent_kafka opencv-python numpy
"""

from __future__ import annotations

import json
import logging
import time
from typing import Callable, Optional

import cv2
import numpy as np

try:
    from confluent_kafka import Consumer, KafkaError, KafkaException
except ImportError:
    raise ImportError("confluent_kafka is required: pip install confluent_kafka")

logger = logging.getLogger(__name__)


class KafkaStreamIngestor:
    """Consumes real-time video frames from Kafka for Continuous Training."""

    def __init__(self, bootstrap_servers: str, group_id: str, topic: str = "raw_ego4d_streams"):
        self.topic = topic
        self.conf = {
            'bootstrap.servers': bootstrap_servers,
            'group.id': group_id,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': False  # Manual commit after successful processing
        }
        self.consumer = Consumer(self.conf)
        self.consumer.subscribe([self.topic])
        self.is_running = False
        logger.info("Kafka Consumer initialized on topic '%s'.", self.topic)

    def start_listening(self, frame_callback: Callable[[np.ndarray, dict], None]) -> None:
        """Start the ingestion loop.
        
        Parameters
        ----------
        frame_callback : Callable[[np.ndarray, dict], None]
            A function that takes the decoded BGR image and its metadata 
            (e.g., timestamp, camera_id) and pushes it into the Inference pipeline.
        """
        self.is_running = True
        logger.info("Starting Kafka polling loop...")

        try:
            while self.is_running:
                msg = self.consumer.poll(timeout=1.0)
                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        # End of partition event
                        logger.debug("Reached end of partition: %s [%d] at offset %d",
                                     msg.topic(), msg.partition(), msg.offset())
                    elif msg.error():
                        raise KafkaException(msg.error())
                else:
                    # Message is valid
                    try:
                        self._process_message(msg, frame_callback)
                        # Synchronous commit to ensure at-least-once delivery semantics
                        self.consumer.commit(asynchronous=False)
                    except Exception as e:
                        logger.error("Failed to process Kafka message: %s", e)
                        
        except KeyboardInterrupt:
            logger.info("Kafka polling interrupted by user.")
        finally:
            self.stop()

    def _process_message(self, msg, frame_callback: Callable[[np.ndarray, dict], None]) -> None:
        """Decode the Kafka payload and trigger the callback."""
        # Assume payload structure: JSON header + binary JPEG payload
        # For this implementation, we assume the message value is purely the JPEG bytes
        # and headers contain the metadata.
        
        headers = msg.headers() or []
        metadata = {k: v.decode('utf-8') for k, v in headers}
        
        payload = msg.value()
        if not payload:
            return

        # Decode JPEG bytes to BGR OpenCV format
        np_arr = np.frombuffer(payload, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is not None:
            # Route directly to the Inference Engine callback
            frame_callback(img_bgr, metadata)
        else:
            logger.warning("Failed to decode image payload from Kafka.")

    def stop(self) -> None:
        """Gracefully shut down the consumer."""
        self.is_running = False
        self.consumer.close()
        logger.info("Kafka Consumer shut down safely.")
