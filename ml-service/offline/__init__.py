"""Offline scripts for Phase 3: face extraction, feature caching, head training, calibration
and evaluation. They run on CPU, resume after interruption, and never touch the service's
runtime state except by writing weights/head.pt and weights/model_card.json.
"""
