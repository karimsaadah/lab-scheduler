"""Observability layer: structured event logging for the lab scheduler."""

from .event_log import EventLog, EventType

__all__ = ["EventLog", "EventType"]
