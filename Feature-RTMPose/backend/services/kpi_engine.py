"""KPI Engine — transforms predictions into business metrics."""

import time
from services.database import Database


class KPIEngine:
    def __init__(self, db: Database):
        self.db = db
        self.current_activity: str | None = None
        self.current_confidence: float = 0.0
        self.activity_start: float = time.time()
        self.predictions_count = 0
        self.total_latency = 0.0

    async def process_prediction(self, prediction: dict) -> dict | None:
        """Process a prediction, generate event if activity changed."""
        self.predictions_count += 1
        self.total_latency += prediction.get("latency_ms", 0)

        activity = prediction.get("activity", "unknown")
        confidence = prediction.get("confidence", 0)

        # No pose, or no trained model yet: nothing happened worth logging
        if not prediction.get("pose_detected") or activity == "untrained":
            return None

        if activity != self.current_activity:
            event = None
            now = time.time()

            # Close the previous activity with the confidence it was last seen at,
            # not with the confidence of the new prediction
            if self.current_activity is not None:
                duration = now - self.activity_start
                event = await self.db.insert_event(
                    activity=self.current_activity,
                    confidence=self.current_confidence,
                    duration=duration,
                )

            self.current_activity = activity
            self.current_confidence = confidence
            self.activity_start = now
            return event

        self.current_confidence = confidence

        return None

    async def compute_metrics(self) -> dict:
        """Compute current KPI metrics from stored events."""
        now = time.time()
        # Last 30 minutes of events
        since = now - 1800
        events = await self.db.get_events_since(since)

        if not events:
            return {
                "active_time_pct": 0,
                "idle_time_pct": 0,
                "operations_per_hour": 0,
                "avg_operation_duration": 0,
                "total_events": 0,
                "avg_confidence": 0,
                "avg_latency_ms": 0,
                "activity_breakdown": {},
                "current_activity": self.current_activity,
                "uptime_seconds": round(now - self.activity_start, 1),
            }

        total_duration = sum(e.get("duration", 0) for e in events)
        idle_duration = sum(e.get("duration", 0) for e in events if e["activity"] == "idle")
        active_duration = total_duration - idle_duration

        # Activity breakdown
        breakdown = {}
        for e in events:
            act = e["activity"]
            dur = e.get("duration", 0)
            if act not in breakdown:
                breakdown[act] = {"count": 0, "total_duration": 0}
            breakdown[act]["count"] += 1
            breakdown[act]["total_duration"] = round(breakdown[act]["total_duration"] + dur, 2)

        total_events = len(events)
        confidences = [e["confidence"] for e in events if e.get("confidence")]
        avg_confidence = round(sum(confidences) / len(confidences), 3) if confidences else 0

        time_window_seconds = min(now - events[0]["timestamp"], 1800) if events else 1800
        time_window_hours = max(time_window_seconds / 3600, 0.01)

        avg_latency = round(self.total_latency / self.predictions_count, 1) if self.predictions_count else 0

        non_idle_events = [e for e in events if e["activity"] != "idle"]

        return {
            "active_time_pct": round((active_duration / max(total_duration, 1)) * 100, 1),
            "idle_time_pct": round((idle_duration / max(total_duration, 1)) * 100, 1),
            "operations_per_hour": round(len(non_idle_events) / time_window_hours, 1),
            "avg_operation_duration": round(
                sum(e.get("duration", 0) for e in non_idle_events) / max(len(non_idle_events), 1), 1
            ),
            "total_events": total_events,
            "avg_confidence": avg_confidence,
            "avg_latency_ms": avg_latency,
            "activity_breakdown": breakdown,
            "current_activity": self.current_activity,
            "uptime_seconds": round(now - self.activity_start, 1),
        }

    async def get_timeline(self, minutes: int = 30) -> dict:
        """Return activity timeline data for charts."""
        since = time.time() - (minutes * 60)
        events = await self.db.get_events_since(since)

        timeline = []
        for e in events:
            timeline.append({
                "timestamp": e["timestamp"],
                "activity": e["activity"],
                "confidence": e["confidence"],
                "duration": e.get("duration", 0),
            })

        return {"timeline": timeline, "minutes": minutes}
