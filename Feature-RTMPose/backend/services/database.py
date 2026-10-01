"""SQLite database service for events and training data."""

import aiosqlite
import json
import os
import time
import uuid


DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "warehouse.db")


class Database:
    def __init__(self):
        self.db: aiosqlite.Connection | None = None

    async def init(self):
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        self.db = await aiosqlite.connect(DB_PATH)
        self.db.row_factory = aiosqlite.Row
        await self.db.executescript("""
            CREATE TABLE IF NOT EXISTS events (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                activity TEXT NOT NULL,
                confidence REAL NOT NULL,
                duration REAL DEFAULT 0,
                worker_id TEXT DEFAULT 'worker-1',
                zone TEXT DEFAULT 'zone-A',
                metadata TEXT DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_events_ts ON events(timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_events_activity ON events(activity);

            CREATE TABLE IF NOT EXISTS annotation_projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                video_filename TEXT,
                video_path TEXT,
                frame_width INTEGER DEFAULT 0,
                frame_height INTEGER DEFAULT 0,
                total_frames INTEGER DEFAULT 0,
                fps REAL DEFAULT 0,
                created_at REAL NOT NULL
            );

            CREATE TABLE IF NOT EXISTS annotation_frames (
                id TEXT PRIMARY KEY,
                project_id TEXT NOT NULL,
                frame_index INTEGER NOT NULL,
                keypoints TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                FOREIGN KEY (project_id) REFERENCES annotation_projects(id),
                UNIQUE(project_id, frame_index)
            );

            CREATE INDEX IF NOT EXISTS idx_annot_frames_project
                ON annotation_frames(project_id, frame_index);
        """)
        await self.db.commit()

    async def close(self):
        if self.db:
            await self.db.close()

    async def insert_event(self, activity: str, confidence: float,
                           duration: float = 0, worker_id: str = "worker-1",
                           zone: str = "zone-A", metadata: dict | None = None) -> dict:
        event_id = str(uuid.uuid4())[:8]
        ts = time.time()
        meta_json = json.dumps(metadata or {})
        await self.db.execute(
            "INSERT INTO events (id, timestamp, activity, confidence, duration, worker_id, zone, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (event_id, ts, activity, confidence, duration, worker_id, zone, meta_json),
        )
        await self.db.commit()
        return {
            "id": event_id,
            "timestamp": ts,
            "activity": activity,
            "confidence": confidence,
            "duration": duration,
            "worker_id": worker_id,
            "zone": zone,
        }

    async def get_events(self, limit: int = 100, offset: int = 0) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM events ORDER BY timestamp DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    async def count_events(self) -> int:
        cursor = await self.db.execute("SELECT COUNT(*) as c FROM events")
        row = await cursor.fetchone()
        return row["c"]

    async def get_events_since(self, since_ts: float) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM events WHERE timestamp >= ? ORDER BY timestamp ASC",
            (since_ts,),
        )
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]

    # ---- Annotation projects ----

    async def create_annotation_project(self, project_id: str, name: str,
                                        video_filename: str, video_path: str,
                                        frame_width: int, frame_height: int,
                                        total_frames: int, fps: float) -> dict:
        ts = time.time()
        await self.db.execute(
            "INSERT INTO annotation_projects "
            "(id, name, video_filename, video_path, frame_width, frame_height, total_frames, fps, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (project_id, name, video_filename, video_path, frame_width, frame_height, total_frames, fps, ts),
        )
        await self.db.commit()
        return {
            "id": project_id, "name": name, "video_filename": video_filename,
            "frame_width": frame_width, "frame_height": frame_height,
            "total_frames": total_frames, "fps": fps, "created_at": ts,
        }

    async def list_annotation_projects(self) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM annotation_projects ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            cnt = await self.db.execute(
                "SELECT COUNT(*) as c FROM annotation_frames WHERE project_id = ?", (d["id"],)
            )
            row = await cnt.fetchone()
            d["annotated_frames"] = row["c"]
            result.append(d)
        return result

    async def get_annotation_project(self, project_id: str) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM annotation_projects WHERE id = ?", (project_id,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def delete_annotation_project(self, project_id: str) -> None:
        await self.db.execute("DELETE FROM annotation_frames WHERE project_id = ?", (project_id,))
        await self.db.execute("DELETE FROM annotation_projects WHERE id = ?", (project_id,))
        await self.db.commit()

    async def save_annotation_frame(self, frame_id: str, project_id: str,
                                    frame_index: int, keypoints: list) -> dict:
        ts = time.time()
        kp_json = json.dumps(keypoints)
        await self.db.execute(
            "INSERT INTO annotation_frames (id, project_id, frame_index, keypoints, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(project_id, frame_index) DO UPDATE SET keypoints=excluded.keypoints, updated_at=excluded.updated_at",
            (frame_id, project_id, frame_index, kp_json, ts, ts),
        )
        await self.db.commit()
        return {"id": frame_id, "project_id": project_id, "frame_index": frame_index, "updated_at": ts}

    async def get_annotation_frames(self, project_id: str) -> list[dict]:
        cursor = await self.db.execute(
            "SELECT * FROM annotation_frames WHERE project_id = ? ORDER BY frame_index ASC",
            (project_id,),
        )
        rows = await cursor.fetchall()
        result = []
        for r in rows:
            d = dict(r)
            d["keypoints"] = json.loads(d["keypoints"])
            result.append(d)
        return result

    async def get_annotation_frame(self, project_id: str, frame_index: int) -> dict | None:
        cursor = await self.db.execute(
            "SELECT * FROM annotation_frames WHERE project_id = ? AND frame_index = ?",
            (project_id, frame_index),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        d["keypoints"] = json.loads(d["keypoints"])
        return d
