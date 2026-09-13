import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ml_model import predict_risk

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "machine_registry.db"

app = FastAPI(title="ForgeSight Machine Registry")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class FieldCreate(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    type: str
    required: bool = False
    options: List[str] = []


class MachineCreate(BaseModel):
    values: Dict[str, Any]


class PredictionRequest(BaseModel):
    machine_id: int


def connection():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    return db


def setup_db():
    db = connection()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS fields (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            type TEXT NOT NULL CHECK(type IN ('text', 'number', 'dropdown')),
            required INTEGER NOT NULL DEFAULT 0,
            options TEXT NOT NULL DEFAULT '[]',
            position INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS machines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            "values" TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
    """)
    if db.execute("SELECT COUNT(*) FROM fields").fetchone()[0] == 0:
        defaults = [
            ("Machine Name", "text", 1, [], 0),
            ("Temperature", "number", 1, [], 1),
            ("Pressure", "number", 1, [], 2),
            ("Vibration", "dropdown", 1, ["Low", "Medium", "High"], 3),
        ]
        db.executemany("INSERT INTO fields(name,type,required,options,position) VALUES (?,?,?,?,?)", [(n, t, r, json.dumps(o), p) for n, t, r, o, p in defaults])
    if db.execute("SELECT COUNT(*) FROM machines").fetchone()[0] == 0:
        samples = [
            {"Machine Name": "Press Line A", "Temperature": 85, "Pressure": 120, "Vibration": "High"},
            {"Machine Name": "Cooling Unit B", "Temperature": 62, "Pressure": 98, "Vibration": "Low"},
            {"Machine Name": "Pump Station C", "Temperature": 74, "Pressure": 110, "Vibration": "Medium"},
        ]
        db.executemany("INSERT INTO machines(\"values\") VALUES (?)", [(json.dumps(item),) for item in samples])
    db.commit()
    db.close()


def field_dict(row):
    return {"id": row["id"], "name": row["name"], "type": row["type"], "required": bool(row["required"]), "options": json.loads(row["options"])}


def machine_dict(row):
    return {"id": row["id"], "values": json.loads(row["values"]), "created_at": row["created_at"], "updated_at": row["updated_at"]}


@app.on_event("startup")
def startup():
    setup_db()


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "local"}


@app.get("/api/fields")
def get_fields():
    db = connection()
    rows = db.execute("SELECT * FROM fields ORDER BY position, id").fetchall()
    db.close()
    return [field_dict(row) for row in rows]


@app.post("/api/fields", status_code=201)
def add_field(payload: FieldCreate):
    if payload.type not in {"text", "number", "dropdown"}:
        raise HTTPException(400, "Unsupported field type")
    if payload.type == "dropdown" and len(payload.options) == 0:
        raise HTTPException(400, "Dropdown fields need at least one option")
    db = connection()
    try:
        position = db.execute("SELECT COALESCE(MAX(position), -1) + 1 FROM fields").fetchone()[0]
        cursor = db.execute("INSERT INTO fields(name,type,required,options,position) VALUES (?,?,?,?,?)", (payload.name.strip(), payload.type, int(payload.required), json.dumps(payload.options), position))
        db.commit()
        row = db.execute("SELECT * FROM fields WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return field_dict(row)
    except sqlite3.IntegrityError:
        raise HTTPException(409, "A field with that name already exists")
    finally:
        db.close()


@app.get("/api/machines")
def get_machines():
    db = connection()
    rows = db.execute("SELECT * FROM machines ORDER BY id DESC").fetchall()
    db.close()
    return [machine_dict(row) for row in rows]


def validate_values(values):
    for field in get_fields():
        value = values.get(field["name"])
        if field["required"] and (value is None or str(value).strip() == ""):
            raise HTTPException(422, f'{field["name"]} is required')
        if value not in (None, "") and field["type"] == "number":
            try:
                values[field["name"]] = float(value)
            except (TypeError, ValueError):
                raise HTTPException(422, f'{field["name"]} must be a number')
        if value not in (None, "") and field["type"] == "dropdown" and value not in field["options"]:
            raise HTTPException(422, f'Choose a valid {field["name"]} option')


@app.post("/api/machines", status_code=201)
def create_machine(payload: MachineCreate):
    values = dict(payload.values)
    validate_values(values)
    db = connection()
    cursor = db.execute("INSERT INTO machines(\"values\") VALUES (?)", (json.dumps(values),))
    db.commit()
    row = db.execute("SELECT * FROM machines WHERE id = ?", (cursor.lastrowid,)).fetchone()
    db.close()
    return machine_dict(row)


@app.put("/api/machines/{machine_id}")
def update_machine(machine_id: int, payload: MachineCreate):
    values = dict(payload.values)
    validate_values(values)
    db = connection()
    if db.execute("SELECT id FROM machines WHERE id = ?", (machine_id,)).fetchone() is None:
        db.close()
        raise HTTPException(404, "Machine not found")
    db.execute("UPDATE machines SET \"values\" = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?", (json.dumps(values), machine_id))
    db.commit()
    row = db.execute("SELECT * FROM machines WHERE id = ?", (machine_id,)).fetchone()
    db.close()
    return machine_dict(row)


@app.delete("/api/machines/{machine_id}", status_code=204)
def delete_machine(machine_id: int):
    db = connection()
    cursor = db.execute("DELETE FROM machines WHERE id = ?", (machine_id,))
    db.commit()
    db.close()
    if cursor.rowcount == 0:
        raise HTTPException(404, "Machine not found")


@app.post("/api/predict")
def predict(payload: PredictionRequest):
    db = connection()
    row = db.execute("SELECT * FROM machines WHERE id = ?", (payload.machine_id,)).fetchone()
    db.close()
    if row is None:
        raise HTTPException(404, "Machine not found")
    result = predict_risk(json.loads(row["values"]))
    return {"machine_id": payload.machine_id, "machine_name": json.loads(row["values"]).get("Machine Name", "Unnamed"), **result}


app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")
