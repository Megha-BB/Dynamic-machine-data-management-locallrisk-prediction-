# ForgeSight Machine Registry

A local machine registry demonstrating dynamic fields, CRUD records, and a Python ML risk prediction flow. No cloud AI services or external AI APIs are used.

## Stack

- **Backend:** Python, FastAPI, SQLite
- **ML:** scikit-learn `DictVectorizer` + `LogisticRegression`, trained on small synthetic examples
- **Frontend:** HTML, CSS, and vanilla JavaScript served by FastAPI

## Run locally

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Start the local application (this also starts the local Python ML component inside the backend process):

   ```powershell
   uvicorn main:app --reload
   ```

4. Open <http://127.0.0.1:8000>.

The SQLite file `machine_registry.db` is created automatically with the initial fields and sample machines. It is runtime data and should not be committed.

## Architecture

The browser calls FastAPI endpoints for field configuration and machine CRUD. Field definitions are stored in the `fields` table. Machine values are stored as JSON in the `machines.values` column, which means a new field does not require a database schema change. The prediction endpoint loads a machine's JSON values and passes them to `ml_model.py`.

## Dynamic fields and Humidity

Adding `Humidity` through **Field schema** creates a new field configuration and causes it to appear in the generated create/edit form, table, and record JSON automatically. The existing model intentionally continues to use only Temperature, Pressure, and Vibration; extra fields are safely ignored during prediction.

To make the model use Humidity later, the training dataset would need humidity values and labels, the feature preparation in `ml_model.py` would need to include it, and the model would need to be retrained or replaced. Automatic retraining is intentionally outside this demonstration.

## API surface

- `GET/POST /api/fields`
- `GET/POST/PUT/DELETE /api/machines`
- `POST /api/predict`
- `GET /api/health`
