# Pipeline Data Flow & Data Dictionary

This document details the end-to-end logic of the streaming prediction pipeline and provides a dictionary for the data fields used at each stage.

## 1. Pipeline Overview

The pipeline is designed to predict the **Minutes Between Trains (MBT)** for the NYC Subway (specifically Route E at stop F11S) using the NHITS deep learning model.

**Flow:**
`MTA Feed` -> `Ingestion Service` -> `Pub/Sub` -> `Dataflow Pipeline` -> `Vertex AI Endpoint` -> `Firestore`

---

## 2. Step-by-Step Logic

### Step 1: Ingestion (Producer)
*   **Source:** Polls the MTA GTFS-Realtime API (Feed `nyct/gtfs-ace`) every 30 seconds.
*   **Action:** Fetches the Protobuf feed, converts it to JSON, and publishes it to a Google Cloud Pub/Sub topic.
*   **Data:** Contains a list of "Entities", where each entity represents a vehicle (train) or a trip update.

### Step 2: Parsing & Filtering
*   **Component:** `ParseVehicleUpdates` (in `streaming/transform.py`)
*   **Logic:**
    1.  Reads raw JSON messages from Pub/Sub.
    2.  Iterates through all entities.
    3.  **Filters:** Keeps only updates for:
        *   **Route:** `E`
        *   **Stops:** Origin (`G05S` - Jamaica Center) OR Target (`F11S` - 7th Ave).
    4.  **Status Check:** Captures both `STOPPED_AT` (actual arrival) and `UNKNOWN` (often scheduled/future) statuses for logging, but downstream logic relies on timestamps.

### Step 3: Duration Calculation (Stateful)
*   **Component:** `CalculateTripDuration` (in `streaming/transform.py`)
*   **Logic:**
    *   Uses **Stateful Processing** keyed by `trip_id`.
    *   **On Origin Arrival (`G05S`):** Stores the arrival timestamp in the state (`origin_ts`).
    *   **On Target Arrival (`F11S`):**
        1.  Retrieves the stored `origin_ts` for this specific trip.
        2.  Calculates `duration = target_timestamp - origin_timestamp`.
        3.  Emits the duration if valid (> 0).

### Step 4: Windowing & Aggregation
*   **Component:** `AccumulateArrivals` (in `streaming/transform.py`)
*   **Logic:**
    *   Accumulates a history of the last 150 arrivals for the Route/Stop key (`E_F11S`).
    *   This history is required to calculate rolling statistics (trends) for the model.

### Step 5: Feature Engineering & Prediction
*   **Component:** `VertexAIPrediction` (in `streaming/prediction.py`)
*   **Logic:**
    1.  **MBT Calculation:** Calculates "Minutes Between Trains" (the target variable) by taking the difference between consecutive arrival timestamps in the history window.
    2.  **Rolling Stats:** Calculates mean, standard deviation, and max MBT over the last 10 and 50 intervals.
    3.  **Exogenous Features:**
        *   **Calendar:** Adds Day of Week (`dow`).
        *   **Weather:** Fetches live weather (Temp, Wind, etc.) for the timestamp.
    4.  **Inference:** Sends this feature vector to the **Vertex AI Endpoint** (hosting the NHITS model).
    5.  **Output:** Receives a forecast (next 10 MBT values).

---

## 3. Data Dictionary

### A. Raw Ingestion (GTFS-Realtime)
| Field | Type | Description |
|-------|------|-------------|
| `trip_id` | String | Unique identifier for a specific train run (e.g., `062600_E..S71R`). Often contains scheduled start time. |
| `route_id` | String | The subway line (e.g., `E`, `A`, `C`). |
| `stop_id` | String | The specific station ID (e.g., `G05S`, `F11S`). |
| `current_status` | String | `STOPPED_AT` (at station), `IN_TRANSIT_TO` (moving), or `UNKNOWN` (often scheduled). |
| `timestamp` | Integer | Unix timestamp (seconds since epoch) of the update. |

### B. Intermediate Processing
| Field | Type | Description |
|-------|------|-------------|
| `duration` | Float | Time (in minutes) taken for the train to travel from Origin to Target. |
| `mbt` (or `y`) | Float | **Minutes Between Trains**. The time difference between the current train's arrival and the previous train's arrival. This is what the model predicts. |

### C. Model Feature Vector (Input to NHITS)
These fields are generated for every arrival and sent to the model.

| Field | Type | Description |
|-------|------|-------------|
| `ds` | String | Datetime string (YYYY-MM-DD HH:MM:SS) of the arrival. |
| `unique_id` | String | Identifier for the time series (e.g., `E`). |
| `y` | Float | The target variable (MBT) for this step. |
| `duration` | Float | The trip duration for this specific train. |
| `rolling_mean_10` | Float | Average MBT over the last 10 trains. Captures short-term frequency. |
| `rolling_std_10` | Float | Standard deviation of MBT over last 10 trains. Captures irregularity. |
| `rolling_max_10` | Float | Longest gap between trains in the last 10 arrivals. |
| `rolling_mean_50` | Float | Average MBT over the last 50 trains. Captures longer-term trends. |
| `rolling_std_50` | Float | Standard deviation of MBT over last 50 trains. |
| `dow` | Integer | Day of Week (0=Monday, 6=Sunday). Used to capture weekend schedules. |
| `temp` | Float | Temperature (Fahrenheit). |
| `precip` | Float | Precipitation amount. |
| `windspeed` | Float | Wind speed (mph). |
| `visibility` | Float | Visibility (miles). |

### D. Prediction Output
| Field | Type | Description |
|-------|------|-------------|
| `forecast` | List[Float] | A list of predicted MBT values for the next *N* trains (e.g., next 10 arrivals). |
