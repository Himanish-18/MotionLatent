"""
PAMAP2 schema definitions.

Defines column names, sensor locations, modalities, and the activity
dictionary for the PAMAP2 Physical Activity Monitoring dataset.

Reference: PAMAP2 readme.pdf — Column ordering for the 54-column format.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


# ===================================================================
# Activity dictionary  (from PAMAP2 documentation)
# ===================================================================
ACTIVITY_MAP: Dict[int, str] = {
    0: "transient",
    1: "lying",
    2: "sitting",
    3: "standing",
    4: "walking",
    5: "running",
    6: "cycling",
    7: "Nordic_walking",
    9: "watching_TV",
    10: "computer_work",
    11: "car_driving",
    12: "ascending_stairs",
    13: "descending_stairs",
    16: "vacuum_cleaning",
    17: "ironing",
    18: "folding_laundry",
    19: "house_cleaning",
    20: "playing_soccer",
    24: "rope_jumping",
}

# Protocol activities (IDs 1-7, 12-13, 16-17, 24)
PROTOCOL_ACTIVITY_IDS = [1, 2, 3, 4, 5, 6, 7, 12, 13, 16, 17, 24]

# Optional activities
OPTIONAL_ACTIVITY_IDS = [9, 10, 11, 18, 19, 20]


# ===================================================================
# Sensor locations and modalities
# ===================================================================
SENSOR_LOCATIONS = ["hand", "chest", "ankle"]

# IMU column sub-groups: 17 columns per IMU (per the PAMAP2 readme)
# temperature(1) + acc_16g(3) + acc_6g(3) + gyroscope(3) + magnetometer(3) + orientation(4)
IMU_SUBCOLUMNS = [
    "temperature",
    "acc_16g_x", "acc_16g_y", "acc_16g_z",
    "acc_6g_x", "acc_6g_y", "acc_6g_z",
    "gyro_x", "gyro_y", "gyro_z",
    "mag_x", "mag_y", "mag_z",
    "orientation_0", "orientation_1", "orientation_2", "orientation_3",
]

# Total columns: timestamp(1) + activityID(1) + heart_rate(1) + 3 IMUs × 17 = 54
TOTAL_COLUMNS = 54


def build_column_names() -> List[str]:
    """Build the full ordered list of 54 column names."""
    cols = ["timestamp", "activity_id", "heart_rate"]
    for loc in SENSOR_LOCATIONS:
        for sub in IMU_SUBCOLUMNS:
            cols.append(f"{loc}_{sub}")
    assert len(cols) == TOTAL_COLUMNS, f"Expected {TOTAL_COLUMNS}, got {len(cols)}"
    return cols


COLUMN_NAMES: List[str] = build_column_names()


# ===================================================================
# Column classification
# ===================================================================
@dataclass
class ColumnClassification:
    """Classification of PAMAP2 columns into semantic groups."""

    metadata: List[str]
    heart_rate: List[str]
    temperature: List[str]
    accelerometer_16g: List[str]
    accelerometer_6g: List[str]
    gyroscope: List[str]
    magnetometer: List[str]
    orientation: List[str]


def classify_columns() -> ColumnClassification:
    """Classify all 54 columns into semantic groups."""
    meta = ["timestamp", "activity_id"]
    hr = ["heart_rate"]
    temp, acc16, acc6, gyro, mag, orient = [], [], [], [], [], []

    for loc in SENSOR_LOCATIONS:
        temp.append(f"{loc}_temperature")
        for axis in ["x", "y", "z"]:
            acc16.append(f"{loc}_acc_16g_{axis}")
            acc6.append(f"{loc}_acc_6g_{axis}")
            gyro.append(f"{loc}_gyro_{axis}")
            mag.append(f"{loc}_mag_{axis}")
        for i in range(4):
            orient.append(f"{loc}_orientation_{i}")

    return ColumnClassification(
        metadata=meta,
        heart_rate=hr,
        temperature=temp,
        accelerometer_16g=acc16,
        accelerometer_6g=acc6,
        gyroscope=gyro,
        magnetometer=mag,
        orientation=orient,
    )


# Pre-built classification
COLUMN_GROUPS = classify_columns()


def get_primary_motion_columns() -> List[str]:
    """Return the columns selected for the primary MotionLatent
    motion representation: acc_16g + gyro + mag across all 3 locations.

    Rationale:
    - acc_16g is preferred over acc_6g (wider range for vigorous activities).
    - Orientation quaternions are *invalid* per PAMAP2 readme (all NaN).
    - Temperature is environmental, not motion.
    - Heart rate is physiological, not motion.
    - acc_6g is excluded to avoid redundancy but retained for later
      experiments.
    """
    return (
        COLUMN_GROUPS.accelerometer_16g
        + COLUMN_GROUPS.gyroscope
        + COLUMN_GROUPS.magnetometer
    )


def get_excluded_columns() -> Dict[str, Tuple[List[str], str]]:
    """Return columns excluded from the primary motion representation
    along with exclusion rationale."""
    return {
        "heart_rate": (
            COLUMN_GROUPS.heart_rate,
            "Physiological signal, not motion sensing.",
        ),
        "temperature": (
            COLUMN_GROUPS.temperature,
            "Environmental/skin temperature, not motion.",
        ),
        "accelerometer_6g": (
            COLUMN_GROUPS.accelerometer_6g,
            "Redundant with acc_16g; narrower range. "
            "Retained for potential later comparison.",
        ),
        "orientation": (
            COLUMN_GROUPS.orientation,
            "PAMAP2 readme states orientation data is INVALID "
            "(all NaN). Must not enter the ML pipeline.",
        ),
    }


# Nominal sampling frequency (from PAMAP2 documentation)
NOMINAL_SAMPLING_FREQ_HZ = 100.0

# Heart rate sampling frequency (different sensor)
HEART_RATE_SAMPLING_FREQ_HZ = 9.0  # ~9 Hz, interpolated to 100 Hz in dataset
