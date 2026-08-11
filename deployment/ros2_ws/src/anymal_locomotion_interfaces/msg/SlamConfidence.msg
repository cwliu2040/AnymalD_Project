# Backend-neutral, atomic SLAM tracking-quality snapshot. See
# docs/slam_confidence_contract.md for the normative semantics.

uint8 SCHEMA_VERSION=1

uint8 STATE_INITIALIZING=0
uint8 STATE_TRACKING=1
uint8 STATE_DEGRADED=2
uint8 STATE_LOST=3

uint32 REASON_NONE=0
uint32 REASON_INITIALIZING=1
uint32 REASON_SOURCE_STALE=2
uint32 REASON_ODOMETRY_STALE=4
uint32 REASON_LIDAR_STALE=8
uint32 REASON_IMU_STALE=16
uint32 REASON_INSUFFICIENT_SUPPORT=32
uint32 REASON_DEGENERATE_GEOMETRY=64
uint32 REASON_HIGH_RESIDUAL=128
uint32 REASON_NOT_CONVERGED=256
uint32 REASON_ESTIMATOR_RESET=512
uint32 REASON_TIMESTAMP_INVALID=1024
uint32 REASON_NUMERIC_INVALID=2048
uint32 REASON_BACKEND_ERROR=4096
uint32 REASON_SIGNAL_MISSING=8192
uint32 REASON_LOW_CONFIDENCE=16384
uint32 REASON_RECOVERY_PENDING=32768
uint32 REASON_CLOCK_RESET=65536
uint32 REASON_UNCALIBRATED=131072

uint8 schema_version
# Algorithm family only. Exact source/config provenance belongs to the
# calibration artifact identified by calibration_id.
string<=32 backend_id

# Fingerprint of the versioned calibration/provenance artifact. A missing or
# incompatible artifact must fail closed with REASON_UNCALIBRATED.
string<=128 calibration_id

# True when source_stamp and confidence_age below are meaningful. ROS time zero
# is valid and therefore must not be used as a no-source sentinel.
bool source_stamp_valid
# Stamp of the canonical /slam/odom pose evaluated by this confidence sample.
# It must exactly match that Odometry header stamp; newer IMU/LiDAR callbacks
# must not advance it on their own.
builtin_interfaces/Time source_stamp
# ROS-clock time at which the extractor evaluated this snapshot.
builtin_interfaces/Time evaluation_stamp
# Snapshot of max(0, evaluation_stamp - source_stamp). Consumers must
# recompute current age from source_stamp rather than trusting this value later.
builtin_interfaces/Duration confidence_age

# Calibrated probability of common short-horizon pose usability in [0, 1],
# derived from backend-local runtime signals. It may retain its last value when
# tracking_valid is false, including the deliberate stale-high-confidence case.
float32 slam_confidence
# Hard pose-usability gate after freshness checks and validity hysteresis.
bool slam_tracking_valid
uint8 tracking_state
# Bitwise OR of REASON_* values. REASON_NONE is only valid when the value is 0.
uint32 degradation_reasons
