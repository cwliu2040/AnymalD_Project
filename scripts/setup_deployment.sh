#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
ROS_WORKSPACE="${PROJECT_ROOT}/deployment/ros2_ws"
LIO_SAM_ROOT="${ROS_WORKSPACE}/src/lio_sam"
PYTHON_VENDOR="${PROJECT_ROOT}/deployment/python_vendor"
POLICY_ROOT="${PROJECT_ROOT}/exported/anymal_d_locomotion_v1/recovery_v0.4.0"
FACTORY_USD="${PROJECT_ROOT}/assets/maps/factory/Factory_Layout.usd"
ISAACLAB_PATH="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"
EXPECTED_LIO_SAM_COMMIT="08af3f32f01725372d4269838dc44c19c6d9e76b"
CHECK_ONLY=false

usage() {
    echo "Usage: $0 [--check]"
    echo
    echo "Without arguments, fetch project dependencies and build the ROS 2 workspace."
    echo "With --check, only verify an already prepared checkout."
}

fail() {
    echo "ERROR: $*" >&2
    exit 1
}

if [[ $# -gt 1 ]]; then
    usage
    exit 2
fi
if [[ $# -eq 1 ]]; then
    case "$1" in
        --check)
            CHECK_ONLY=true
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            usage
            exit 2
            ;;
    esac
fi

for command_name in git git-lfs vcs colcon rosdep python3 sha256sum; do
    command -v "${command_name}" >/dev/null 2>&1 \
        || fail "Missing command '${command_name}'. Follow README prerequisites."
done

[[ -f "${ROS_SETUP}" ]] \
    || fail "ROS 2 Humble is missing: ${ROS_SETUP}"
[[ -x "/opt/ros/humble/bin/iox-roudi" ]] \
    || fail "Iceoryx RouDi is missing: install ros-humble-iceoryx-posh"
[[ -f "/opt/ros/humble/lib/librmw_cyclonedds_cpp.so" ]] \
    || fail "CycloneDDS RMW is missing: install ros-humble-rmw-cyclonedds-cpp"
[[ -x "${ISAACLAB_PATH}/isaaclab.sh" ]] \
    || fail "Isaac Lab is missing: ${ISAACLAB_PATH}/isaaclab.sh"
[[ -x "${ISAACLAB_PATH}/_isaac_sim/python.sh" ]] \
    || fail "Isaac Sim is not linked at ${ISAACLAB_PATH}/_isaac_sim"

if [[ "${CHECK_ONLY}" == false ]]; then
    echo "[1/4] Fetching Git LFS assets..."
    git -C "${PROJECT_ROOT}" lfs pull

    echo "[2/4] Importing pinned LIO-SAM source..."
    if [[ -e "${LIO_SAM_ROOT}" && ! -d "${LIO_SAM_ROOT}/.git" ]]; then
        [[ -d "${LIO_SAM_ROOT}" ]] \
            || fail "LIO-SAM path exists but is not a directory: ${LIO_SAM_ROOT}"
        rmdir "${LIO_SAM_ROOT}" 2>/dev/null \
            || fail "LIO-SAM path is non-empty but is not a Git checkout: ${LIO_SAM_ROOT}"
        echo "Removed an empty LIO-SAM directory left by an interrupted import."
    fi
    vcs import \
        --input "${ROS_WORKSPACE}/lio_sam.repos" \
        --skip-existing \
        "${ROS_WORKSPACE}/src"

    echo "[3/4] Installing ROS and ONNX dependencies..."
    # ROS 2 Humble's generated setup scripts reference variables that may be
    # unset. Temporarily disable nounset while sourcing, then restore it for
    # the remainder of this script.
    set +u
    # shellcheck disable=SC1090
    source "${ROS_SETUP}"
    set -u
    rosdep install \
        --from-paths "${ROS_WORKSPACE}/src" \
        --ignore-src \
        --rosdistro humble \
        --dependency-types build \
        --dependency-types buildtool \
        --dependency-types exec \
        --skip-keys ament_python \
        -r \
        -y
    python3 -m pip install \
        --upgrade \
        --target "${PYTHON_VENDOR}" \
        -r "${ROS_WORKSPACE}/requirements-inference.txt"

    echo "[4/4] Building the ROS 2 workspace..."
    (
        cd "${ROS_WORKSPACE}"
        colcon build \
            --symlink-install \
            --packages-up-to anymal_locomotion_ros2
    )
fi

git -C "${PROJECT_ROOT}" lfs fsck
[[ -f "${FACTORY_USD}" ]] || fail "Factory map is missing: ${FACTORY_USD}"
[[ -f "${POLICY_ROOT}/policy.onnx" ]] || fail "ONNX policy is missing"
[[ -f "${POLICY_ROOT}/policy.pt" ]] || fail "TorchScript policy is missing"
[[ -f "${POLICY_ROOT}/policy_metadata.yaml" ]] || fail "Policy metadata is missing"

(
    cd "${PROJECT_ROOT}"
    printf '%s  %s\n' \
        "721a918533cd00e605cf6edfcd8bf8bba9cbd56f26e22c2eda26604f03ee54c0" \
        "exported/anymal_d_locomotion_v1/recovery_v0.4.0/policy.onnx" \
        "95f20f7618adc9be808d9b3d424d61116abf52012d88b3e92c9c0ff6e55cfe5b" \
        "exported/anymal_d_locomotion_v1/recovery_v0.4.0/policy.pt" \
        | sha256sum --check --status
) || fail "Deployment policy checksum mismatch"

[[ -d "${LIO_SAM_ROOT}/.git" ]] || fail "Pinned LIO-SAM checkout is missing"
ACTUAL_LIO_SAM_COMMIT="$(git -C "${LIO_SAM_ROOT}" rev-parse HEAD)"
[[ "${ACTUAL_LIO_SAM_COMMIT}" == "${EXPECTED_LIO_SAM_COMMIT}" ]] \
    || fail "LIO-SAM commit mismatch: ${ACTUAL_LIO_SAM_COMMIT}"

[[ -f "${ROS_WORKSPACE}/install/setup.bash" ]] \
    || fail "ROS workspace is not built"
PYTHONPATH="${PYTHON_VENDOR}${PYTHONPATH:+:${PYTHONPATH}}" \
    python3 -c "from onnx.reference import ReferenceEvaluator" \
    || fail "Project-local ONNX runtime is unavailable"

echo
echo "Deployment checkout is ready."
echo "Run:"
echo "  source /opt/ros/humble/setup.bash"
echo "  source ${ROS_WORKSPACE}/install/setup.bash"
echo "  ros2 launch anymal_locomotion_ros2 bringup.launch.py"
