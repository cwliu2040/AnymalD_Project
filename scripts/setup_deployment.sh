#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROS_SETUP="/opt/ros/humble/setup.bash"
ROS_WORKSPACE="${PROJECT_ROOT}/deployment/ros2_ws"
LIO_SAM_ROOT="${ROS_WORKSPACE}/src/lio_sam"
FAST_LIO_ROOT="${ROS_WORKSPACE}/src/fast_lio"
FAST_LIO_PACKAGE="${FAST_LIO_ROOT}/FAST_LIO"
LIVOX_DRIVER_ROOT="${ROS_WORKSPACE}/src/livox_ros_driver2"
LIVOX_SDK_ROOT="${ROS_WORKSPACE}/src/livox_sdk2"
LIVOX_SDK_BUILD="${ROS_WORKSPACE}/build/livox_sdk2"
LIVOX_SDK_INSTALL="${ROS_WORKSPACE}/vendor/livox_sdk2"
FAST_LIO_PATCH="${PROJECT_ROOT}/docs/validation/fastlio2_map_pub_downstream.patch"
FAST_LIO_DIAGNOSTICS_PATCH="${PROJECT_ROOT}/docs/validation/fastlio2_effect_diagnostics_downstream.patch"
THIRD_PARTY_GITIGNORE="${ROS_WORKSPACE}/third_party.gitignore"
PYTHON_VENDOR="${PROJECT_ROOT}/deployment/python_vendor"
POLICY_ROOT="${PROJECT_ROOT}/exported/anymal_d_locomotion_v1/recovery_v0.4.0"
FACTORY_USD="${PROJECT_ROOT}/assets/maps/factory/Factory_Layout.usd"
ISAACLAB_PATH="${ISAACLAB_ROOT:-${HOME}/IsaacLab}"
EXPECTED_LIO_SAM_COMMIT="08af3f32f01725372d4269838dc44c19c6d9e76b"
EXPECTED_FAST_LIO_COMMIT="373aa886402b6307db2995ca12b3f4596ef4f633"
EXPECTED_IKD_TREE_COMMIT="e2e3f4e9d3b95a9e66b1ba83dc98d4a05ed8a3c4"
EXPECTED_LIVOX_DRIVER_COMMIT="bf4d062a2ac4030272be6fe1ee1505a4eb2ffdcb"
EXPECTED_LIVOX_SDK_COMMIT="08f523c930b2f0ba1e98a6afaa8d7476bf479908"
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

for command_name in git git-lfs vcs colcon rosdep python3 sha256sum cmake patch; do
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
    echo "[1/5] Fetching Git LFS assets..."
    git -C "${PROJECT_ROOT}" lfs pull

    echo "[2/5] Importing pinned ROS 2 sources..."
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

    vcs import \
        --input "${ROS_WORKSPACE}/fastlio2.repos" \
        --skip-existing \
        "${ROS_WORKSPACE}/src"
    git -C "${FAST_LIO_ROOT}" submodule update --init --recursive
    # The candidate embeds a second Livox driver checkout, while SDK2 is a
    # plain CMake dependency.  Neither should be discovered as a colcon package.
    cmake -E touch "${FAST_LIO_ROOT}/livox_ros_driver2/COLCON_IGNORE"
    cmake -E touch "${LIVOX_SDK_ROOT}/COLCON_IGNORE"
    git -C "${FAST_LIO_ROOT}/livox_ros_driver2" config \
        core.excludesFile "${THIRD_PARTY_GITIGNORE}"
    git -C "${LIVOX_DRIVER_ROOT}" config \
        core.excludesFile "${THIRD_PARTY_GITIGNORE}"
    git -C "${LIVOX_SDK_ROOT}" config \
        core.excludesFile "${THIRD_PARTY_GITIGNORE}"
    cp "${LIVOX_DRIVER_ROOT}/package_ROS2.xml" \
        "${LIVOX_DRIVER_ROOT}/package.xml"
    if git -C "${FAST_LIO_ROOT}" apply --unidiff-zero --check "${FAST_LIO_PATCH}"; then
        git -C "${FAST_LIO_ROOT}" apply --unidiff-zero "${FAST_LIO_PATCH}"
    elif ! git -C "${FAST_LIO_ROOT}" apply --unidiff-zero --reverse --check "${FAST_LIO_PATCH}"; then
        fail "FAST-LIO2 downstream patch cannot be applied cleanly"
    fi
    if git -C "${FAST_LIO_ROOT}" apply --check "${FAST_LIO_DIAGNOSTICS_PATCH}"; then
        git -C "${FAST_LIO_ROOT}" apply "${FAST_LIO_DIAGNOSTICS_PATCH}"
    elif ! git -C "${FAST_LIO_ROOT}" apply --reverse --check "${FAST_LIO_DIAGNOSTICS_PATCH}"; then
        fail "FAST-LIO2 diagnostics patch cannot be applied cleanly"
    fi

    echo "[3/5] Building pinned Livox SDK2..."
    cmake -S "${LIVOX_SDK_ROOT}" -B "${LIVOX_SDK_BUILD}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="${LIVOX_SDK_INSTALL}"
    cmake --build "${LIVOX_SDK_BUILD}" --target install --parallel 2

    echo "[4/5] Installing ROS and ONNX dependencies..."
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

    echo "[5/5] Building the ROS 2 workspace..."
    (
        cd "${ROS_WORKSPACE}"
        colcon build \
            --symlink-install \
            --packages-up-to anymal_locomotion_ros2 \
            --cmake-args \
                -DCMAKE_BUILD_TYPE=Release \
                -DROS_EDITION=ROS2 \
                -DHUMBLE_ROS=humble \
                -DLIVOX_LIDAR_SDK_LIBRARY="${LIVOX_SDK_INSTALL}/lib/liblivox_lidar_sdk_shared.so" \
                -DLIVOX_LIDAR_SDK_INCLUDE_DIR="${LIVOX_SDK_INSTALL}/include"
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

for dependency_root in \
    "${FAST_LIO_ROOT}" \
    "${FAST_LIO_ROOT}/FAST_LIO/include/ikd-Tree" \
    "${LIVOX_DRIVER_ROOT}" \
    "${LIVOX_SDK_ROOT}"; do
    git -C "${dependency_root}" rev-parse --git-dir >/dev/null 2>&1 \
        || fail "Pinned dependency checkout is missing: ${dependency_root}"
done
[[ "$(git -C "${FAST_LIO_ROOT}" rev-parse HEAD)" == "${EXPECTED_FAST_LIO_COMMIT}" ]] \
    || fail "FAST-LIO2 commit mismatch"
[[ "$(git -C "${FAST_LIO_ROOT}/FAST_LIO/include/ikd-Tree" rev-parse HEAD)" == "${EXPECTED_IKD_TREE_COMMIT}" ]] \
    || fail "ikd-Tree commit mismatch"
[[ "$(git -C "${LIVOX_DRIVER_ROOT}" rev-parse HEAD)" == "${EXPECTED_LIVOX_DRIVER_COMMIT}" ]] \
    || fail "Livox ROS driver commit mismatch"
[[ "$(git -C "${LIVOX_SDK_ROOT}" rev-parse HEAD)" == "${EXPECTED_LIVOX_SDK_COMMIT}" ]] \
    || fail "Livox SDK2 commit mismatch"
cmp -s "${LIVOX_DRIVER_ROOT}/package_ROS2.xml" "${LIVOX_DRIVER_ROOT}/package.xml" \
    || fail "Livox ROS 2 package manifest is missing or stale"
[[ -f "${FAST_LIO_ROOT}/livox_ros_driver2/COLCON_IGNORE" ]] \
    || fail "Nested Livox driver is not excluded from colcon discovery"
[[ -f "${LIVOX_SDK_ROOT}/COLCON_IGNORE" ]] \
    || fail "Livox SDK2 is not excluded from colcon discovery"
git -C "${FAST_LIO_ROOT}" apply --unidiff-zero --reverse --check "${FAST_LIO_PATCH}" \
    || fail "FAST-LIO2 downstream map publisher patch is missing"
git -C "${FAST_LIO_ROOT}" apply --reverse --check "${FAST_LIO_DIAGNOSTICS_PATCH}" \
    || fail "FAST-LIO2 downstream diagnostics patch is missing"
[[ -f "${LIVOX_SDK_INSTALL}/lib/liblivox_lidar_sdk_shared.so" ]] \
    || fail "Pinned Livox SDK2 installation is missing"

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
