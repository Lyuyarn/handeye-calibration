import os
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R


class HTM:
    """
    简单的齐次变换矩阵封装，兼容原 notebook/engine 中的用法：
    - 属性：R(3x3)、P(3,) 、matrix(4x4)、inv
    - 支持 HTM @ HTM 进行链式相乘
    """

    def __init__(
        self,
        matrix: Union[np.ndarray, None] = None,
        R: Union[np.ndarray, None] = None,
        P: Union[np.ndarray, None] = None,
        rvec: Union[np.ndarray, None] = None,
        tvec: Union[np.ndarray, None] = None,
    ) -> None:
        if matrix is not None:
            self.matrix = np.array(matrix, dtype=float)
        else:
            self.matrix = np.eye(4, dtype=float)
            if R is not None:
                self.matrix[:3, :3] = np.array(R, dtype=float)
            if P is not None:
                p = np.array(P, dtype=float).reshape(3)
                self.matrix[:3, 3] = p

        self.R = self.matrix[:3, :3]
        self.P = self.matrix[:3, 3]

        self.rvec = rvec
        self.tvec = tvec

    @property
    def inv(self) -> "HTM":
        return HTM(matrix=np.linalg.inv(self.matrix))

    def __matmul__(self, other: "HTM") -> "HTM":
        if isinstance(other, HTM):
            return HTM(matrix=self.matrix @ other.matrix)
        return HTM(matrix=self.matrix @ np.asarray(other, dtype=float))

    def __repr__(self) -> str:
        return f"HTM(R={self.R!r}, P={self.P!r})"


def read_robot_poses(file_path: Union[str, Path]) -> List[HTM]:
    """
    从文本文件读取机器人位姿（x, y, z, roll, pitch, yaw），
    每行 6 个浮点数，roll/pitch/yaw 为 XYZ 欧拉角（弧度），
    返回 HTM 列表，方便后续直接使用 .R / .P / .inv。
    """
    poses: List[HTM] = []
    file_path = Path(file_path)
    with file_path.open("r") as f:
        lines = f.readlines()

    for line in lines:
        try:
            data = list(map(float, line.split(",")))
            if len(data) == 6:
                x, y, z = data[0:3]
                roll, pitch, yaw = data[3:6]
                rotation = R.from_euler("xyz", [roll, pitch, yaw])
                R_matrix = rotation.as_matrix()
                t_vec = np.array([x, y, z], dtype=float)
                poses.append(HTM(R=R_matrix, P=t_vec))
            else:
                print(f"Invalid data in line: {line}")
        except Exception as e:
            print(f"Error processing line: {line}. Error: {e}")
            continue

    return poses


# 相机内参（与 notebook 中保持一致）
CAMERA_MATRIX = np.array([
        [2414.1939545640535, 0, 967.3404409524326],
        [0, 2414.4741049405593, 618.9615768349123],
        [0, 0, 1]
    ])

DIST_COEFFS = np.zeros((5, 1))  # 因为 d 全为 0


def get_board2cam_htm_from_pic(
    img_path: Union[str, Path],
    board: "cv2.aruco_CharucoBoard",
    visualize: bool = False,
) -> Union[HTM, None]:
    """
    使用 Charuco 棋盘从图片中估计 board→cam 的位姿，返回 HTM。
    失败时返回 None。
    """
    img_path = Path(img_path)
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"无法读取图像: {img_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

    if charuco_ids is None or len(charuco_ids) < 4:
        print(f"{img_path} ❌ 棋盘角点不足，跳过")
        return None

    ok, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        charuco_corners,
        charuco_ids,
        board,
        CAMERA_MATRIX,
        DIST_COEFFS,
        None,
        None
    )
    if not ok:
        print(f"{img_path} ❌ estimatePoseCharucoBoard 失败")
        return None

    R_cam, _ = cv2.Rodrigues(rvec)
    t_cam = tvec.reshape(3)

    htm = HTM(R=R_cam, P=t_cam, rvec=rvec, tvec=tvec)

    if visualize:
        cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
        cv2.drawFrameAxes(img, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.05)
        cv2.imshow("charuco pose", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return htm


def create_charuco_board() -> "cv2.aruco_CharucoBoard":
    """
    创建与笔记本中一致的 Charuco 标定板。
    参数来自 calib.io 生成的棋盘配置：
    - 内角点数量 w, h
    - 方格边长 squarelength (mm)
    - ArUco 标记边长 markerlength (mm)
    """
    # 标定板内角点个数（与笔记本一致）
    w, h = 7, 5

    # 方格尺寸 / 标记尺寸（单位：mm）
    squarelength_mm, markerlength_mm = 35, 26

    # 简单的边距检查（与笔记本逻辑一致）
    n = 4  # DICT_4X4_250 有 4×4 个模块
    module_size = markerlength_mm / n
    required_margin = 0.7 * module_size
    actual_margin = (squarelength_mm - markerlength_mm) / 2.0

    if actual_margin < required_margin:
        print(
            f"⚠️ 标定板边距可能不足：需要 {required_margin:.4f}mm，"
            f"实际 {actual_margin:.4f}mm"
        )

    # 转为米，供 OpenCV 使用
    squarelength = squarelength_mm / 1000.0
    markerlength = markerlength_mm / 1000.0

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    board = cv2.aruco.CharucoBoard(
        size=(w, h),
        squareLength=squarelength,
        markerLength=markerlength,
        dictionary=dictionary,
    )
    return board


def load_board2cam_poses(
    src_dir: Path, board
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Path], List[int]]:
    """
    使用 Charuco 棋盘提取每张图片的 board→cam 位姿。
    返回：
    - R_board2cam: List[(3,3)]
    - t_board2cam: List[(3,1)]
    - image_paths: 有效图片路径（去除失败图像）
    - error_ids: 失败图像的原始索引
    """
    dst_dir = src_dir
    if not dst_dir.exists():
        raise FileNotFoundError(f"标定图片目录不存在: {dst_dir}")

    image_paths = list(dst_dir.rglob("*.png"))
    image_paths.sort(key=lambda x: x.name)

    if not image_paths:
        raise RuntimeError(f"在目录 {dst_dir} 下未找到任何 PNG 标定图片")

    print(f"🔍 共找到 {len(image_paths)} 张 PNG 标定图片")

    R_board2cam: List[np.ndarray] = []
    t_board2cam: List[np.ndarray] = []
    valid_paths: List[Path] = []
    error_ids: List[int] = []

    for idx, img_path in enumerate(image_paths):
        htm_board2cam = get_board2cam_htm_from_pic(img_path, board)
        if htm_board2cam is None:
            print(f"[{idx:02d}] {img_path} ❌ 棋盘位姿提取失败，跳过")
            error_ids.append(idx)
            continue

        print(f"[{idx:02d}] {img_path} ✅ 棋盘位姿提取成功")
        R_board2cam.append(htm_board2cam.R)
        t_board2cam.append(htm_board2cam.P)
        valid_paths.append(img_path)

    if not R_board2cam:
        raise RuntimeError("所有图像的棋盘检测均失败，无法继续标定")

    return R_board2cam, t_board2cam, valid_paths, error_ids


def load_robot_poses(
    src_dir: Path, error_ids: List[int]
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray], List[HTM]]:
    """
    读取机器人位姿，并按照 error_ids 清洗（去除与失败图像对应的位姿）。
    返回：
    - R_gripper2base
    - t_gripper2base
    - R_base2gripper
    - t_base2gripper
    - robot_poses（清洗后的 HTM 列表）
    """
    pose_file = src_dir / "poses.txt"
    if not pose_file.exists():
        raise FileNotFoundError(f"未找到机器人位姿文件: {pose_file}")

    robot_poses: List[HTM] = read_robot_poses(pose_file)
    print(f"🤖 原始机器人位姿数量: {len(robot_poses)}")

    if error_ids:
        print("去除错误图像对应的位姿数据")
        print(f"错误图像索引: {error_ids}")
        robot_poses = [pose for idx, pose in enumerate(robot_poses) if idx not in error_ids]

    print(f"✅ 清洗后机器人位姿数量: {len(robot_poses)}")

    R_gripper2base = [pose.R for pose in robot_poses]
    t_gripper2base = [pose.P for pose in robot_poses]
    R_base2gripper = [pose.inv.R for pose in robot_poses]
    t_base2gripper = [pose.inv.P for pose in robot_poses]

    return R_gripper2base, t_gripper2base, R_base2gripper, t_base2gripper, robot_poses


def compose_transform(RR: np.ndarray, t: np.ndarray) -> np.ndarray:
    """将 (R, t) 组成 4x4 齐次变换矩阵。"""
    T = np.eye(4, dtype=float)
    T[:3, :3] = RR
    T[:3, 3] = t.reshape(3)
    return T


def calibrate_eye_to_hand(
    src_dir: Path,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[np.ndarray], List[np.ndarray]]:
    """
    眼在手外标定（camera fixed in workspace, 求解 cam→base）。
    返回：
    - results: [(R_cam2base, t_cam2base)]，按 method 顺序
    - R_board2cam, t_board2cam：用于后续 AX=XB 与误差验证
    """
    print("====== 📷 手在眼外标定（eye-to-hand） ======")
    board = create_charuco_board()

    R_board2cam, t_board2cam, image_paths, error_ids = load_board2cam_poses(src_dir, board)
    (
        R_gripper2base,
        t_gripper2base,
        R_base2gripper,
        t_base2gripper,
        robot_poses,
    ) = load_robot_poses(src_dir, error_ids)

    if len(R_base2gripper) != len(R_board2cam):
        print(
            f"❌ 图像数与机器人位姿数不匹配: "
            f"图像数={len(R_board2cam)}, 机器人位姿数={len(R_base2gripper)}"
        )
        raise RuntimeError("数据数量不匹配，无法标定")

    methods = [
        cv2.CALIB_HAND_EYE_TSAI,
        cv2.CALIB_HAND_EYE_PARK,
        cv2.CALIB_HAND_EYE_HORAUD,
        cv2.CALIB_HAND_EYE_DANIILIDIS,
    ]

    results: List[Tuple[np.ndarray, np.ndarray]] = []

    print("\n===== 🤖 手眼标定结果（cam → base）=====")
    np.set_printoptions(suppress=True)

    for method in methods:
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            R_base2gripper,
            t_base2gripper,
            R_board2cam,
            t_board2cam,
            method=method,
        )

        results.append((R_cam2base, t_cam2base))
        T_cam2base = compose_transform(R_cam2base, t_cam2base)

        print(f"\n--- method = {method} ---")
        print("cam_T_base:\n", T_cam2base)

        save_path = src_dir / f"cam_T_base_{method}.npy"
        np.save(save_path, T_cam2base)
        print(f"已保存: {save_path}")

    np.set_printoptions(suppress=False)

    # 比较不同方法之间的差异
    print("\n===== 方法间差异对比 =====")
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            R1, t1 = results[i]
            R2, t2 = results[j]
            rot_diff = np.rad2deg(np.arccos((np.trace(R1.T @ R2) - 1.0) / 2.0))
            trans_diff = np.linalg.norm(t1 - t2) * 1000.0
            print(f"Diff between method {methods[i]} and {methods[j]}:")
            print(f"  Rotation diff (deg): {rot_diff:.6f}")
            print(f"  Translation diff (mm): {trans_diff:.6f}")

    return results, R_board2cam, t_board2cam


def validate_AX_XB_eye_to_hand(
    results: List[Tuple[np.ndarray, np.ndarray]],
    R_base2gripper: List[np.ndarray],
    t_base2gripper: List[np.ndarray],
    R_board2cam: List[np.ndarray],
    t_board2cam: List[np.ndarray],
) -> None:
    """
    使用 AX = XB 关系做简单验证。
    这里 X 取第一个方法的结果（cam→base）。
    """
    if len(R_base2gripper) < 2 or len(R_board2cam) < 2:
        print("样本数量不足，无法进行 AX=XB 验证")
        return

    print("\n===== AX = XB 简单验证（eye-to-hand）=====")

    # 取两组样本
    idx1, idx2 = 5, 4  # 保持与笔记本类似的索引设置
    if idx2 >= len(R_base2gripper) or idx2 >= len(R_board2cam):
        idx1, idx2 = 0, 1

    Tg1 = HTM(R=R_base2gripper[idx1], P=t_base2gripper[idx1])
    Tg2 = HTM(R=R_base2gripper[idx2], P=t_base2gripper[idx2])
    Tt1 = HTM(R=R_board2cam[idx1], P=t_board2cam[idx1])
    Tt2 = HTM(R=R_board2cam[idx2], P=t_board2cam[idx2])

    B = Tt2.matrix @ np.linalg.inv(Tt1.matrix)
    A = np.linalg.inv(Tg2.matrix) @ Tg1.matrix

    X = HTM(R=results[2][0], P=results[2][1]).matrix

    left = A @ X
    right = X @ B

    print("AX:\n", left)
    print("XB:\n", right)
    print("Difference (AX - XB):\n", left - right)


def validate_constant_grip_board(
    src_dir: Path,
    X_cam2base: np.ndarray,
    board,
    robot_poses: List[HTM],
) -> None:
    """
    根据待验证的 X = ^{base}_{cam}T 、相机读出的 ^{cam}_{board}T
    以及机械臂读出的 ^{base}_{grip}T，反推出 ^{grip}_{board}T，
    检查是否近似为常量。
    """
    print("\n===== 标定结果验证：^grip_board T 是否为常量（eye-to-hand）=====")

    src_png_dir = src_dir
    img_files = list(src_png_dir.rglob("*.png"))
    img_files.sort(key=lambda x: x.name)

    if not img_files:
        print(f"在 {src_png_dir} 下找不到 PNG 图片，跳过验证")
        return

    tmp: List[HTM] = []

    for i, img in enumerate(img_files):
        htm_board2cam = get_board2cam_htm_from_pic(img, board)
        if htm_board2cam is None:
            print(f"图像 {img} 未能检测到棋盘，跳过误差计算。")
            continue

        if i >= len(robot_poses):
            print(f"索引 {i} 超出机器人位姿数量范围，停止。")
            break

        htm_cam2base = HTM(X_cam2base)
        htm_grip2base = robot_poses[i]

        # ^grip_board T = ^grip_base T · ^base_cam T · ^cam_board T
        htm_board2grip = htm_grip2base.inv @ htm_cam2base @ htm_board2cam
        np.set_printoptions(precision=5, suppress=True)
        tmp.append(htm_board2grip)
        print(f"反推 grip_T_board 标定板到夹爪结果：\n{htm_board2grip}")

    if len(tmp) < 2:
        print("有效样本不足，无法统计误差。")
        return

    max_translation = max(
        np.linalg.norm(tmp[i].P - tmp[j].P)
        for i in range(len(tmp))
        for j in range(i + 1, len(tmp))
    )
    max_x_translation = max(
        abs(tmp[i].P[0] - tmp[j].P[0])
        for i in range(len(tmp))
        for j in range(i + 1, len(tmp))
    )
    max_y_translation = max(
        abs(tmp[i].P[1] - tmp[j].P[1])
        for i in range(len(tmp))
        for j in range(i + 1, len(tmp))
    )
    max_z_translation = max(
        abs(tmp[i].P[2] - tmp[j].P[2])
        for i in range(len(tmp))
        for j in range(i + 1, len(tmp))
    )

    max_degree = max(
        np.rad2deg(np.arccos((np.trace(tmp[i].R.T @ tmp[j].R) - 1.0) / 2.0))
        for i in range(len(tmp))
        for j in range(i + 1, len(tmp))
    )
    avg_translation = np.mean(
        [
            np.linalg.norm(tmp[i].P - tmp[j].P)
            for i in range(len(tmp))
            for j in range(i + 1, len(tmp))
        ]
    )
    avg_degree = np.mean(
        [
            np.rad2deg(np.arccos((np.trace(tmp[i].R.T @ tmp[j].R) - 1.0) / 2.0))
            for i in range(len(tmp))
            for j in range(i + 1, len(tmp))
        ]
    )

    print(f"最大旋转误差 (deg): {max_degree:.6f}")
    print(f"最大平移误差 (mm): {max_translation * 1000.0:.6f}")
    print(f"最大X轴平移误差 (mm): {max_x_translation * 1000.0:.6f}")
    print(f"最大Y轴平移误差 (mm): {max_y_translation * 1000.0:.6f}")
    print(f"最大Z轴平移误差 (mm): {max_z_translation * 1000.0:.6f}")
    print(f"平均旋转误差 (deg): {avg_degree:.6f}")
    print(f"平均平移误差 (mm): {avg_translation * 1000.0:.6f}")


def main():
    # 默认与笔记本相同的数据目录（可按需修改）
    src_dir = Path("/workspace/dense_fine_harness/calibration/captured_data/exp2")

    # 允许通过命令行参数覆盖
    if len(os.sys.argv) > 1:
        src_dir = Path(os.sys.argv[1])

    if not src_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {src_dir}")

    board = create_charuco_board()
    (
        R_board2cam,
        t_board2cam,
        image_paths,
        error_ids,
    ) = load_board2cam_poses(src_dir, board)
    (
        R_gripper2base,
        t_gripper2base,
        R_base2gripper,
        t_base2gripper,
        robot_poses,
    ) = load_robot_poses(src_dir, error_ids)

    results, _, _ = calibrate_eye_to_hand(src_dir)

    # AX = XB 验证
    validate_AX_XB_eye_to_hand(
        results, R_base2gripper, t_base2gripper, R_board2cam, t_board2cam
    )

    # 使用第一个方法结果做 grip_board 常量验证
    X_cam2base = compose_transform(results[0][0], results[0][1])
    validate_constant_grip_board(src_dir, X_cam2base, board, robot_poses)


if __name__ == "__main__":
    main()


