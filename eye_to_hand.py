import cv2
import numpy as np
import glob
import os
import sys

# 当前文件路径
FILE_DIR = os.path.dirname(os.path.abspath(__file__))

# 项目根目录（src 的上一级）
PROJECT_ROOT = os.path.abspath(os.path.join(FILE_DIR, ".."))

# 设置为工作路径
os.chdir(PROJECT_ROOT)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


class EyeToHandCalibrator:

    def __init__(self, K, dist, board_cfg):

        self.K = K
        self.dist = dist

        self.R_world2cam = []
        self.t_world2cam = []

        self.R_base2gripper = []
        self.t_base2gripper = []

        self.images = []
        self.vis_data = []

        dictionary = cv2.aruco.getPredefinedDictionary(board_cfg['dict_id'])

        self.board = cv2.aruco.CharucoBoard(
            (board_cfg['squares_x'], board_cfg['squares_y']),
            board_cfg['square_len'],
            board_cfg['marker_len'],
            dictionary
        )

        self.detector = cv2.aruco.CharucoDetector(self.board)

    # ------------------------------------------------------------
    # Euler -> Rotation matrix
    # ------------------------------------------------------------
    def euler_to_R(self, rx, ry, rz):

        Rx = np.array([
            [1,0,0],
            [0,np.cos(rx),-np.sin(rx)],
            [0,np.sin(rx),np.cos(rx)]
        ])

        Ry = np.array([
            [np.cos(ry),0,np.sin(ry)],
            [0,1,0],
            [-np.sin(ry),0,np.cos(ry)]
        ])

        Rz = np.array([
            [np.cos(rz),-np.sin(rz),0],
            [np.sin(rz),np.cos(rz),0],
            [0,0,1]
        ])

        return Rz @ Ry @ Rx


    # ------------------------------------------------------------
    # 检测Charuco标定板
    # ------------------------------------------------------------
    def detect_board_pose(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        charuco_corners, charuco_ids, marker_corners, marker_ids = \
            self.detector.detectBoard(gray)

        if charuco_ids is None or len(charuco_ids) < 4:
            return False, None, None, None

        obj_points = self.board.getChessboardCorners()[charuco_ids]

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            charuco_corners,
            self.K,
            self.dist
        )

        if not success:
            return False, None, None, None

        R, _ = cv2.Rodrigues(rvec)

        return True, R, tvec.reshape(3), (charuco_corners, charuco_ids, rvec, tvec)


    # ------------------------------------------------------------
    # 添加标定样本
    # ------------------------------------------------------------
    def add_sample(self, image, pose_values, rot_type="euler_rad"):

        success, R_target2cam, t_target2cam, vis_data = \
            self.detect_board_pose(image)

        if not success:
            print("Board detection failed")
            return False

        x, y, z, rx, ry, rz = pose_values

        if rot_type == "euler_rad":
            R = self.euler_to_R(rx, ry, rz)
        else:
            R, _ = cv2.Rodrigues(np.array([rx, ry, rz]))

        t = np.array([x, y, z])

        self.R_base2gripper.append(R)
        self.t_base2gripper.append(t)

        self.R_world2cam.append(R_target2cam)
        self.t_world2cam.append(t_target2cam)

        self.images.append(image)
        self.vis_data.append(vis_data)

        return True


    # ------------------------------------------------------------
    # 标定
    # ------------------------------------------------------------
    def calibrate(self):

        print("\nCollected samples:", len(self.R_world2cam))

        if len(self.R_world2cam) < 3:
            print("样本不足，至少需要3组数据")
            sys.exit()

        R_base2cam, t_base2cam, R_gripper2world, t_gripper2world = \
            cv2.calibrateRobotWorldHandEye(
                self.R_world2cam,
                self.t_world2cam,
                self.R_base2gripper,
                self.t_base2gripper,
                method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH
            )

        print("\n===== Calibration Result =====")

        print("\nR_base2cam =")
        print(R_base2cam)

        print("\nt_base2cam =")
        print(t_base2cam)

        self.R_base2cam = R_base2cam
        self.t_base2cam = t_base2cam

        return R_base2cam, t_base2cam


    # ------------------------------------------------------------
    # 计算误差
    # ------------------------------------------------------------
    def compute_error(self):

        errors = []

        for i in range(len(self.R_world2cam)):

            R_wc = self.R_world2cam[i]
            t_wc = self.t_world2cam[i]

            R_bg = self.R_base2gripper[i]
            t_bg = self.t_base2gripper[i]

            R_bc = self.R_base2cam
            t_bc = self.t_base2cam

            R_pred = R_bc @ R_bg @ R_wc
            t_pred = R_bc @ (R_bg @ t_wc + t_bg) + t_bc

            error = np.linalg.norm(t_pred - t_wc)

            errors.append(error)

        print("\n===== HandEye Error =====")
        print("Mean error:", np.mean(errors))
        print("Max error:", np.max(errors))


    # ------------------------------------------------------------
    # 重投影误差 + 可视化
    # ------------------------------------------------------------
    def compute_reprojection(self, save_dir="reprojection_vis"):

        os.makedirs(save_dir, exist_ok=True)

        errors = []

        for i in range(len(self.images)):

            image = self.images[i].copy()

            charuco_corners, charuco_ids, rvec, tvec = self.vis_data[i]

            obj_points = self.board.getChessboardCorners()[charuco_ids]

            proj_points, _ = cv2.projectPoints(
                obj_points,
                rvec,
                tvec,
                self.K,
                self.dist
            )

            proj_points = proj_points.reshape(-1,2)
            img_points = charuco_corners.reshape(-1,2)

            err = np.linalg.norm(img_points - proj_points, axis=1)
            mean_err = np.mean(err)

            errors.append(mean_err)

            # 画检测角点
            # for p in img_points:
            #     cv2.circle(image, tuple(np.int32(p)), 4, (0,255,0), -1)

            # 画重投影角点
            for p in proj_points:
                cv2.circle(image, tuple(np.int32(p)), 4, (0,0,255), -1)

            save_path = os.path.join(save_dir, f"reproj_{i:03d}.png")

            cv2.imwrite(save_path, image)

        print("\n===== Reprojection Error =====")
        print("Mean:", np.mean(errors))
        print("Max :", np.max(errors))
        print("Min :", np.min(errors))

        np.savetxt(
            os.path.join(save_dir, "reprojection_error.txt"),
            np.array(errors),
            fmt="%.6f"
        )


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    # 相机内参
    K = np.array([
        [2414.1939545640535, 0, 967.3404409524326],
        [0, 2414.4741049405593, 618.9615768349123],
        [0, 0, 1]
    ])

    dist = np.zeros((5, 1))


    # Charuco板配置
    charucoboard_cfg = {

        'type': 'charuco',

        'squares_x': 7,
        'squares_y': 5,

        'square_len': 0.035,
        'marker_len': 0.026,

        'dict_id': cv2.aruco.DICT_4X4_50
    }


    calibrator = EyeToHandCalibrator(K, dist, charucoboard_cfg)


    # 数据路径
    image_files = sorted(
        glob.glob("calibration/captured_data/exp1/*.png")
    )

    pose_file = "calibration/captured_data/exp1/poses.txt"


    if len(image_files) == 0 or not os.path.exists(pose_file):
        print("未找到图片或poses.txt")
        sys.exit()


    with open(pose_file, 'r') as f:
        lines = f.readlines()


    # 添加样本
    for i, img_path in enumerate(image_files):

        img = cv2.imread(img_path)

        pose_values = [
            float(x)
            for x in lines[i].strip().split(',')
        ]

        calibrator.add_sample(
            img,
            pose_values,
            rot_type='euler_rad'
        )


    # 执行标定
    R, t = calibrator.calibrate()

    # 计算手眼误差
    calibrator.compute_error()

    # 计算重投影误差
    calibrator.compute_reprojection(save_dir="calibration/captured_data/exp1/reprojection_vis")
