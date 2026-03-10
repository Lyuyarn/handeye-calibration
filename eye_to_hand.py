import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import glob

# ==========================================
# 1. 辅助工具函数
# ==========================================

def pose_to_matrix(pose_list, rot_type='euler_rad'):
    """
    将位姿列表转换为4x4齐次变换矩阵
    pose_list: [x, y, z, rx, ry, rz]
    rot_type: 
        'euler_rad': 欧拉角，弧度制 (常用)
        'euler_deg': 欧拉角，角度制
        'axis_angle': 轴角
    """
    x, y, z, rx, ry, rz = pose_list
    
    # 处理旋转部分
    if rot_type == 'euler_rad':
        # 假设欧拉角顺序为ZYX (即RPY)，若不同请修改 'ZYX'
        rot = Rotation.from_euler('ZYX', [rz, ry, rx], degrees=False)
    elif rot_type == 'euler_deg':
        rot = Rotation.from_euler('ZYX', [rz, ry, rx], degrees=True)
    elif rot_type == 'axis_angle':
        rot = Rotation.from_rotvec([rx, ry, rz])
    else:
        raise ValueError("不支持的旋转类型")

    R = rot.as_matrix()
    
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def inverse_matrix(T):
    """计算齐次变换矩阵的逆"""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv

def detect_board_pose(image, camera_matrix, dist_coeffs, board_config):
    """检测标定板并返回相对于相机的位姿 (R, t)"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    board_type = board_config.get('type', 'chess')
    
    if board_type == 'chess':
        rows = board_config['rows']
        cols = board_config['cols']
        square_size = board_config['square_size']
        
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
        
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            if success:
                R, _ = cv2.Rodrigues(rvec)
                return True, R, tvec
                
    elif board_type == 'charuco':
        squares_x = board_config['squares_x']
        squares_y = board_config['squares_y']
        square_len = board_config['square_len']
        marker_len = board_config['marker_len']
        dict_id = board_config['dict_id']
        
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_id)
        board = cv2.aruco.CharucoBoard_create(squares_x, squares_y, square_len, marker_len, aruco_dict)
        
        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict)
        if len(corners) > 0:
            ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board)
            if ret > 4:
                success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, board, camera_matrix, dist_coeffs)
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    return True, R, tvec

    return False, None, None

# ==========================================
# 2. 眼在手外标定核心类
# ==========================================

class EyeToHandCalibrator:
    def __init__(self, camera_matrix, dist_coeffs, board_config):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.board_config = board_config
        
        # 原始数据
        self.T_gripper2base_list = [] # 机械臂位姿
        self.T_target2cam_list = []   # 标定板相对于相机的位姿
        
        # 标定结果
        self.R_cam2base = None
        self.t_cam2base = None

    def add_sample(self, image, robot_pose, rot_type='euler_rad'):
        """
        添加一组采样数据
        image: 图片
        robot_pose: 机械臂末端在base下的坐标 [x, y, z, rx, ry, rz]
        """
        # 1. 解析机械臂位姿 -> T_gripper2base
        T_gripper2base = pose_to_matrix(robot_pose, rot_type)
        
        # 2. 检测标定板 -> T_target2cam
        success, R_target2cam, t_target2cam = detect_board_pose(image, self.camera_matrix, self.dist_coeffs, self.board_config)
        
        if success:
            T_target2cam = np.eye(4)
            T_target2cam[:3, :3] = R_target2cam
            T_target2cam[:3, 3] = t_target2cam.flatten()
            
            self.T_gripper2base_list.append(T_gripper2base)
            self.T_target2cam_list.append(T_target2cam)
            print(f"样本添加成功，当前数量: {len(self.T_gripper2base_list)}")
            return True
        else:
            print("样本添加失败: 未检测到标定板")
            return False

    def calibrate(self, method=cv2.CALIB_HAND_EYE_TSAI):
        """
        执行标定
        眼在手外公式推导：
        我们要解的是 X (cam2base)
        原方程: T_base2gripper * T_target2gripper = T_cam2base * T_target2cam
        取逆变换: (T_target2cam)^-1 * T_cam2base = (T_target2gripper)^-1 * T_base2gripper
        令 A = (T_target2cam)^-1, B = (T_base2gripper)^-1
        方程变为: A * X = Y * B
        这符合 OpenCV calibrateHandEye 的 AX=XB 形式 (或 AX=YB，视算法而定)
        OpenCV 输出的 X 此时为 T_cam2base (或其逆，需验证)
        
        实际上，OpenCV的 calibrateHandEye(R_gripper2base, t_gripper2base...) 接口 
        默认求解的是眼在手上的 X (gripper2cam)。
        
        为了在眼在手外场景复用该接口，我们需要转换数据：
        设输入为 R_g2b_input = inv(R_target2cam), t_g2b_input = -inv(R_target2cam)*t_target2cam
        设输入为 R_t2c_input = inv(R_gripper2base), t_t2c_input = -inv(R_gripper2base)*t_gripper2base
        调用 cv2.calibrateHandEye(R_g2b_input, ..., R_t2c_input, ...)
        得到的输出 R_cam2gripper_output 实际上是 R_base2cam (base在相机下的姿态)
        取逆即可得到 R_cam2base。
        """
        if len(self.T_gripper2base_list) < 3:
            raise ValueError("样本数量不足，至少需要3组")

        R_g2b_input_list = []
        t_g2b_input_list = []
        R_t2c_input_list = []
        t_t2c_input_list = []

        for T_gb, T_tc in zip(self.T_gripper2base_list, self.T_target2cam_list):
            # 构造输入数据 1: Cam -> Target 的逆
            T_ct = inverse_matrix(T_tc)
            R_g2b_input_list.append(T_ct[:3, :3])
            t_g2b_input_list.append(T_ct[:3, 3].reshape(3, 1))
            
            # 构造输入数据 2: Base -> Gripper 的逆
            T_bg = inverse_matrix(T_gb)
            R_t2c_input_list.append(T_bg[:3, :3])
            t_t2c_input_list.append(T_bg[:3, 3].reshape(3, 1))

        # 调用 OpenCV 标定
        R_base2cam, t_base2cam = cv2.calibrateHandEye(
            R_g2b_input_list, t_g2b_input_list,
            R_t2c_input_list, t_t2c_input_list,
            method=method
        )
        
        # 结果转换
        # OpenCV输出的是方程 A*X = Y*B 中的 X，在此构造下对应 T_base2cam
        # 我们通常需要 T_cam2base (相机在基座下的位置)
        T_base2cam = np.eye(4)
        T_base2cam[:3, :3] = R_base2cam
        T_base2cam[:3, 3] = t_base2cam.flatten()
        
        T_cam2base = inverse_matrix(T_base2cam)
        
        self.R_cam2base = T_cam2base[:3, :3]
        self.t_cam2base = T_cam2base[:3, 3].reshape(3, 1)
        
        print("\n===== 标定结果 =====")
        print("R_cam2base (旋转矩阵):\n", self.R_cam2base)
        print("t_cam2base (平移向量):\n", self.t_cam2base)
        
        r = Rotation.from_matrix(self.R_cam2base)
        print("相机在基座下的位置: ", self.t_cam2base.flatten())
        print("相机姿态: ", r.as_euler('xyz', degrees=True))
        
        return self.R_cam2base, self.t_cam2base

    def compute_reprojection_error(self):
        """
        计算重定位误差。
        原理：标定板刚性连接在机械臂末端，因此标定板在末端坐标系下的位置应该是固定的。
        误差 = 计算出的T_target2gripper的位置波动。
        """
        if self.R_cam2base is None:
            raise RuntimeError("请先运行 calibrate()")

        points_in_gripper = []
        
        T_cam2base = np.eye(4)
        T_cam2base[:3, :3] = self.R_cam2base
        T_cam2base[:3, 3] = self.t_cam2base.flatten()

        for T_gb, T_tc in zip(self.T_gripper2base_list, self.T_target2cam_list):
            # 链式法则: Target -> Cam -> Base -> Gripper
            # T_target2gripper = inv(T_gripper2base) * T_cam2base * T_target2cam
            T_base2gripper = inverse_matrix(T_gb)
            T_target2gripper = T_base2gripper @ T_cam2base @ T_tc
            
            points_in_gripper.append(T_target2gripper[:3, 3])
            
        points_in_gripper = np.array(points_in_gripper)
        
        mean_point = np.mean(points_in_gripper, axis=0)
        errors = np.linalg.norm(points_in_gripper - mean_point, axis=1)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        
        print(f"\n===== 重定位误差分析 =====")
        print(f"标定板相对于末端的理论中心位置: {mean_point}")
        print(f"平均误差: {mean_error*1000:.3f} mm")
        print(f"误差标准差: {std_error*1000:.3f} mm")
        
        self.visualize_errors(points_in_gripper)
        return mean_error, std_error

    def visualize_errors(self, points):
        """可视化标定板原点在末端坐标系下的分布（应该重合）"""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', label='Board Pos in Gripper Frame')
        center = np.mean(points, axis=0)
        ax.scatter(center[0], center[1], center[2], c='b', marker='x', s=100, label='Mean')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Reprojection Consistency Check\n(Board Position in Gripper Frame)')
        ax.legend()
        plt.show()

# ==========================================
# 3. 应用示例：坐标转换
# ==========================================

def transform_cam_to_base(cam_point, R_cam2base, t_cam2base):
    """
    将相机坐标系下的点转换到基座坐标系
    cam_point: [x, y, z] in camera frame
    """
    p_cam = np.array(cam_point).reshape(3, 1)
    p_base = R_cam2base @ p_cam + t_cam2base
    return p_base.flatten()

# ==========================================
# 4. 主程序模拟运行
# ==========================================

if __name__ == "__main__":
    # --- A. 参数设置 ---
    # 相机内参 (需替换)
    K = np.array([[600, 0, 320], 
                  [0, 600, 240], 
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1))
    
    # 标定板参数
    board_cfg = {
        'type': 'chess',
        'rows': 6,
        'cols': 9,
        'square_size': 0.025
    }
    
    calibrator = EyeToHandCalibrator(K, dist, board_cfg)
    
    # --- B. 读取数据 ---
    image_files = sorted(glob.glob("images/*.png"))
    pose_file = "poses.txt" # 假设每行是一组位姿

    with open(pose_file, 'r') as f:
        lines = f.readlines()
        
    for i, img_path in enumerate(image_files):
        img = cv2.imread(img_path)
        pose_values = [float(x) for x in lines[i].strip().split(',')] # 根据文件格式调整
        calibrator.add_sample(img, pose_values, rot_type='euler_rad')

    # --- C. 执行标定 ---
    try:
        R_result, t_result = calibrator.calibrate(method=cv2.CALIB_HAND_EYE_TSAI)
        
        # --- D. 误差分析 ---
        calibrator.compute_reprojection_error()
        
        # --- E. 应用示例 ---
        print("\n===== 使用示例 =====")
        # 假设相机识别到一个物体，在相机坐标系下坐标为 [0.1, 0.05, 0.6]
        obj_cam = [0.1, 0.05, 0.6]
        obj_base = transform_cam_to_base(obj_cam, R_result, t_result)
        print(f"物体在相机坐标系下: {obj_cam}")
        print(f"物体在基座坐标系下: {obj_base}")
        print("此时机械臂可根据此坐标进行抓取。")
        
    except Exception as e:
        print(f"发生错误: {e}")