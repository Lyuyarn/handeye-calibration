import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import glob

# ==========================================
# 1. 辅助工具函数
# ==========================================

def parse_robot_pose(pose_list, rot_type='euler_rad'):
    """
    将机器人位姿列表转换为4x4齐次变换矩阵.
    pose_list: [x, y, z, rx, ry, rz]
    rot_type: 
        'euler_rad': 欧拉角，弧度制 (常见于如Universal Robots等机器人)
        'euler_deg': 欧拉角，角度制
        'axis_angle': 轴角
    """
    x, y, z, rx, ry, rz = pose_list
    
    # 处理旋转部分
    if rot_type == 'euler_rad':
        # 假设欧拉角顺序为ZYX (即RPY: Roll, Pitch, Yaw)，这在工业机器人中很常见
        # 注意：不同品牌机器人欧拉角定义不同，如果是ABB可能是ZYX，如果是KUKA可能是ZXZ
        # 这里使用scipy的 'zyx' 顺序对应通常的 RPY (先绕Z转，再绕Y，最后绕X)
        rot = Rotation.from_euler('ZYX', [rz, ry, rx], degrees=False)
    elif rot_type == 'euler_deg':
        rot = Rotation.from_euler('ZYX', [rz, ry, rx], degrees=True)
    elif rot_type == 'axis_angle':
        # 某些机器人直接输出旋转向量
        rot = Rotation.from_rotvec([rx, ry, rz])
    else:
        raise ValueError("不支持的旋转类型")

    R = rot.as_matrix()
    
    # 构建矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

def detect_board_pose(image, camera_matrix, dist_coeffs, board_config):
    """
    检测标定板并返回相对于相机的位姿
    board_config: 包含标定板参数的字典
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    board_type = board_config.get('type', 'chess')
    
    # 1. 棋盘格检测
    if board_type == 'chess':
        rows = board_config['rows']      # 内角点行数
        cols = board_config['cols']      # 内角点列数
        square_size = board_config['square_size'] # 格子边长 (米)
        
        # 生成物体坐标系下的3D点
        objp = np.zeros((rows * cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
        
        ret, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if ret:
            # 亚像素优化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            # PnP求解位姿
            success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            if success:
                R, _ = cv2.Rodrigues(rvec)
                return True, R, tvec
                
    # 2. ArUco/Charuco 检测
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
# 2. 标定核心类
# ==========================================

class EyeInHandCalibrator:
    def __init__(self, camera_matrix, dist_coeffs, board_config):
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.board_config = board_config
        
        # 存储数据
        self.R_gripper2base_list = []
        self.t_gripper2base_list = []
        self.R_target2cam_list = []
        self.t_target2cam_list = []
        
        self.R_cam2gripper = None
        self.t_cam2gripper = None

    def add_sample(self, image, robot_pose, rot_type='euler_rad'):
        """
        添加一组采样数据
        image: 图片 (BGR)
        robot_pose: [x, y, z, rx, ry, rz]
        """
        # 1. 获取机器人位姿 (Base -> Gripper)
        T_gripper2base = parse_robot_pose(robot_pose, rot_type)
        R_gripper2base = T_gripper2base[:3, :3]
        t_gripper2base = T_gripper2base[:3, 3].reshape(3, 1)
        
        # 2. 获取标定板相对于相机的位姿
        success, R_target2cam, t_target2cam = detect_board_pose(image, self.camera_matrix, self.dist_coeffs, self.board_config)
        
        if success:
            self.R_gripper2base_list.append(R_gripper2base)
            self.t_gripper2base_list.append(t_gripper2base)
            self.R_target2cam_list.append(R_target2cam)
            self.t_target2cam_list.append(t_target2cam)
            print(f"样本添加成功，当前数量: {len(self.R_gripper2base_list)}")
            return True
        else:
            print("样本添加失败: 未检测到标定板")
            return False

    def calibrate(self, method=cv2.CALIB_HAND_EYE_TSAI):
        """
        执行标定
        method: 
            cv2.CALIB_HAND_EYE_TSAI (最常用，经典方法)
            cv2.CALIB_HAND_EYE_PARK
            cv2.CALIB_HAND_EYE_HORAUD
            cv2.CALIB_HAND_EYE_ANDREFF
        """
        if len(self.R_gripper2base_list) < 3:
            raise ValueError("样本数量不足，至少需要3组")

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            self.R_gripper2base_list, 
            self.t_gripper2base_list, 
            self.R_target2cam_list, 
            self.t_target2cam_list,
            method=method
        )
        
        self.R_cam2gripper = R_cam2gripper
        self.t_cam2gripper = t_cam2gripper
        
        print("\n===== 标定结果 =====")
        print("R_cam2gripper (旋转矩阵):\n", self.R_cam2gripper)
        print("t_cam2gripper (平移向量):\n", self.t_cam2gripper)
        
        # 计算欧拉角方便查看
        r = Rotation.from_matrix(self.R_cam2gripper)
        euler_deg = r.as_euler('xyz', degrees=True)
        print("旋转 (XYZ欧拉角, 度): ", euler_deg)
        
        return self.R_cam2gripper, self.t_cam2gripper

    def compute_reprojection_error(self):
        """
        计算重定位误差。
        原理：眼在手标定中，标定板相对于基座是静止的。
        我们计算每一帧推算出的标定板在基座下的位置，看其离散程度。
        """
        if self.R_cam2gripper is None:
            raise RuntimeError("请先运行 calibrate()")

        points_in_base = []
        
        # 构建相机到末端的变换矩阵
        T_cam2gripper = np.eye(4)
        T_cam2gripper[:3, :3] = self.R_cam2gripper
        T_cam2gripper[:3, 3] = self.t_cam2gripper.flatten()

        for i in range(len(self.R_gripper2base_list)):
            # T_base2gripper
            T_base2gripper = np.eye(4)
            T_base2gripper[:3, :3] = self.R_gripper2base_list[i]
            T_base2gripper[:3, 3] = self.t_gripper2base_list[i].flatten()
            
            # T_cam2target
            T_cam2target = np.eye(4)
            T_cam2target[:3, :3] = self.R_target2cam_list[i]
            T_cam2target[:3, 3] = self.t_target2cam_list[i].flatten()
            
            # 链式法则: Base -> Gripper -> Cam -> Target
            # T_target_in_base = T_base2gripper * T_cam2gripper * T_cam2target
            T_base2target = T_base2gripper @ T_cam2gripper @ T_cam2target
            
            points_in_base.append(T_base2target[:3, 3])
            
        points_in_base = np.array(points_in_base)
        
        # 计算均值
        mean_point = np.mean(points_in_base, axis=0)
        
        # 计算每个点距离均值的距离
        errors = np.linalg.norm(points_in_base - mean_point, axis=1)
        mean_error = np.mean(errors)
        std_error = np.std(errors)
        max_error = np.max(errors)
        
        print("\n===== 重定位误差分析 =====")
        print(f"计算出的标定板在基座下的平均位置: {mean_point}")
        print(f"平均误差: {mean_error*1000:.3f} mm")
        print(f"误差标准差: {std_error*1000:.3f} mm")
        print(f"最大误差: {max_error*1000:.3f} mm")
        
        # 可视化
        self.visualize_errors(points_in_base)
        
        return mean_error, std_error

    def visualize_errors(self, points):
        """绘制标定板原点在基座坐标系下的分布"""
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制散点
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c='r', marker='o', label='Calculated Board Pos')
        
        # 绘制中心点
        center = np.mean(points, axis=0)
        ax.scatter(center[0], center[1], center[2], c='b', marker='x', s=100, label='Mean Position')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('Reprojection Consistency Check\n(Board Position in Base Frame)')
        ax.legend()
        plt.show()

# ==========================================
# 3. 应用示例：坐标转换
# ==========================================

def transform_pixel_to_base(u, v, depth, R_cam2gripper, t_cam2gripper, robot_pose, camera_matrix, rot_type='euler_rad'):
    """
    将图像像素坐标和深度转换为机械臂基座坐标
    u, v: 像素坐标
    depth: 深度值 (米)
    robot_pose: 拍摄该图时的机械臂位姿 [x,y,z,rx,ry,rz]
    """
    # 1. 像素坐标 -> 相机坐标系
    # 公式: x = (u - cx) * z / fx; y = (v - cy) * z / fy; z = depth
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    
    x_cam = (u - cx) * depth / fx
    y_cam = (v - cy) * depth / fy
    z_cam = depth
    point_cam = np.array([x_cam, y_cam, z_cam])
    
    # 2. 相机坐标系 -> 末端坐标系
    # P_gripper = R * P_cam + t
    point_gripper = R_cam2gripper @ point_cam + t_cam2gripper.flatten()
    
    # 3. 末端坐标系 -> 基座坐标系
    # 需要当前的 Base -> Gripper 变换
    T_gripper2base = parse_robot_pose(robot_pose, rot_type)
    R_gripper2base = T_gripper2base[:3, :3]
    t_gripper2base = T_gripper2base[:3, 3]
    
    # P_base = R * P_gripper + t
    point_base = R_gripper2base @ point_gripper + t_gripper2base
    
    return point_base

# ==========================================
# 4. 主程序模拟运行 (使用模拟数据)
# ==========================================

if __name__ == "__main__":
    # --- A. 参数设置 ---
    # 相机内参 (需根据实际相机填写)
    K = np.array([[600, 0, 320], 
                  [0, 600, 240], 
                  [0, 0, 1]], dtype=np.float64)
    dist = np.zeros((5, 1)) # 假设无畸变
    
    # 标定板参数
    board_cfg = {
        'type': 'chess',
        'rows': 6,      # 内角点数
        'cols': 9,
        'square_size': 0.02 # 米
    }
    
    calibrator = EyeInHandCalibrator(K, dist, board_cfg)
    
    # 实际使用代码示例
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
        
        # --- D. 计算误差 ---
        calibrator.compute_reprojection_error()
        
        # --- E. 使用示例 ---
        print("\n===== 使用示例 =====")
        # 假设当前机械臂位姿
        current_robot_pose = [0.45, 0.05, 0.6, -3.1, 0.5, -3.0]
        # 假设识别到图像中心点 (320, 240)，深度为 0.5米
        pixel_u, pixel_v = 320, 240
        depth_val = 0.5
        
        base_coord = transform_pixel_to_base(
            pixel_u, pixel_v, depth_val, 
            R_result, t_result, 
            current_robot_pose, 
            K, 
            rot_type='euler_rad'
        )
        
        print(f"像素 ({pixel_u}, {pixel_v}), 深度 {depth_val}m -> 基座坐标: {base_coord}")
        
    except Exception as e:
        print(f"发生错误: {e}")