import time

import numpy as np
import torch
import torch.multiprocessing as mp

from gaussian_splatting.gaussian_renderer import render
from gaussian_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from gui import gui_utils
from utils.camera_utils import Camera
from utils.eval_utils import eval_ate, save_gaussians
from utils.logging_utils import Log
from utils.multiprocessing_utils import clone_obj
from utils.pose_utils import update_pose
from utils.slam_utils import get_loss_tracking, get_median_depth
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


class FrontEnd(mp.Process):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.background = None
        self.pipeline_params = None
        self.frontend_queue = None
        self.backend_queue = None
        self.q_main2vis = None
        self.q_vis2main = None

        self.initialized = False
        self.kf_indices = []
        self.monocular = config["Training"]["monocular"]
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []

        self.reset = True
        self.requested_init = False
        self.requested_keyframe = 0
        self.use_every_n_frames = 1

        self.orbslam = None
        self.orbslamTimeStampsFile = None
        self.orbslamImgFiles = []
        self.orbslamTimeStamps = []
        self.orbslamDepthFiles = []
        self.orbslamImageScale = None
        self.viewpoint_num = 0

        self.gaussians = None
        self.cameras = dict()
        self.device = "cuda:0"
        self.pause = False

        self.translation_error = []
        self.quaternion_error = []

    def set_hyperparams(self):
        self.save_dir = self.config["Results"]["save_dir"]
        self.save_results = self.config["Results"]["save_results"]
        self.save_trj = self.config["Results"]["save_trj"]
        self.save_trj_kf_intv = self.config["Results"]["save_trj_kf_intv"]

        self.tracking_itr_num = self.config["Training"]["tracking_itr_num"]
        self.kf_interval = self.config["Training"]["kf_interval"]
        self.window_size = self.config["Training"]["window_size"]
        self.single_thread = self.config["Training"]["single_thread"]

    def add_new_keyframe(self, cur_frame_idx, depth=None, opacity=None, init=False):
        rgb_boundary_threshold = self.config["Training"]["rgb_boundary_threshold"]
        self.kf_indices.append(cur_frame_idx)
        viewpoint = self.cameras[cur_frame_idx]
        gt_img = viewpoint.original_image.cuda()
        valid_rgb = (gt_img.sum(dim=0) > rgb_boundary_threshold)[None]
        if self.monocular:
            if depth is None:
                initial_depth = 2 * torch.ones(1, gt_img.shape[1], gt_img.shape[2])
                initial_depth += torch.randn_like(initial_depth) * 0.3
            else:
                depth = depth.detach().clone()
                opacity = opacity.detach()
                use_inv_depth = False
                if use_inv_depth:
                    inv_depth = 1.0 / depth
                    inv_median_depth, inv_std, valid_mask = get_median_depth(
                        inv_depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        inv_depth > inv_median_depth + inv_std,
                        inv_depth < inv_median_depth - inv_std,
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    inv_depth[invalid_depth_mask] = inv_median_depth
                    inv_initial_depth = inv_depth + torch.randn_like(
                        inv_depth
                    ) * torch.where(invalid_depth_mask, inv_std * 0.5, inv_std * 0.2)
                    initial_depth = 1.0 / inv_initial_depth
                else:
                    median_depth, std, valid_mask = get_median_depth(
                        depth, opacity, mask=valid_rgb, return_std=True
                    )
                    invalid_depth_mask = torch.logical_or(
                        depth > median_depth + std, depth < median_depth - std
                    )
                    invalid_depth_mask = torch.logical_or(
                        invalid_depth_mask, ~valid_mask
                    )
                    depth[invalid_depth_mask] = median_depth
                    initial_depth = depth + torch.randn_like(depth) * torch.where(
                        invalid_depth_mask, std * 0.5, std * 0.2
                    )

                initial_depth[~valid_rgb] = 0  # Ignore the invalid rgb pixels
            return initial_depth.cpu().numpy()[0]
        # use the observed depth
        # plot the error in gt depth and observed depth
        if not init:
            depth_error = np.abs(viewpoint.depth - depth.detach().cpu().numpy())
            depth_mask = (depth_error > 0.05)[None]
            initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
            initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
            initial_depth[~depth_mask] = 0
            return initial_depth[0].numpy()
            # plt.grid(False)
            # plt.imshow(np.transpose(depth_mask, (1, 2, 0)))
            # fig, ax = plt.subplots(1, 3)
            # ax[0].imshow(np.transpose(depth_error, (1, 2, 0)))
            # ax[1].imshow(viewpoint.depth)
            # ax[2].imshow(np.transpose(depth.detach().cpu().numpy(), (1, 2, 0)))

        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = not self.monocular
        self.kf_indices = []
        self.iteration_count = 0
        self.occ_aware_visibility = {}
        self.current_window = []
        # remove everything from the queues
        while not self.backend_queue.empty():
            self.backend_queue.get()

        # Initialise the frame at the ground truth pose
        viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        self.kf_indices = []
        depth_map = self.add_new_keyframe(cur_frame_idx, init=True)
        self.request_init(cur_frame_idx, viewpoint, depth_map)
        self.reset = False

    def tracking(self, cur_frame_idx, viewpoint):
        if self.config["Orbslam"]["sensor_type"] == "rgbd":
            img = cv2.imread(viewpoint.color_path, cv2.IMREAD_UNCHANGED)
            imgD = cv2.imread(viewpoint.depth_path, cv2.IMREAD_UNCHANGED)
            currentTimeStamp = self.dataset.timestamps[cur_frame_idx]

            if self.orbslamImageScale != 1.0:
                width = img.cols * self.orbslamImageScale
                height = img.rows * self.orbslamImageScale
                img = cv2.resize(img, (width, height))

            success = self.orbslam.process_image_rgbd(img, imgD, currentTimeStamp)
        elif self.config["Orbslam"]["sensor_type"] == "mono":
            img = cv2.imread(viewpoint.color_path, cv2.IMREAD_UNCHANGED)
            currentTimeStamp = self.dataset.timestamps[cur_frame_idx]

            if self.orbslamImageScale != 1.0:
                width = img.cols * self.orbslamImageScale
                height = img.rows * self.orbslamImageScale
                img = cv2.resize(img, (width, height))

            success = self.orbslam.process_image_mono(img, currentTimeStamp)
        elif self.config["Orbslam"]["sensor_type"] == "mono_inertial":
            img = cv2.imread(viewpoint.color_path, cv2.IMREAD_UNCHANGED)
            currentTimeStamp = float(self.dataset.timestamps[cur_frame_idx] * 1e-9)
            imu_data = self.dataset.imu[cur_frame_idx]

            if self.orbslamImageScale != 1.0:
                width = img.cols * self.orbslamImageScale
                height = img.rows * self.orbslamImageScale
                img = cv2.resize(img, (width, height))

            success = self.orbslam.process_image_mono(img, currentTimeStamp, imu_data)

        if self.orbslam.get_tracking_state() != 2:
            # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
            return
        elif self.orbslam.get_tracking_state() == 2:
            if cur_frame_idx == 0:
                self.viewpoint_num += 1
                return
            self.cameras[cur_frame_idx] = viewpoint
            trajectory = self.orbslam.get_full_trajectory()
            first_frame_rot = self.cameras[0].R_gt
            first_frame_trans = self.cameras[0].T_gt
            T = torch.eye(4)
            T[:3, :3] = first_frame_rot
            T[:3, 3] = first_frame_trans
            # traj_start = time.time()
            current_pose = torch.from_numpy(trajectory[-1])
            current_pose = torch.inverse(current_pose) @ T
            viewpoint.update_RT(current_pose[:3, :3], current_pose[:3, 3])
            # print("Number of poses: ", len(trajectory))
            # print("Number of cameras: ", len(self.cameras))
            # print("------------")
            
            # for (traj, cam) in zip(trajectory, list(self.cameras.values())[1:]):
            #     current_pose = torch.from_numpy(traj)
            #     current_pose = torch.inverse(current_pose) @ T
            #     cam.update_RT(current_pose[:3, :3], current_pose[:3, 3])
            
            # traj_end = time.time()
            # Log("Took ", traj_end - traj_start, " seconds to update viewpoints.")
            # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
                
        else:
            '''This condition needs checking, what to do if orbslam.get_tracking_state() != 2'''
            prev = self.cameras[cur_frame_idx - self.use_every_n_frames]
            viewpoint.update_RT(prev.R, prev.T)

        # self.translation_error.append(torch.norm(viewpoint.T - viewpoint.T_gt).detach().cpu().numpy())
        # quat_est = R.from_matrix(viewpoint.R.detach().cpu().numpy()).as_quat()
        # quat_gt = R.from_matrix(viewpoint.R_gt.detach().cpu().numpy()).as_quat()
        # self.quaternion_error.append(np.linalg.norm(quat_est - quat_gt))

        render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
            
        image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
        
        # plt.grid(False)
        # plt.imshow(image.permute(1, 2, 0).detach().cpu().numpy())
        # Normalize image to 0-1
        # image = image - image.min()
        # image = image / image.max()

        # plt.imsave(f"/root/Datasets/renderedimgs/monotumseq3/image_{cur_frame_idx}.png", image.permute(1, 2, 0).detach().cpu().numpy())
        
        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def is_keyframe(
        self,
        cur_frame_idx,
        last_keyframe_idx,
        cur_frame_visibility_filter,
        occ_aware_visibility,
    ):
        kf_translation = self.config["Training"]["kf_translation"]
        kf_min_translation = self.config["Training"]["kf_min_translation"]
        kf_overlap = self.config["Training"]["kf_overlap"]

        curr_frame = self.cameras[cur_frame_idx]
        last_kf = self.cameras[last_keyframe_idx]
        pose_CW = getWorld2View2(curr_frame.R, curr_frame.T)
        last_kf_CW = getWorld2View2(last_kf.R, last_kf.T)
        last_kf_WC = torch.linalg.inv(last_kf_CW)
        dist = torch.norm((pose_CW @ last_kf_WC)[0:3, 3])
        dist_check = dist > kf_translation * self.median_depth
        dist_check2 = dist > kf_min_translation * self.median_depth

        union = torch.logical_or(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        intersection = torch.logical_and(
            cur_frame_visibility_filter, occ_aware_visibility[last_keyframe_idx]
        ).count_nonzero()
        point_ratio_2 = intersection / union
        return (point_ratio_2 < kf_overlap and dist_check2) or dist_check

    def add_to_window(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        N_dont_touch = 2
        window = [cur_frame_idx] + window
        # remove frames which has little overlap with the current frame
        curr_frame = self.cameras[cur_frame_idx]
        to_remove = []
        removed_frame = None
        for i in range(N_dont_touch, len(window)):
            kf_idx = window[i]
            # szymkiewicz–simpson coefficient
            intersection = torch.logical_and(
                cur_frame_visibility_filter, occ_aware_visibility[kf_idx]
            ).count_nonzero()
            denom = min(
                cur_frame_visibility_filter.count_nonzero(),
                occ_aware_visibility[kf_idx].count_nonzero(),
            )
            point_ratio_2 = intersection / denom
            cut_off = (
                self.config["Training"]["kf_cutoff"]
                if "kf_cutoff" in self.config["Training"]
                else 0.4
            )
            if not self.initialized:
                cut_off = 0.4
            if point_ratio_2 <= cut_off:
                to_remove.append(kf_idx)

        if to_remove:
            window.remove(to_remove[-1])
            removed_frame = to_remove[-1]
        kf_0_WC = torch.linalg.inv(getWorld2View2(curr_frame.R, curr_frame.T))

        if len(window) > self.config["Training"]["window_size"]:
            # we need to find the keyframe to remove...
            inv_dist = []
            for i in range(N_dont_touch, len(window)):
                inv_dists = []
                kf_i_idx = window[i]
                kf_i = self.cameras[kf_i_idx]
                kf_i_CW = getWorld2View2(kf_i.R, kf_i.T)
                for j in range(N_dont_touch, len(window)):
                    if i == j:
                        continue
                    kf_j_idx = window[j]
                    kf_j = self.cameras[kf_j_idx]
                    kf_j_WC = torch.linalg.inv(getWorld2View2(kf_j.R, kf_j.T))
                    T_CiCj = kf_i_CW @ kf_j_WC
                    inv_dists.append(1.0 / (torch.norm(T_CiCj[0:3, 3]) + 1e-6).item())
                T_CiC0 = kf_i_CW @ kf_0_WC
                k = torch.sqrt(torch.norm(T_CiC0[0:3, 3])).item()
                inv_dist.append(k * sum(inv_dists))

            idx = np.argmax(inv_dist)
            removed_frame = window[N_dont_touch + idx]
            window.remove(removed_frame)

        return window, removed_frame
    
    def add_to_window2(
        self, cur_frame_idx, cur_frame_visibility_filter, occ_aware_visibility, window
    ):
        window = [cur_frame_idx] + window[ :self.window_size]
        removed_frame = None

        return window, removed_frame

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap]
        self.backend_queue.put(msg)
        self.requested_keyframe += 1

    def reqeust_mapping(self, cur_frame_idx, viewpoint):
        msg = ["map", cur_frame_idx, viewpoint]
        self.backend_queue.put(msg)

    def request_init(self, cur_frame_idx, viewpoint, depth_map):
        msg = ["init", cur_frame_idx, viewpoint, depth_map]
        self.backend_queue.put(msg)
        self.requested_init = True

    def sync_backend(self, data):
        self.gaussians = data[1]
        occ_aware_visibility = data[2]
        keyframes = data[3]
        self.occ_aware_visibility = occ_aware_visibility

        # for kf_id, kf_R, kf_T in keyframes:
        #     self.cameras[kf_id].update_RT(kf_R.clone(), kf_T.clone())

    def cleanup(self, cur_frame_idx):
        self.cameras[cur_frame_idx].clean()
        if cur_frame_idx % 10 == 0:
            torch.cuda.empty_cache()

    def run(self):
        cur_frame_idx = 0
        self.viewpoint_num = 0
        projection_matrix = getProjectionMatrix2(
            znear=0.01,
            zfar=100.0,
            fx=self.dataset.fx,
            fy=self.dataset.fy,
            cx=self.dataset.cx,
            cy=self.dataset.cy,
            W=self.dataset.width,
            H=self.dataset.height,
        ).transpose(0, 1)
        projection_matrix = projection_matrix.to(device=self.device)
        tic = torch.cuda.Event(enable_timing=True)
        toc = torch.cuda.Event(enable_timing=True)

        while True:
            if self.q_vis2main.empty():
                if self.pause:
                    continue
            else:
                data_vis2main = self.q_vis2main.get()
                self.pause = data_vis2main.flag_pause
                if self.pause:
                    self.backend_queue.put(["pause"])
                    continue
                else:
                    self.backend_queue.put(["unpause"])

            if self.frontend_queue.empty():
                tic.record()
                # if cur_frame_idx > 20:
                #     fig = plt.figure()
                #     ax = fig.add_subplot(111)
                #     ax.set_title("OrbSlam Error")
                #     ax.plot(self.translation_error[:5], color="blue", label="Translation")
                #     ax.plot(self.quaternion_error[:5], color="red", label="Quaternion")
                #     ax.legend()
                #     ax.set_ylim([0, 0.1])
                #     # ax.set_xlim([0, len(self.translation_error)])
                #     fig.show()
                #     fig.waitforbuttonpress()
                #     break

                if cur_frame_idx >= len(self.dataset):
                    if self.save_results:
                        eval_ate(
                            self.cameras,
                            self.kf_indices,
                            self.save_dir,
                            0,
                            final=True,
                            monocular=self.monocular,
                        )
                        save_gaussians(
                            self.gaussians, self.save_dir, "final", final=True
                        )
                    break

                if self.requested_init:
                    time.sleep(0.01)
                    continue

                if self.single_thread and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                if not self.initialized and self.requested_keyframe > 0:
                    time.sleep(0.01)
                    continue

                # time.sleep(0.5)

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                # viewpoint.compute_grad_mask(self.config)

                # '''TODO: Add only those frames which were tracked successfully by ORBSLAM'''
                # self.cameras[cur_frame_idx] = viewpoint

                if cur_frame_idx == 0:
                    self.cameras[cur_frame_idx] = viewpoint

                if self.reset:
                    self.initialize(cur_frame_idx, viewpoint)
                    self.current_window.append(cur_frame_idx)
                    self.tracking(cur_frame_idx, viewpoint)
                    cur_frame_idx += 1
                    continue

                # Tracking
                render_pkg = self.tracking(cur_frame_idx, viewpoint)
                # viewpoint_check = self.cameras[cur_frame_idx]

                if self.orbslam.get_tracking_state() != 2:
                    cur_frame_idx += 1
                    continue

                '''TODO: Add only those frames which were tracked successfully by ORBSLAM'''
                # self.cameras[cur_frame_idx] = viewpoint

                self.initialized = self.initialized or (
                    len(self.current_window) == self.window_size
                )
                # self.initialized = self.orbslam.get_tracking_state() == 2

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        gtcolor=viewpoint.original_image,
                        gtdepth=viewpoint.depth
                        if not self.monocular
                        else np.zeros((viewpoint.image_height, viewpoint.image_width)),
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    continue

                last_keyframe_idx = self.current_window[0]
                check_time = (cur_frame_idx - last_keyframe_idx) >= self.kf_interval
                curr_visibility = (render_pkg["n_touched"] > 0).long()
                # create_kf = self.is_keyframe(
                #     cur_frame_idx,
                #     last_keyframe_idx,
                #     curr_visibility,
                #     self.occ_aware_visibility,
                # )

                create_kf = self.orbslam.is_keyframe()
                # if len(self.current_window) < self.window_size:
                #     union = torch.logical_or(
                #         curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                #     ).count_nonzero()
                #     intersection = torch.logical_and(
                #         curr_visibility, self.occ_aware_visibility[last_keyframe_idx]
                #     ).count_nonzero()
                #     point_ratio = intersection / union
                #     create_kf = (
                #         check_time
                #         and point_ratio < self.config["Training"]["kf_overlap"]
                #     )
                if self.single_thread:
                    create_kf = check_time and create_kf
                if create_kf:
                    self.current_window, removed = self.add_to_window2(
                        cur_frame_idx,
                        curr_visibility,
                        self.occ_aware_visibility,
                        self.current_window,
                    )
                    if self.monocular and not self.initialized and removed is not None:
                        self.reset = True
                        Log(
                            "Keyframes lacks sufficient overlap to initialize the map, resetting."
                        )
                        continue
                    depth_map = self.add_new_keyframe(
                        cur_frame_idx,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        cur_frame_idx, viewpoint, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)
                cur_frame_idx += 1

                if (
                    self.save_results
                    and self.save_trj
                    and create_kf
                    and len(self.kf_indices) % self.save_trj_kf_intv == 0
                ):
                    Log("Evaluating ATE at frame: ", cur_frame_idx)
                    eval_ate(
                        self.cameras,
                        self.kf_indices,
                        self.save_dir,
                        cur_frame_idx,
                        monocular=self.monocular,
                    )
                toc.record()
                torch.cuda.synchronize()
                if create_kf:
                    # throttle at 3fps when keyframe is added
                    duration = tic.elapsed_time(toc)
                    time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
            else:
                data = self.frontend_queue.get()
                if data[0] == "sync_backend":
                    self.sync_backend(data)

                elif data[0] == "keyframe":
                    self.sync_backend(data)
                    self.requested_keyframe -= 1

                elif data[0] == "init":
                    self.sync_backend(data)
                    self.requested_init = False

                elif data[0] == "stop":
                    Log("Frontend Stopped.")
                    break