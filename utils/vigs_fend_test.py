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
        '''After distortion correction not all pixels are valid'''
        # display gt image, depth and valid_rgb
        # plt.imshow(gt_img.permute(1, 2, 0).cpu().numpy())
        # plt.show()
        # plt.figure()
        # plt.imshow(valid_rgb.permute(1, 2, 0).cpu().numpy())
        # plt.show()
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
        initial_depth = torch.from_numpy(viewpoint.depth).unsqueeze(0)
        initial_depth[~valid_rgb.cpu()] = 0  # Ignore the invalid rgb pixels
        return initial_depth[0].numpy()

    def initialize(self, cur_frame_idx, viewpoint):
        self.initialized = self.orbslam.get_tracking_state() == 2
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
        # self.initialized = True

    def tracking(self, cur_frame_idx, viewpoint):
        img = cv2.imread(viewpoint.color_path, cv2.IMREAD_UNCHANGED)
        imgD = cv2.imread(viewpoint.depth_path, cv2.IMREAD_UNCHANGED)
        currentTimeStamp = viewpoint.timestamp

        if self.orbslamImageScale != 1.0:
            width = img.cols * self.orbslamImageScale
            height = img.rows * self.orbslamImageScale
            img = cv2.resize(img, (width, height))

        success = self.orbslam.process_image_rgbd(img, imgD, currentTimeStamp)
        if self.orbslam.get_tracking_state() != 2:
            '''This condition needs checking, what to do if orbslam.get_tracking_state() != 2'''
            viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)
            return
        elif self.orbslam.get_tracking_state() == 2:
            if cur_frame_idx == 0:
                self.viewpoint_num += 1
                return
            '''TODO: Update all previous viewpoint poses'''
            trajectory = self.orbslam.get_full_trajectory()
            current_pose = torch.from_numpy(trajectory[-1])
            first_frame_rot = self.cameras[0].R_gt
            first_frame_trans = self.cameras[0].T_gt
            T = torch.eye(4)
            T[:3, :3] = first_frame_rot
            T[:3, 3] = first_frame_trans
            current_pose = torch.inverse(current_pose) @ T
            viewpoint.update_RT(current_pose[:3, :3], current_pose[:3, 3])
            # viewpoint.update_RT(viewpoint.R_gt, viewpoint.T_gt)

        render_pkg = render(
                    viewpoint, self.gaussians, self.pipeline_params, self.background
                )
            
        image, depth, opacity = (
                render_pkg["render"],
                render_pkg["depth"],
                render_pkg["opacity"],
            )
        
        self.median_depth = get_median_depth(depth, opacity)
        return render_pkg

    def add_to_window(self, cur_frame_idx, window):
        window = [cur_frame_idx] + window
        return window

    def request_keyframe(self, cur_frame_idx, viewpoint, current_window, depthmap):
        msg = ["keyframe", cur_frame_idx, viewpoint, current_window, depthmap, self.cameras]
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

                viewpoint = Camera.init_from_dataset(
                    self.dataset, cur_frame_idx, projection_matrix
                )
                viewpoint.compute_grad_mask(self.config)

                '''TODO: Add only those frames which were tracked successfully by ORBSLAM'''
                self.cameras[cur_frame_idx] = viewpoint

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

                if self.orbslam.get_tracking_state() != 2:
                    cur_frame_idx += 1
                    continue

                # '''TODO: Add only those frames which were tracked successfully by ORBSLAM'''
                # self.cameras[cur_frame_idx] = viewpoint

                self.initialized = self.orbslam.get_tracking_state() == 2

                num_keyframes = self.orbslam.get_num_keyframes()
                if num_keyframes % 4 == 0:
                    self.current_window = []
                    # Get the latest keyFrame in the current map
                    '''Double check what this function returns'''
                    latest_keyframe = self.orbslam.get_latest_keyframe_id()
                    # Get the frame ids corresponding to all the keyframes in the map
                    keyframes_frames_id_mapping = self.orbslam.get_keyframe_ids()

                    # Get the frame id corresponding to the latest keyframe
                    for (keyframe, frame) in keyframes_frames_id_mapping:
                        if keyframe == latest_keyframe:
                            self.frame_to_add = frame
                            break

                    # Get the best co visible frames with the latest keyframe
                    covisible = self.orbslam.get_covisible_frame_ids(latest_keyframe, 5)
                    # Add the latest keyframe to the covisible frames
                    covisible.append(latest_keyframe)
                    # Generate the window for optimization
                    for (keyframe, frame) in keyframes_frames_id_mapping:
                        if keyframe in covisible:
                            self.current_window.append(frame)

                    viewpoint_to_add = self.cameras[self.frame_to_add]
                    depth_map = self.add_new_keyframe(
                        self.frame_to_add,
                        depth=render_pkg["depth"],
                        opacity=render_pkg["opacity"],
                        init=False,
                    )
                    self.request_keyframe(
                        self.frame_to_add, viewpoint_to_add, self.current_window, depth_map
                    )
                else:
                    self.cleanup(cur_frame_idx)

                current_window_dict = {}
                current_window_dict[self.current_window[0]] = self.current_window[1:]
                keyframes = [self.cameras[kf_idx] for kf_idx in self.current_window]

                self.q_main2vis.put(
                    gui_utils.GaussianPacket(
                        gaussians=clone_obj(self.gaussians),
                        current_frame=viewpoint,
                        keyframes=keyframes,
                        kf_window=current_window_dict,
                    )
                )

                if self.requested_keyframe > 0:
                    self.cleanup(cur_frame_idx)
                    cur_frame_idx += 1
                    self.viewpoint_num += 1
                    continue

                if self.single_thread:
                    create_kf = create_kf
                # if create_kf:
                    # self.current_window = self.add_to_window(
                    #     cur_frame_idx,
                    #     self.current_window,
                    # )
                    # if self.monocular and not self.initialized:
                    #     self.reset = True
                    #     Log(
                    #         "Keyframes lacks sufficient overlap to initialize the map, resetting."
                    #     )
                    #     continue
                    # depth_map = self.add_new_keyframe(
                    #     cur_frame_idx,
                    #     depth=render_pkg["depth"],
                    #     opacity=render_pkg["opacity"],
                    #     init=False,
                    # )
                    # self.request_keyframe(
                    #     cur_frame_idx, viewpoint, self.current_window, depth_map
                    # )
                #     pass
                # else:
                #     self.cleanup(cur_frame_idx)
                cur_frame_idx += 1
                self.viewpoint_num += 1

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
                # if create_kf:
                #     # throttle at 3fps when keyframe is added
                #     duration = tic.elapsed_time(toc)
                #     time.sleep(max(0.01, 1.0 / 3.0 - duration / 1000))
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
