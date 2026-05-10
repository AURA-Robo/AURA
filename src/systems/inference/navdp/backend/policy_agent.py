from contextlib import nullcontext

import torch
import numpy as np
import cv2
from PIL import Image
from matplotlib import colormaps as cm
from .policy_network import NavDP_Policy
from .tensorrt_acceleration import NavDPTensorRtEngineSpec, configure_navdp_tensorrt

class NavDP_Agent:
    def __init__(self,
                 image_intrinsic,
                 image_size=224,
                 memory_size=8,
                 predict_size=24,
                 temporal_depth=16,
                 heads=8,
                 token_dim=384,
                 navi_model = "./100.ckpt",
                 device='cuda:0',
                 use_amp=False,
                 amp_dtype='float16',
                 enable_tf32=False,
                 tensorrt_mode='off',
                 tensorrt_engine_dir='',
                 tensorrt_precision='fp16'):
        self.image_intrinsic = image_intrinsic
        self.device = device
        self.predict_size = predict_size
        self.image_size = image_size
        self.memory_size = memory_size
        self._device = torch.device(device)
        self._cuda_available = self._device.type == "cuda" and torch.cuda.is_available()
        self._autocast_dtype = self._resolve_amp_dtype(amp_dtype)
        self._use_amp = bool(use_amp and self._cuda_available)
        self._enable_tf32 = bool(enable_tf32 and self._cuda_available)
        self.tensorrt_status = {
            "mode": str(tensorrt_mode or "off"),
            "enabled": False,
            "reason": "not_configured",
        }
        if self._enable_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")
        if self._cuda_available:
            torch.backends.cudnn.benchmark = True
        self.navi_former = NavDP_Policy(image_size,memory_size,predict_size,temporal_depth,heads,token_dim,device)
        self.navi_former.load_state_dict(torch.load(navi_model,map_location=self.device),strict=False)
        self.navi_former.to(self.device)
        self.navi_former.eval()
        self._configure_tensorrt(
            mode=tensorrt_mode,
            engine_dir=tensorrt_engine_dir,
            precision=tensorrt_precision,
            checkpoint_path=navi_model,
            temporal_depth=temporal_depth,
            heads=heads,
            token_dim=token_dim,
        )

    def _configure_tensorrt(
        self,
        *,
        mode,
        engine_dir,
        precision,
        checkpoint_path,
        temporal_depth,
        heads,
        token_dim,
    ):
        normalized_mode = str(mode or "off").strip().lower()
        if normalized_mode == "off":
            self.tensorrt_status = {
                "mode": "off",
                "enabled": False,
                "reason": "disabled",
            }
            return
        if not str(engine_dir or "").strip():
            self.tensorrt_status = {
                "mode": normalized_mode,
                "enabled": False,
                "reason": "missing_engine_dir",
            }
            if normalized_mode in {"required", "build"}:
                raise RuntimeError("NavDP TensorRT mode requires --tensorrt-engine-dir.")
            return
        spec = NavDPTensorRtEngineSpec(
            precision=str(precision or "fp16").strip().lower(),
            batch_size=1,
            sample_num=16,
            memory_size=self.memory_size,
            predict_size=self.predict_size,
            token_dim=int(token_dim),
            temporal_depth=int(temporal_depth),
            heads=int(heads),
        )
        self.tensorrt_status = configure_navdp_tensorrt(
            self.navi_former,
            mode=normalized_mode,
            engine_dir=engine_dir,
            checkpoint_path=checkpoint_path,
            spec=spec,
        )

    @staticmethod
    def _resolve_amp_dtype(amp_dtype):
        value = str(amp_dtype).strip().lower()
        if value == "float16":
            return torch.float16
        if value == "bfloat16":
            return torch.bfloat16
        raise ValueError(f"Unsupported NavDP amp_dtype: {amp_dtype!r}. Expected 'float16' or 'bfloat16'.")

    def _execution_context(self):
        amp_context = nullcontext()
        if self._use_amp:
            amp_context = torch.autocast(device_type="cuda", dtype=self._autocast_dtype)
        return torch.inference_mode(), amp_context

    def _run_model(self, fn, *args):
        inference_context, amp_context = self._execution_context()
        with inference_context:
            with amp_context:
                return fn(*args)

    def _to_device_tensor(self, value):
        if torch.is_tensor(value):
            return value.to(device=self._device, dtype=torch.float32).contiguous()
        return torch.as_tensor(np.ascontiguousarray(value), dtype=torch.float32, device=self._device)

    def _update_memory_queue(self, process_images):
        current_images = self._to_device_tensor(process_images)
        if current_images.shape[0] != len(self.memory_queue):
            raise ValueError(
                f"Batch size mismatch between memory queue ({len(self.memory_queue)}) and images ({current_images.shape[0]})."
            )
        zero_frame = current_images.new_zeros((self.image_size, self.image_size, current_images.shape[-1]))
        input_images = []
        for index, queue in enumerate(self.memory_queue):
            frame = current_images[index]
            if len(queue) >= self.memory_size:
                del queue[0]
            queue.append(frame)
            padded_queue = list(queue)
            if len(padded_queue) < self.memory_size:
                padded_queue = [zero_frame] * (self.memory_size - len(padded_queue)) + padded_queue
            input_images.append(torch.stack(padded_queue, dim=0))
        return torch.stack(input_images, dim=0)

    def _build_external_input_images(self, process_images, history_images):
        current_images = np.asarray(process_images, dtype=np.float32)
        if current_images.ndim != 4:
            raise ValueError(f"Expected processed images with shape [B,H,W,C], got {current_images.shape}.")
        if current_images.shape[0] != 1:
            raise ValueError("history_npz is only supported for NavDP batch_size=1.")

        history_array = np.asarray(history_images)
        if history_array.size == 0:
            processed_history = []
        else:
            if history_array.ndim != 4 or history_array.shape[-1] != 3:
                raise ValueError(f"history_npz rgb_history must have shape [T,H,W,3], got {history_array.shape}.")
            processed_history = list(self.process_image(history_array))

        zero_frame = np.zeros_like(current_images[0], dtype=np.float32)
        recent_history = processed_history[-max(0, self.memory_size - 1) :]
        padded_frames = [zero_frame] * max(0, self.memory_size - 1 - len(recent_history))
        padded_frames.extend(recent_history)
        padded_frames.append(current_images[0])
        return self._to_device_tensor(np.expand_dims(np.stack(padded_frames, axis=0), axis=0))

    def _build_input_images(self, process_images, history_images=None):
        if history_images is None:
            return self._update_memory_queue(process_images)
        return self._build_external_input_images(process_images, history_images)

    def reset(self,batch_size,threshold):
        self.batch_size = batch_size
        self.stop_threshold = threshold
        self.memory_queue = [[] for i in range(batch_size)]
    def reset_env(self,i):
        self.memory_queue[i] = []

    def project_trajectory(self,images,n_trajectories,n_values):
        trajectory_masks = []
        for i in range(images.shape[0]):
            trajectory_mask = np.array(images[i])
            n_trajectory = n_trajectories[i,:,:,0:2]
            n_value = n_values[i]
            for waypoints,value in zip(n_trajectory,n_value):
                norm_value = np.clip(-value*0.1,0,1)
                colormap = cm.get('jet')
                color = np.array(colormap(norm_value)[0:3]) * 255.0
                input_points = np.zeros((waypoints.shape[0],3)) - 0.2
                input_points[:,0:2] = waypoints
                input_points[:,1] = -input_points[:,1]
                camera_z = images[0].shape[0] - 1 - self.image_intrinsic[1][1] * input_points[:,2] / (input_points[:,0] + 1e-8) - self.image_intrinsic[1][2]
                camera_x = self.image_intrinsic[0][0] * input_points[:,1] / (input_points[:,0] + 1e-8) + self.image_intrinsic[0][2]
                for i in range(camera_x.shape[0]-1):
                    try:
                        if camera_x[i] > 0 and camera_z[i] > 0 and camera_x[i+1] > 0 and camera_z[i+1] > 0:
                            trajectory_mask = cv2.line(trajectory_mask,(int(camera_x[i]),int(camera_z[i])),(int(camera_x[i+1]),int(camera_z[i+1])),color.astype(np.uint8).tolist(),5)
                    except:
                        pass
            trajectory_masks.append(trajectory_mask)
        return np.concatenate(trajectory_masks,axis=1)

    def process_image(self,images):
        assert len(images.shape) == 4
        H,W,C = images.shape[1],images.shape[2],images.shape[3]
        prop = self.image_size/max(H,W)
        return_images = []
        for img in images:
            resize_image = cv2.resize(img,(-1,-1),fx=prop,fy=prop)
            pad_width = max((self.image_size - resize_image.shape[1])//2,0)
            pad_height = max((self.image_size - resize_image.shape[0])//2,0)
            pad_image = np.pad(resize_image,((pad_height,pad_height),(pad_width,pad_width),(0,0)),mode='constant',constant_values=0)
            resize_image = cv2.resize(pad_image,(self.image_size,self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_images.append(resize_image)
        return np.array(return_images)

    def process_depth(self,depths):
        assert len(depths.shape) == 4
        depths[depths==np.inf] = 0
        H,W,C = depths.shape[1],depths.shape[2],depths.shape[3]
        prop = self.image_size/max(H,W)
        return_depths = []
        for depth in depths:
            resize_depth = cv2.resize(depth,(-1,-1),fx=prop,fy=prop)
            pad_width = max((self.image_size - resize_depth.shape[1])//2,0)
            pad_height = max((self.image_size - resize_depth.shape[0])//2,0)
            pad_depth = np.pad(resize_depth,((pad_height,pad_height),(pad_width,pad_width)),mode='constant',constant_values=0)
            resize_depth = cv2.resize(pad_depth,(self.image_size,self.image_size))
            resize_depth[resize_depth>5.0] = 0
            resize_depth[resize_depth<0.1] = 0
            return_depths.append(resize_depth[:,:,np.newaxis])
        return np.array(return_depths)

    def process_pixel(self,pixel_coords,input_images):
        return_pixels = []
        H,W,C = input_images.shape[1],input_images.shape[2],input_images.shape[3]
        prop = self.image_size/max(H,W)
        for pixel_coord,input_image in zip(pixel_coords,input_images):
            panel_image = np.zeros_like(input_image,dtype=np.uint8)
            min_x = pixel_coord[0] - 10
            min_y = pixel_coord[1] - 10
            max_x = pixel_coord[0] + 10
            max_y = pixel_coord[1] + 10

            if min_x <= 0:
                panel_image[:,0:10] = 255
            elif min_y <= 0:
                panel_image[0:10,:] = 255
            elif max_x >= panel_image.shape[1]:
                panel_image[:,panel_image.shape[1]-10:] = 255
            elif max_y >= panel_image.shape[0]:
                panel_image[panel_image.shape[0]-10:,:] = 255
            elif min_x > 0 and min_y > 0 and max_x < panel_image.shape[1] and max_y < panel_image.shape[0]:
                panel_image[min_y:max_y,min_x:max_x] = 255

            resize_image = cv2.resize(panel_image,(-1,-1),fx=prop,fy=prop, interpolation=cv2.INTER_NEAREST)
            pad_width = max((self.image_size - resize_image.shape[1])//2,0)
            pad_height = max((self.image_size - resize_image.shape[0])//2,0)
            pad_image = np.pad(resize_image,((pad_height,pad_height),(pad_width,pad_width),(0,0)),mode='constant',constant_values=0)
            resize_image = cv2.resize(pad_image,(self.image_size,self.image_size))
            resize_image = np.array(resize_image)
            resize_image = resize_image.astype(np.float32) / 255.0
            return_pixels.append(resize_image)
        return np.array(return_pixels).mean(axis=-1)

    def process_pointgoal(self,goals):
        clip_goals = goals.clip(-10,10)
        clip_goals[:,0] = np.clip(clip_goals[:,0],0,10)
        return clip_goals

    def step_nogoal(self,images,depths,history_images=None):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_image = self._build_input_images(process_images, history_images=history_images)
        input_depth = self._to_device_tensor(process_depths)
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self._run_model(
            self.navi_former.predict_nogoal_action,
            input_image,
            input_depth,
        )
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values)
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask

    def step_pointgoal(self,goals,images,depths,history_images=None):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_image = self._build_input_images(process_images, history_images=history_images)
        input_depth = self._to_device_tensor(process_depths)
        input_goals = self._to_device_tensor(self.process_pointgoal(goals))
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self._run_model(
            self.navi_former.predict_pointgoal_action,
            input_goals,
            input_image,
            input_depth,
        )
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())

        print(all_values.max(),all_values.min())

        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values)
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask

    def step_imagegoal(self,goals,images,depths,history_images=None):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_image = self._build_input_images(process_images, history_images=history_images)
        input_depth = self._to_device_tensor(process_depths)
        input_goals = self._to_device_tensor(self.process_image(goals))
        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self._run_model(
            self.navi_former.predict_imagegoal_action,
            input_goals,
            input_image,
            input_depth,
        )
        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())

        print(all_values.max(),all_values.min())
        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values)
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask

    def step_pixelgoal(self,goals,images,depths,history_images=None):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_image = self._build_input_images(process_images, history_images=history_images)
        input_depth = self._to_device_tensor(process_depths)
        input_goals = self._to_device_tensor(self.process_pixel(goals,images))

        # cv2.imwrite("input_image.jpg",np.concatenate(self.memory_queue[0],axis=0)*255)
        # cv2.imwrite("pixel_goal.jpg",pixel_vis_image)
        all_trajectory, all_values, good_trajectory, bad_trajectory = self._run_model(
            self.navi_former.predict_pixelgoal_action,
            input_goals,
            input_image,
            input_depth,
        )

        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())

        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values)
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask

    def step_point_image_goal(self,pointgoal,imagegoal,images,depths,history_images=None):
        process_images = self.process_image(images)
        process_depths = self.process_depth(depths)
        input_image = self._build_input_images(process_images, history_images=history_images)
        input_depth = self._to_device_tensor(process_depths)
        input_pointgoal = self._to_device_tensor(self.process_pointgoal(pointgoal))
        input_imagegoal = self._to_device_tensor(self.process_image(imagegoal))

        if history_images is None and self.memory_queue and self.memory_queue[0]:
            cv2.imwrite(
                "input_image.jpg",
                np.concatenate([frame.detach().cpu().numpy() for frame in self.memory_queue[0]], axis=0) * 255,
            )
        all_trajectory, all_values, good_trajectory, bad_trajectory = self._run_model(
            self.navi_former.predict_ip_action,
            input_pointgoal,
            input_imagegoal,
            input_image,
            input_depth,
        )

        if all_values.max() < self.stop_threshold:
            good_trajectory[:,:,:,0] = good_trajectory[:,:,:,0] * 0.0
            good_trajectory[:,:,:,1] = np.sign(good_trajectory[:,:,:,1].mean())

        trajectory_mask = self.project_trajectory(images,all_trajectory,all_values)
        return good_trajectory[:,0], all_trajectory, all_values, trajectory_mask
