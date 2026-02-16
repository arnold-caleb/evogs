import os
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from scene.dataset import FourDGSdataset
from scene.dataset_readers import add_points

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_coarse=False):
        """
        :param path: Path to DyNeRF scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians
        
        # Helper function to get the actual model (handles DataParallel)
        def get_gaussians_model():
            return self.gaussians.module if hasattr(self.gaussians, 'module') else self.gaussians
        self.get_gaussians_model = get_gaussians_model

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        # Load dataset (dynerf or dynerf_static)
        dataset_type = getattr(args, 'dataset_type', 'dynerf')  # Default to dynerf if not specified
        
        # Pass frame_idx for static dataset
        if dataset_type == 'dynerf_static':
            frame_idx = getattr(args, 'frame_idx', 0)
            print(f"[SCENE INIT] Loading static dataset with frame_idx={frame_idx}")
            scene_info = sceneLoadTypeCallbacks[dataset_type](args.source_path, args.white_background, args.eval, frame_idx=frame_idx)
        else:
            scene_info = sceneLoadTypeCallbacks[dataset_type](args.source_path, args.white_background, args.eval)
        
        def _extract_sorted_times(data_source):
            times = set()
            if data_source is None:
                return []
            if hasattr(data_source, "image_times"):
                try:
                    return sorted(float(t) for t in data_source.image_times)
                except TypeError:
                    pass
            if isinstance(data_source, (list, tuple)):
                iterable = data_source
            else:
                try:
                    length = len(data_source)
                except TypeError:
                    length = 0
                iterable = (data_source[idx] for idx in range(length)) if length else []
            for entry in iterable:
                time_val = None
                if hasattr(entry, "time"):
                    time_val = getattr(entry, "time")
                elif isinstance(entry, (list, tuple)) and len(entry) >= 3:
                    time_val = entry[2]
                elif isinstance(entry, dict) and "time" in entry:
                    time_val = entry["time"]
                if time_val is None:
                    continue
                times.add(float(time_val))
            return sorted(times)

        self.train_times = _extract_sorted_times(scene_info.train_cameras)
        self.num_time_steps = len(self.train_times)
        self.dataset_type = dataset_type
        self.maxtime = scene_info.maxtime
        self.cameras_extent = scene_info.nerf_normalization["radius"]
        
        print("Loading Training Cameras")
        self.train_camera = FourDGSdataset(scene_info.train_cameras, args, self.dataset_type)
        print("Loading Test Cameras")
        self.test_camera = FourDGSdataset(scene_info.test_cameras, args, self.dataset_type)
        print("Loading Video Cameras")
        self.video_camera = FourDGSdataset(scene_info.video_cameras, args, self.dataset_type)
        
        # Set up deformation network bounds
        xyz_max = scene_info.point_cloud.points.max(axis=0)
        xyz_min = scene_info.point_cloud.points.min(axis=0)
        
        if args.add_points:
            print("Adding points to point cloud")
            scene_info = scene_info._replace(point_cloud=add_points(scene_info.point_cloud, xyz_max=xyz_max, xyz_min=xyz_min))
        
        gaussians_model = self.get_gaussians_model()
        gaussians_model._deformation.deformation_net.set_aabb(xyz_max, xyz_min)
        
        # Set AABB for dynamics model if it exists
        if hasattr(gaussians_model, 'dynamics_model') and gaussians_model.dynamics_model is not None:
            if hasattr(gaussians_model.dynamics_model, 'set_aabb'):
                gaussians_model.dynamics_model.set_aabb(xyz_max, xyz_min)
                print(f"Set AABB for dynamics model: max={xyz_max}, min={xyz_min}")
        
        # Load or create Gaussians
        if self.loaded_iter:
            gaussians_model.load_ply(os.path.join(self.model_path, "point_cloud", f"iteration_{self.loaded_iter}", "point_cloud.ply"))
            gaussians_model.load_model(os.path.join(self.model_path, "point_cloud", f"iteration_{self.loaded_iter}"))
        else:
            gaussians_model.create_from_pcd(scene_info.point_cloud, self.cameras_extent, self.maxtime)

    def save(self, iteration, stage=None):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, f"point_cloud/coarse_iteration_{iteration}")
        else:
            point_cloud_path = os.path.join(self.model_path, f"point_cloud/iteration_{iteration}")
        
        gaussians_model = self.get_gaussians_model()
        gaussians_model.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        gaussians_model.save_deformation(point_cloud_path)

    def getTrainCameras(self, scale=1.0):
        return self.train_camera

    def getTestCameras(self, scale=1.0):
        return self.test_camera
        
    def getVideoCameras(self, scale=1.0):
        return self.video_camera
