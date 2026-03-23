import bz2
import logging
import operator
import pathlib
import tempfile

import cv2
import torch.hub
import yaml
from omegaconf import DictConfig

from ptgaze.common.face_model import FaceModel
from ptgaze.common.face_model_68 import FaceModel68
from ptgaze.common.face_model_mediapipe import FaceModelMediaPipe

logger = logging.getLogger(__name__)


def get_3d_face_model(config: DictConfig) -> FaceModel:
    """根据配置返回对应的 3D 人脸模型实例。"""
    if config.face_detector.mode == 'mediapipe':
        return FaceModelMediaPipe()
    return FaceModel68()

class PtGazeUtils:
    """ptgaze 工具类，封装模型下载、配置校验、路径规格化等公共操作。"""

    # ------------------------------------------------------------------ #
    #  预训练模型下载                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def download_dlib_pretrained_model() -> None:
        """下载 dlib 68 点人脸关键点预测模型（若已存在则跳过）。"""
        logger.debug('Called download_dlib_pretrained_model()')

        dlib_model_dir = pathlib.Path('~/.ptgaze/dlib/').expanduser()
        dlib_model_dir.mkdir(exist_ok=True, parents=True)
        dlib_model_path = dlib_model_dir / 'shape_predictor_68_face_landmarks.dat'
        logger.debug(f'dlib model path: {dlib_model_path.as_posix()}')

        if dlib_model_path.exists():
            logger.debug(f'dlib pretrained model already exists: {dlib_model_path.as_posix()}')
            return

        logger.debug('Downloading dlib pretrained model...')
        bz2_path = dlib_model_path.as_posix() + '.bz2'
        torch.hub.download_url_to_file(
            'http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2',
            bz2_path,
        )
        with bz2.BZ2File(bz2_path, 'rb') as f_in, open(dlib_model_path, 'wb') as f_out:
            f_out.write(f_in.read())

    @staticmethod
    def download_mpiigaze_model() -> pathlib.Path:
        """下载 MPIIGaze ResNet 预训练权重（若已存在则跳过）。"""
        return PtGazeUtils._download_model(
            filename='mpiigaze_resnet_preact.pth',
            url='https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiigaze_resnet_preact.pth',
        )

    @staticmethod
    def download_mpiifacegaze_model() -> pathlib.Path:
        """下载 MPIIFaceGaze ResNet 预训练权重（若已存在则跳过）。"""
        return PtGazeUtils._download_model(
            filename='mpiifacegaze_resnet_simple.pth',
            url='https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.1.0/mpiifacegaze_resnet_simple.pth',
        )

    @staticmethod
    def download_ethxgaze_model() -> pathlib.Path:
        """下载 ETH-XGaze ResNet18 预训练权重（若已存在则跳过）。"""
        return PtGazeUtils._download_model(
            filename='eth-xgaze_resnet18.pth',
            url='https://github.com/hysts/pytorch_mpiigaze_demo/releases/download/v0.2.2/eth-xgaze_resnet18.pth',
        )

    @staticmethod
    def _download_model(filename: str, url: str) -> pathlib.Path:
        """通用模型下载辅助方法。"""
        output_dir = pathlib.Path('~/.ptgaze/models/').expanduser()
        output_dir.mkdir(exist_ok=True, parents=True)
        output_path = output_dir / filename
        if not output_path.exists():
            logger.debug(f'Downloading pretrained model: {filename}')
            torch.hub.download_url_to_file(url, output_path.as_posix())
        else:
            logger.debug(f'Pretrained model already exists: {output_path}')
        return output_path

    # ------------------------------------------------------------------ #
    #  相机参数                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def generate_dummy_camera_params(config: DictConfig) -> None:
        """根据图像/视频尺寸生成虚拟相机内参文件，并写入 config。"""
        logger.debug('Called generate_dummy_camera_params()')

        h, w = PtGazeUtils._resolve_frame_size(config)
        logger.debug(f'Frame size: ({w}, {h})')

        out_file = tempfile.NamedTemporaryFile(suffix='.yaml', delete=False)
        logger.debug(f'Creating dummy camera param file: {out_file.name}')

        camera_dict = {
            'image_width': w,
            'image_height': h,
            'camera_matrix': {
                'rows': 3,
                'cols': 3,
                'data': [w, 0., w // 2, 0., w, h // 2, 0., 0., 1.],
            },
            'distortion_coefficients': {
                'rows': 1,
                'cols': 5,
                'data': [0., 0., 0., 0., 0.],
            },
        }
        with open(out_file.name, 'w') as f:
            yaml.safe_dump(camera_dict, f)

        config.gaze_estimator.camera_params = out_file.name
        logger.debug(f'Updated config.gaze_estimator.camera_params -> {out_file.name}')

    @staticmethod
    def _resolve_frame_size(config: DictConfig) -> tuple[int, int]:
        """从图像或视频配置中解析帧的 (height, width)。"""
        if config.demo.image_path:
            image = cv2.imread(pathlib.Path(config.demo.image_path).expanduser().as_posix())
            return image.shape[:2]  # (h, w)

        if config.demo.video_path:
            path = pathlib.Path(config.demo.video_path).expanduser().as_posix()
            logger.debug(f'Opening video: {path}')
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f'{config.demo.video_path} could not be opened.')
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            cap.release()
            logger.debug(f'Closed video: {path}')
            return h, w

        raise ValueError('Neither image_path nor video_path is specified in config.demo.')

    # ------------------------------------------------------------------ #
    #  路径规格化                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def expanduser_all(config: DictConfig) -> None:
        """将 config 中所有路径字段展开为绝对路径（~ 展开）。"""
        if hasattr(config.face_detector, 'dlib_model_path'):
            config.face_detector.dlib_model_path = PtGazeUtils._expanduser(
                config.face_detector.dlib_model_path
            )
        config.gaze_estimator.checkpoint = PtGazeUtils._expanduser(
            config.gaze_estimator.checkpoint
        )
        config.gaze_estimator.camera_params = PtGazeUtils._expanduser(
            config.gaze_estimator.camera_params
        )
        config.gaze_estimator.normalized_camera_params = PtGazeUtils._expanduser(
            config.gaze_estimator.normalized_camera_params
        )
        for attr in ('image_path', 'video_path', 'output_dir'):
            if hasattr(config.demo, attr):
                setattr(config.demo, attr, PtGazeUtils._expanduser(getattr(config.demo, attr)))

    @staticmethod
    def _expanduser(path: str) -> str:
        """展开单个路径字符串中的 ~，若为空则原样返回。"""
        if not path:
            return path
        return pathlib.Path(path).expanduser().as_posix()

    # ------------------------------------------------------------------ #
    #  路径合法性校验                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def check_path_all(config: DictConfig) -> None:
        """校验 config 中所有必要路径是否存在且为文件。"""
        if config.face_detector.mode == 'dlib':
            PtGazeUtils._check_path(config, 'face_detector.dlib_model_path')
        PtGazeUtils._check_path(config, 'gaze_estimator.checkpoint')
        PtGazeUtils._check_path(config, 'gaze_estimator.camera_params')
        PtGazeUtils._check_path(config, 'gaze_estimator.normalized_camera_params')
        if config.demo.image_path:
            PtGazeUtils._check_path(config, 'demo.image_path')
        if config.demo.video_path:
            PtGazeUtils._check_path(config, 'demo.video_path')

    @staticmethod
    def _check_path(config: DictConfig, key: str) -> None:
        """校验单个配置路径是否存在且为文件，否则抛出异常。"""
        path_str = operator.attrgetter(key)(config)
        path = pathlib.Path(path_str)
        if not path.exists():
            raise FileNotFoundError(f'config.{key}: {path.as_posix()} not found.')
        if not path.is_file():
            raise ValueError(f'config.{key}: {path.as_posix()} is not a file.')