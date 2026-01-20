import depthai as dai


# Create a function to initialise the person detection neural network
def det_nn_init(pipeline, camera, stereo, det_nn_archive,
                fps_limit, CONFIDENCE_THRESHOLD):
    det_nn = pipeline.create(dai.node.SpatialDetectionNetwork).build(
        input=camera,
        stereo=stereo,
        nnArchive=det_nn_archive,
        fps=float(fps_limit),)
    det_nn.setBoundingBoxScaleFactor(1)
    det_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    return det_nn


# Create a function to initialise the spatial camera
def spatial_camera_init(pipeline, nn_size, fps_limit):
    camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
    left_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
    right_camera = pipeline.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
    stereo = pipeline.create(dai.node.StereoDepth).build(
        left=left_camera.requestOutput(nn_size, fps=fps_limit),
        right=right_camera.requestOutput(nn_size, fps=fps_limit),
        presetMode=dai.node.StereoDepth.PresetMode.FAST_ACCURACY)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    stereo.setOutputSize(*nn_size)
    stereo.setLeftRightCheck(True)
    stereo.setRectification(True)
    return camera, stereo

# Create a function to initialise the video encoder
def video_encoder_init(pipeline, camera, nn_size, fps_limit):
    camera_nv12 = camera.requestOutput(
        size=nn_size,
        fps=fps_limit,
        type=dai.ImgFrame.Type.NV12)

    video_encoder = pipeline.create(dai.node.VideoEncoder)
    video_encoder.setMaxOutputFrameSize(nn_size[0] * nn_size[1] * 3)
    video_encoder.setDefaultProfilePreset(
        fps_limit, dai.VideoEncoderProperties.Profile.H264_MAIN)
    camera_nv12.link(video_encoder.input)


# Create a function to initialise the depth colourmap encoding
def depth_colourmap_encoding_init(pipeline, nn_size, fps_limit, apply_colourmap):
    depth_encoder_manip = pipeline.create(dai.node.ImageManip)
    depth_encoder_manip.setMaxOutputFrameSize(nn_size[0] * nn_size[1] * 3)
    depth_encoder_manip.initialConfig.setOutputSize(*nn_size)
    depth_encoder_manip.initialConfig.setFrameType(dai.ImgFrame.Type.NV12)
    apply_colourmap.out.link(depth_encoder_manip.inputImage)

    depth_encoder = pipeline.create(dai.node.VideoEncoder)
    depth_encoder.setMaxOutputFrameSize(nn_size[0] * nn_size[1] * 3)
    depth_encoder.setDefaultProfilePreset(
        fps_limit, dai.VideoEncoderProperties.Profile.H264_MAIN)
    depth_encoder_manip.out.link(depth_encoder.input)
