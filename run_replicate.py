import os

import numpy as np
import replicate
import rp


@rp.globalize_locals
def run_vace_via_replicate(video, mask, prompt="", name=""):
    video = rp.as_byte_images(rp.as_rgb_images(video))
    mask  = rp.as_byte_images(rp.as_rgb_images(mask ))
    assert video.shape == mask.shape, (video.shape, mask.shape)

    #mask = invert_images(mask) #They do it the opposite of me, I like white to mean "keep"
    masked_video = np.maximum(video, mask)

    masked_video_path = f'replicate_input_masked_video_{name}.mp4'
    mask_path         = f'replicate_input_mask_{name}.mp4'

    masked_video_path=rp.get_unique_copy_path(masked_video_path)
    mask_path        =rp.get_unique_copy_path(mask_path        )

    rp.save_video_mp4(masked_video, masked_video_path, video_bitrate="max", framerate=20)
    rp.save_video_mp4(mask        , mask_path        , video_bitrate="max", framerate=20)

    print(f"Running replicate on {name}")
    replicate_output = replicate.run(
        "prunaai/vace-14b:bbafc615de3e3903470a335f94294810ced166309adcba307ac8692113a7b273",
        input={
            "seed": -1,
            "size": "832*480",
            "prompt": prompt,
            "src_mask": open(mask_path, "rb"),
            "frame_num": 81,
            "src_video": open(masked_video_path, "rb"),
            "speed_mode": "Extra Juiced 🚀 (even more speed)",
            "sample_shift": 16,
            "sample_steps": 50,
            "sample_solver": "unipc",
            "sample_guide_scale": 5,
        },
    )

    output_url = replicate_output.url
    output_file = f'replicate_output_{name}.mp4'
    rp.download_url(output_url, output_file, show_progress=True)
    rp.fansi_print(output_url,'green bold italic')

    output_video = rp.load_video(output_file)

    blended_output_video = rp.as_numpy_array([rp.laplacian_blend(of,f,m) for of,f,m in zip(output_video,video,mask)])

    return blended_output_video

def mask_matrix_to_video(mask):
    """
    X axis of matrix is video X, Y axis of matrix is video Time

    Args:
        mask: 2D matrix [time, width]
        height: Desired height of output video

    Returns:
        3D array [time, height, width]
    """
    mask = rp.as_binary_image(rp.as_grayscale_image(mask))
    return np.repeat(mask[:, np.newaxis, :], H, axis=1)

def shift_video(video, *, dx=0, dy=0, dt=0):
    """
    Video is a THWC tensor
    dx, dy and dt are ratios between -1 and 1 for shifting
    """
    T, H, W, C = video.shape

    x_shift = round(dx * W)
    y_shift = round(dy * H)
    t_shift = round(dt * T)

    result = video
    result = np.roll(result, t_shift, axis=0)
    result = np.roll(result, y_shift, axis=1)
    result = np.roll(result, x_shift, axis=2)

    return result

@rp.globalize_locals
def make_masks():
    #Generate the 2d masks
    mask2d_A = np.zeros((T,W))
    mask2d_B = np.zeros((T,W))
    mask2d_C = np.zeros((T,W))

    mask_W = round(.2 * W)
    mask_T = round(.2 * T)

    mask2d_A[  T//2-mask_T:T//2+mask_T]=1
    mask2d_A[:,W//2-mask_W:W//2+mask_W]=1

    mask2d_B[T//2-mask_T:T//2+mask_T, W//2-mask_W:W//2+mask_W]=1

    mask2d_C = mask2d_B

    mask_A = mask_matrix_to_video(mask2d_A)
    mask_B = mask_matrix_to_video(mask2d_B)
    mask_C = mask_matrix_to_video(mask2d_C)

    # rp.display_image(rp.tiled_images([mask2d_A,mask2d_B,mask2d_C],length=3))

@rp.globalize_locals
def run(raw_video_path="/Users/ryan/Downloads/Aurora.mp4",prompt="The aurora borealis",name='untitled'):
    #Sign in to replicate
    api_key = rp.base64_to_object('eJxrYJmqwwABPRpFFvHBFkFZKf7eBgbOFlFBBT5pvl5RAek+yWFGoemRFuFlhl7OLgGBU/QAgOAPTw==')
    os.environ['REPLICATE_API_TOKEN']=api_key

    # Sync these with run_vace_via_replicate's input
    T = 81
    H = 480
    W = 832

    make_masks()

    raw_video = rp.load_video(raw_video_path, use_cache=True)
    raw_video = rp.resize_list(raw_video, T)
    raw_video = rp.resize_images(raw_video, size=(H, W))

    with rp.SetCurrentDirectoryTemporarily(rp.make_directory('outputs')):
        rp.save_video_mp4(raw_video, f'{name}_raw.mp4', video_bitrate="max", framerate=20)

        in_A  = shift_video(raw_video, dx=1/2, dt=1/2)
        out_A = run_vace_via_replicate(in_A, mask_A, prompt, f"{name}_A")

        in_B  = shift_video(out_A, dt=-1/2)
        out_B = run_vace_via_replicate(in_B, mask_B, prompt, f"{name}_B")

        in_C  = shift_video(out_B, dx=-1/2, dt=1/2)
        out_C = run_vace_via_replicate(in_C, mask_C, prompt, f"{name}_C")

        out_D = shift_video(out_C, dt=-1 / 2)

        out_path = rp.get_unique_copy_path(f'{name}_seamless.mp4')

        rp.save_video_mp4(out_D, out_path, video_bitrate="max", framerate=20)

        return out_path

@rp.globalize_locals
def main():
    from selected_videos import captions as prompts
    from selected_videos import pano_vids as raw_video_paths
    from selected_videos import videoids as names

    for raw_video_path, prompt, name in zip(raw_video_paths, prompts, names):
        run(raw_video_path, prompt, name)


if __name__=='__main__':
    main()
