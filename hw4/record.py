from pathlib import Path
import math
import imageio.v2 as imageio
import numpy as np

def record_hw4(app, step, output, duration, fps, dt):
    output = Path(output)
    frame_count = math.ceil(duration * fps)

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(output), fps=fps, codec='libx264',
        pixelformat='yuv420p', macro_block_size=1)

    simulated_frames = 0
    with writer:
        for frame in range(frame_count):
            target = math.floor(frame / (fps * dt))
            command_encoder = app.device.create_command_encoder()
            while simulated_frames < target:
                step(command_encoder)
                command_encoder.global_barrier()
                simulated_frames += 1
            app.draw(command_encoder)

            submission = app.device.submit_command_buffer(command_encoder.finish())
            app.device.wait_for_submit(submission)

            image = app.color_image.to_numpy()
            rgb = image[:, :, :3]
            rgb = np.clip(rgb, 0.0, 1.0)
            # sRGB conversion
            rgb = np.where(
                rgb <= 0.0031308,
                12.92 * rgb,
                1.055 * np.power(rgb, 1.0 / 2.4) - 0.055
            )
            rgb = np.ascontiguousarray(
                (rgb * 255.0 + 0.5).astype(np.uint8))
            writer.append_data(rgb)

    print(f'Saved {output}')