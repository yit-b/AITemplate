#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import click
import torch
import os

from aitemplate.testing.benchmark_pt import benchmark_torch_function
from aitemplate.utils.import_path import import_parent
from diffusers import EulerDiscreteScheduler

if __name__ == "__main__":
    import_parent(filepath=__file__, level=1)

from src.pipeline_stable_diffusion_ait import StableDiffusionAITPipeline


@click.command()
@click.option(
    "--local-dir",
    default="./tmp/diffusers-pipeline/stabilityai/stable-diffusion-v2",
    help="the local diffusers pipeline directory",
)
@click.option("--width", default=512, help="Width of generated image")
@click.option("--height", default=512, help="Height of generated image")
@click.option(
    "--steps", type=int, default=50, help="The number of denoising steps. More denoising steps usually lead to a higher quality image at the expense of slower inference."
)
@click.option(
    "--guidance-scale", type=float, default=7.5, help="Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598). `guidance_scale` is defined  as `w` of equation 2. of [Imagen Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`, usually at the expense of lower image quality."
)
@click.option("--batch", default=1, help="Batch size of generated image")
@click.option("--prompt", default="A vision of paradise, Unreal Engine", help="prompt")
@click.option("--negative-prompt", default=None, help="The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`)")
@click.option(
    "--out-dir", default="output_images", help="output directory"
)
def run(local_dir, width, height, steps, guidance_scale, batch, prompt, negative_prompt, out_dir):
    pipe = StableDiffusionAITPipeline.from_pretrained(
        local_dir,
        scheduler=EulerDiscreteScheduler.from_pretrained(
            local_dir, subfolder="scheduler"
        ),
        revision="fp16",
        torch_dtype=torch.float16,
    ).to("cuda")

    prompt = [prompt] * batch
    negative_prompt = [negative_prompt] * batch
    with torch.autocast("cuda"):
        images = pipe(
            prompt=prompt, 
            height=height, 
            width=width, 
            num_inference_steps=steps, 
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
        ).images

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    base_count = len(os.listdir(out_dir))
    out_paths = []
    for image in images:
        out = os.path.join(out_dir, f"{base_count:05}.png")
        image.save(out)
        out_paths.append(out)
        base_count += 1
    print(f"Wrote out to {out_paths}")


if __name__ == "__main__":
    run()
