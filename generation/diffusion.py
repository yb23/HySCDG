import torch



from diffusers import ControlNetModel#, StableDiffusionInpaintPipeline 
from controlnet_inpaint import StableDiffusionControlNetInpaintPipeline
from pipeline_inpaint import StableDiffusionInpaintPipeline


def loadControlNet(ControlNet_path="/home/YBenidir/Documents/CHECKPOINTS/ControlNet_V2_2024_01_16/controlnet", device="cuda"):
    controlnet = ControlNetModel.from_pretrained(ControlNet_path).to(device)
    return controlnet

def loadPipeline(model_path="/home/YBenidir/Documents/StableDiffusion2/Inpainting_12000/", controlnet=None, device="cuda"):
    if controlnet is None:
        pipe = StableDiffusionInpaintPipeline.from_pretrained(model_path, torch_dtype=torch.float32)
    else:
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(model_path, controlnet=controlnet, torch_dtype=torch.float32)
    pipe.to(device)
    return pipe