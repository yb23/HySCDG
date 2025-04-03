import argparse
import os

from generation import diffusion
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torch.nn.functional as F

import glob
import rasterio
import pandas as pd
import numpy as np
import geopandas as gpd
import cv2
import matplotlib.pyplot as plt

from src import flair

from scipy.ndimage.measurements import label as connexLabel


BAD_IMGS = ['IMG_028743.tif', 'IMG_010273.tif', 'IMG_010149.tif', 'IMG_045836.tif', 'IMG_041535.tif', 'IMG_056961.tif', 'IMG_045121.tif', 'IMG_037132.tif', 'IMG_042848.tif', 'IMG_008779.tif', 'IMG_041641.tif', 'IMG_043548.tif', 'IMG_004285.tif', 'IMG_033005.tif', 'IMG_055677.tif', 'IMG_047183.tif', 'IMG_051692.tif', 'IMG_006430.tif', 'IMG_020816.tif', 'IMG_058499.tif', 'IMG_021655.tif', 'IMG_000030.tif', 'IMG_058434.tif', 'IMG_038057.tif', 'IMG_017856.tif', 'IMG_045716.tif', 'IMG_000372.tif', 'IMG_018288.tif', 'IMG_058417.tif', 'IMG_044267.tif', 'IMG_005746.tif', 'IMG_014276.tif', 'IMG_017089.tif', 'IMG_049341.tif', 'IMG_036437.tif']

def maskAndReplaceObjects(raster, img, ctrl, data_to_mask, replacing_classes, replacing_freqs, global_mask_buffer=0, semantic_mask_buffer=0, proportional_semantic_mask=False, buffer_func=lambda x:0.25*x**0.5):
    ctrl1 = np.copy(ctrl)
    for idx, row in data_to_mask.iterrows():
        mask_row, _, _ = rasterio.mask.raster_geometry_mask(raster, [row["geometrie"]], crop=False, invert=True)
        mask_buffered, _, _ = rasterio.mask.raster_geometry_mask(raster, [row["geometrie"].buffer(semantic_mask_buffer)], crop=False, invert=True)
        freq_masked_class = np.bincount(ctrl[mask_row.astype(bool)[None,:,:]], minlength=20)
        xx, yy = np.where(mask_row.astype(bool))
        x0, y0 = int(xx.mean()), int(yy.mean())
        masked_class = ctrl[0,x0,y0]
        labls, n_components = connexLabel(ctrl[0]==masked_class)
        mask_row = ((labls == labls[x0,y0]) & (mask_buffered.astype(bool)))
        freq_masked_class = np.bincount(ctrl[mask_row.astype(bool)[None,:,:]], minlength=20)
        freq_masked_class = (1 + freq_masked_class)
        probs = replacing_freqs * 1/freq_masked_class
        probs = probs / probs.sum()
        repl_class = np.random.choice(replacing_classes, p=probs)
        ctrl1 = ctrl1 * (1 - mask_row) + repl_class * mask_row
    if global_mask_buffer>0:
        geometry_to_mask = list(data_to_mask["geometrie"].buffer(global_mask_buffer))
    else:
        geometry_to_mask = list(data_to_mask["geometrie"])
    mask, _, _ = rasterio.mask.raster_geometry_mask(raster, geometry_to_mask, crop=False, invert=True)
    img_mask = img * (1 - mask)
    return mask, img_mask, ctrl1


class FLAIR_Dataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """
    def __init__(
        self,
        instance_data_root="../data/CHG/IMG/",
        size=512,
        prompts_path="../data/Flair_BigPrompts.csv",
        image_channels=3,
        doOneHot=False,
        num_classes=19,
        already_imgs = {},
        dfObjects_path = "../data/instancesFootprints.pkl",
        labels_save_path = "../data/CHG_New/LBL3/",
        mask_blur = False,
        mask_blur_proportion = 1.,
        prompt_from_full_mask = True,
        save_inpainting_mask = False,
        inpaint_mask_save_path = "../data/CHG_New/LBL3/",
        save_root="../data/GEN_SAMPLES2/",
        imgs_list=[]
    ):
        self.size = size
        self.H, self.W = size, size
        self.image_channels = image_channels
        self.doOneHot = doOneHot
        self.save_inpainting_mask = save_inpainting_mask
        self.num_classes = num_classes
        self.dfObjects = pd.read_pickle(dfObjects_path)
        self.dfObjects = self.dfObjects[self.dfObjects["cledeb"].isin(["BATIMENT","TERRSPOR","POSTRANS","CONSLINE","RESERVOI","CONSSURF","CONSPONC"])]
        self.dfObjects[["xmin","ymin","xmax","ymax"]] = self.dfObjects.bounds
        self.dfObjects = self.dfObjects[self.dfObjects.area<250*250]
        self.save_root=save_root
        self.instance_data_root = Path(instance_data_root)
        if not self.instance_data_root.exists():
            raise ValueError("Instance images root doesn't exists.")
        
        self.labels_save_path = labels_save_path
        self.inpaint_mask_save_path = inpaint_mask_save_path

        self.images_path = glob.glob(instance_data_root+"/**/*.tif", recursive=True)
        if len(imgs_list)>0:  # Choose only a list of image files to use for generation
            self.images_path = [x for x in self.images_path if x.split("/")[-1] in imgs_list]
       
        print(len(self.images_path))
        print("Already done : {}".format(len(already_imgs)))
        self.images_path = [x for x in self.images_path if x.split("/")[-1] not in already_imgs]
        print("Remaining : {}".format(len(self.images_path)))
        self.image_ids = [x.split("/")[-1][:-4] for x in self.images_path]
        self.labels_path =  [x.replace("aerial","labels").replace("IMG","MSK").replace("img","msk") for x in self.images_path]
        

        dfPrompts = pd.read_csv(prompts_path, index_col=0)["prompt"].to_dict()
        
        self.instance_prompt = [dfPrompts.get(x+".tif","") for x in self.image_ids]
        

        self.num_instance_images = len(self.images_path)

        print("Total training images : "+str(self.num_instance_images))
        self._length = self.num_instance_images

        self.mask_blur = mask_blur
        self.mask_blur_proportion = mask_blur_proportion

        self.prompt_from_full_mask = prompt_from_full_mask

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        example = {}
        raster = rasterio.open(self.images_path[index % self.num_instance_images]) 
        ctrl = (rasterio.open(self.labels_path[index % self.num_instance_images]).read()).astype(np.uint8)
        image_id = self.image_ids[index % self.num_instance_images]
        instance_image = raster.read()[:self.image_channels]

        data = self.dfObjects[self.dfObjects["image_id"]==image_id]

        if len(data)>0:
            try:
                if "largeur_de_chaussee" in data.columns:
                    data["geometrie"] = data.apply(lambda x:x["geometrie"].buffer(x["largeur_de_chaussee"]*0.5) if ((x["largeur_de_chaussee"] is not None) and (x["largeur_de_chaussee"]>0)) else x["geometrie"], axis=1).buffer(2)
                else:
                    data["geometrie"] = data["geometrie"].buffer(2)
                data["area"] = data.area
                data = data[data["area"]>0].sort_values("area", ascending=False).reset_index(drop=True)
                n = len(data)
            except Exception as e:
                print("Erreur : {}".format(e))
                n = 0
        else:
            n=0
        
        n_objects = min(int(round(np.sqrt(np.random.uniform())/0.3)), n)
        n_noChanges = int(3 * np.sqrt(np.random.uniform()*(4.5 - n_objects)) / np.sqrt(4))

        masks = []
        if n_objects>0:
            idxs = np.random.choice(n, n_objects, replace=False)   # Choix aléatoire d'indices d'objets à masquer
            data_to_mask = data.loc[idxs]
            mask, img_mask, ctrl1 = maskAndReplaceObjects(raster, instance_image, ctrl, data_to_mask, flair.REPLACE_CLASSES, flair.FREQ_CLASSES, global_mask_buffer=7, semantic_mask_buffer=5)
            mask = mask[None,:,:]
            chg_map = np.concatenate(((mask*255), ctrl, ctrl1))
            masks.append(mask)
        else:
            ctrl1 = ctrl
            chg_map = np.concatenate((np.zeros_like(ctrl), ctrl, ctrl))
            print("No change !")

        if n_noChanges>0:
            for k in range(n_noChanges):
                masks.append(flair.little_mask(ellipsis=False, min_size=int(70 *(1. + 2 * np.random.uniform()))))
        if len(masks)==0:
            mask = np.zeros([1,512,512]).astype(int)
            prompt_classes = flair.getMaskedObjects(ctrl1, 1-mask)
        else:
            mask = (sum([x for x in masks])>0).astype(int)
            if (self.prompt_from_full_mask) or (n_objects==0):
                prompt_classes = flair.getMaskedObjects(ctrl1, mask)
            else:
                prompt_classes = flair.getMaskedObjects(ctrl1, ctrl1!=ctrl)
        chg_surface = (ctrl1!=ctrl).sum() / 51.2**2
        prompt = flair.PROMPTS_START[np.random.randint(0,2)] + prompt_classes + " " + self.instance_prompt[index % self.num_instance_images] + ", high resolution, highly detailed"

        mask = mask.astype(float)
        if self.mask_blur and ((self.mask_blur_proportion==1) or (np.random.uniform()<=self.mask_blur_proportion)):
            mask = cv2.GaussianBlur(mask.astype(float), (199,199), 50)
        
    
        flair.saveRaster(self.labels_save_path + "{}.tif".format(image_id.replace("IMG","LBL")), chg_map.astype(np.uint8), raster.transform)
        if self.save_inpainting_mask:
            flair.saveRaster(self.inpaint_mask_save_path + "{}.tif".format(image_id.replace("IMG","MSK")), (mask*255).astype(np.uint8), raster.transform)

        if self.doOneHot:
            example["instance_labels"] = torch.moveaxis(F.one_hot(torch.Tensor(ctrl1[0]-1).to(torch.long), num_classes=self.num_classes),2,0)
        else:
            instance_label = np.moveaxis(flair.convert_to_color(ctrl1[0]),2,0)
            example["instance_labels"] = torch.Tensor(instance_label/255.0)  

        instance_image = instance_image[:self.image_channels]
        example["raster"] = raster
        example["instance_image"] = torch.Tensor((instance_image / 255.))
        example["mask"] = torch.Tensor(mask)
        example["image_id"] = image_id
        example["prompt"] = prompt
        example["chg_surface"] = chg_surface
        return example

def collate_fn(examples):
        examples = [example for example in examples if example is not None] 

        pixel_values = [example["instance_image"] for example in examples]
        masks = [example["mask"] for example in examples]
        conditioning_pixel_values = [example["instance_labels"] for example in examples]
        prompts = [example["prompt"] for example in examples]
        image_ids = [example["image_id"] for example in examples]
        transforms = [example["raster"].transform for example in examples]
        crs = [example["raster"].crs for example in examples]
        chg_surface = [example["chg_surface"] for example in examples]
        return {
            "images": pixel_values,
            "control_images": conditioning_pixel_values,
            "prompts": prompts,
            "masks": masks, 
            "image_ids":image_ids, "transforms":transforms, "crs":crs, "chg_surface":chg_surface
        }



def loadModel(device="cuda", model_path="/home/YBenidir/Documents/CHECKPOINTS/SD_MiniBatch/PIPE_38500/", controlnet_path="/home/YBenidir/Documents/CHECKPOINTS/ControlNetV3/checkpoint-23000/"):
    controlnet = diffusion.loadControlNet(ControlNet_path=controlnet_path, device=device)
    pipe = diffusion.loadPipeline(model_path=model_path, controlnet=controlnet, device=device)
    return pipe


def generateImages(args):
    SAVE_ROOT = f"{args.save_dir}/IMG{args.num_version}/"
    os.makedirs(SAVE_ROOT, exist_ok=True)
    os.makedirs(f"{args.save_dir}/LBL{args.num_version}/", exist_ok=True)
    if args.save_masks:
        os.makedirs(f"{args.save_dir}/MSK{args.num_version}/", exist_ok=True)

    device = "cpu" if args.cpu else "cuda"

    already_imgs = glob.glob(SAVE_ROOT + "*.tif")
    already_imgs = [x.split("/")[-1] for x in already_imgs]


    train_dataset = FLAIR_Dataset(
        instance_data_root=args.images_path,
        prompts_path=args.prompts_path,
        num_classes=args.num_classes,
        image_channels=3,
        already_imgs = set(already_imgs),
        dfObjects_path = args.dfobjects_path,
        labels_save_path = args.save_dir+"/LBL{}/".format(args.num_version),
        mask_blur = args.mask_blur,
        mask_blur_proportion = args.mask_blur_proportion,
        prompt_from_full_mask=args.prompt_from_full_mask,
        save_inpainting_mask = args.save_masks,
        inpaint_mask_save_path = args.save_dir+"/MSK{}/".format(args.num_version),
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch,
        num_workers=args.n_workers
    )

    pipe = loadModel(device=device, model_path=args.model_path, controlnet_path=args.controlnet_path)
    for step, batch in enumerate(train_dataloader):
        
            images = pipe(prompt=batch["prompts"], num_inference_steps=args.inference_steps, image=batch["images"], mask_image=batch["masks"], output_type="np", control_image=batch["control_images"], controlnet_conditioning_scale=args.conditioning_scale)
            img_ids = batch["image_ids"]
            transforms = batch["transforms"]
            crs = batch["crs"]
            for i,image in enumerate(images[0]):
                flair.saveRaster(SAVE_ROOT + img_ids[i]+".tif", (np.moveaxis(image,2,0)*255).astype(np.uint8), transforms[i], crs=crs[i])#"epsg:2154")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generation of the dataset")

    parser.add_argument(
        "--model_path",
        type=str,
        default="/gpfsscratch/rech/jrj/uhz23wx/pretrained/SD_MiniBatch_38500/",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default="/gpfsscratch/rech/jrj/uhz23wx/pretrained/controlnet/",
        help="Path to pretrained controlnet model or model identifier from huggingface.co/models."
        " If not specified controlnet weights are initialized from unet.",
    )
    #python generate.py --model_path="/home/YBenidir/Documents/CHECKPOINTS/SD_MiniBatch/PIPE_38500" --controlnet_path="/home/YBenidir/Documents/CHECKPOINTS/ControlNewPrompts/checkpoint-58500/controlnet" --batch=1 --images_path="/home/YBenidir/Documents/DATASETS/FLAIR1/flair_aerial_train" --save_dir="../data/CHG/" --prompts_path="../data/Flair_Prompts.csv" --dfobjects_path="../data/instancesFootprints.pkl" --num_version=15
    parser.add_argument("-b", "--batch", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--n_workers",type=int,default=0,help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",)
    
    parser.add_argument("--images_path",type=str,default="../path_to_flair/flair_aerial_train/",
                        help="Path to the FLAIR train images folder")
    
    parser.add_argument("--save_dir",type=str,default="../data/CHG/",help="Path to the folder where the generated images ands masks will be stored")
    parser.add_argument("--prompts_path",type=str,default="../data/FLAIR_Prompts.csv",help="Path to the csv file containing the prompts for FLAIR images")
    parser.add_argument("--dfobjects_path",type=str,default="../data/instancesFootprints.pkl", help="Path to the geopandas file (.pkl) containing the footprints of the instances that are inside the zone covered by FLAIR dataset")
    parser.add_argument("--cpu",default=False,action="store_true",help="Train on CPU")
    parser.add_argument("--conditioning_scale", type=float, default=1.0, help="For ControlNet")
    parser.add_argument("--inference_steps", type=int, default=20, help="Number of denoising steps for Stable Diffusion")

    parser.add_argument("--num_classes", type=int, default=20, help="Number of semantic classes for the labels")

    parser.add_argument("--num_version", type=int, default=15, help="To name the save folders for images and labels")
    parser.add_argument("--prompt_from_full_mask",default=False,action="store_true")
    parser.add_argument("--mask_blur",default=False,action="store_true")
    parser.add_argument("--mask_blur_proportion",default=1.0,type=float)
    parser.add_argument("--save_masks",default=False,action="store_true")

    args = parser.parse_args()
    generateImages(args)
    