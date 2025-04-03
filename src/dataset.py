import torch
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning import LightningDataModule
import albumentations as A

from torchvision import transforms
import torchvision.transforms.functional as TF

import rasterio
import glob
import os
import pandas as pd
from skimage import img_as_float
import numpy as np

mappings_syntheworld = {"second":{
    1:1,#bareland
    2:1,#rangeland
    3:2,#developed space
    4:2,#road
    5:3,#tree
    6:4,#water
    7:1,#agriculture land
    8:5,#building
},
"hiucd":{
    1:8,#bareland
    2:2,#rangeland
    3:5,#developed space
    4:5,#road
    5:9,#tree
    6:1,#water
    7:2,#agriculture land
    8:3,#building
}}

mappings_changen = {"second":{
    1:1,#bareland
    2:1,#rangeland
    3:2,#developed space
    4:2,#road
    5:3,#tree
    6:4,#water
    7:1,#agriculture land
    8:5,#building
},
"hiucd":{
    1:8,#bareland
    2:2,#rangeland
    3:5,#developed space
    4:5,#road
    5:9,#tree
    6:1,#water
    7:2,#agriculture land
    8:3,#building
}}

mappings_flair = {"hiucd":{
    1:3,
    2:8, #perméable -> bare land ?
    3:5, #imperméable -> road
    4:8,  #sol nu -> bare land
    5:1, #eau
    6:9, 
    7:9,
    8:2, #brousaille -> grass   ???????
    9:7, #vigne -> others ???????
    10:2,
    11:2,  #culture ->  grass ??????
    12:8, #terre labourée -> bare land
    13:7, #piscine -> others
    14:7,   #neige -> others ???????
    15:8,   #coupe -> bare land
    16:8,   #mixte -> bare land
    17:9, #ligneux -> woodland
    18:4,
    19:7
},
"hrscd":{
    1:1,
    2:4, #perméable -> bare land ?
    3:1, #imperméable -> road
    4:2,  #sol nu -> bare land
    5:5, #eau
    6:3, 
    7:3,
    8:3, #brousaille -> grass   ???????
    9:2, #vigne -> others ???????
    10:2,
    11:2,  #culture ->  grass ??????
    12:2, #terre labourée -> bare land
    13:0, #piscine -> others
    14:0,   #neige -> others ???????
    15:2,   #coupe -> bare land
    16:3,   #mixte -> bare land
    17:3, #ligneux -> woodland
    18:0,
    19:0
},
"changenet":{
    1:1,
    2:4, #perméable -> bare land ?
    3:6, #imperméable -> road
    4:2,  #sol nu -> bare land
    5:5, #eau
    6:3, 
    7:3,
    8:3, #brousaille -> grass   ???????
    9:2, #vigne -> others ???????
    10:2,
    11:2,  #culture ->  grass ??????
    12:2, #terre labourée -> bare land
    13:0, #piscine -> others
    14:0,   #neige -> others ???????
    15:2,   #coupe -> bare land
    16:3,   #mixte -> bare land
    17:3, #ligneux -> woodland
    18:0,
    19:0
},
"second":{
    1:5, # building
    2:1, #perméable -> low vegetation
    3:2, #imperméable -> non vegetated surface
    4:2,  #sol nu -> non vegetated surface
    5:4, #eau 
    6:3,  #coniferous   tree
    7:3,  # deciduous  tree
    8:3, #brousaille -> tree
    9:1, #vigne ->  low vegetation
    10:1, # herbaceous -> low vegetation
    11:1,  #culture ->  low vegetation
    12:1, #terre labourée -> low vegetation
    13:6, #piscine -> playground
    14:0,   #neige -> others ???????
    15:1,   #coupe -> low vegetation
    16:3,   #mixte -> tree
    17:3, #ligneux -> tree
    18:0,
    19:0
},
}

norm_params = {"second" : [[np.array([113.40, 114.08,116.45]), np.array([48.30,  46.27,  48.14])], [np.array([111.07, 114.04, 118.18]), np.array([49.41,  47.01,  47.94])]],
                "hiucd" : [[np.array([110.558246,114.579483,102.117656]), np.array([45.781694,46.032597,45.822763])], [np.array([116.227830,119.769624,106.373077]), np.array([47.691569,48.136453,45.521357])]],
                "flair" : [[np.array([113.969231,118.263447,109.203833]), np.array([52.343770,46.023631,45.373587])], [np.array([110.491333,116.718188,105.585801]), np.array([48.846954,42.711528,41.647261])]],
                "syntheworld" : [[np.array([72.116843,67.363153,55.259185]), np.array([53.430304,47.303968,45.574611])], [np.array([70.548502,67.113177,56.751367]), np.array([54.300922,47.294327,46.971403])]],
                "changen" : [[np.array([113.868586,116.883469,100.237843]), np.array([49.643828,44.377541,46.810398])], [np.array([117.544422,119.909791,101.986801]), np.array([42.120861,36.575275,40.471772])]],
                "levir" : [[np.array([114.816410,113.900366,97.243376]), np.array([55.364678,52.023403,47.598292])], [np.array([88.108325,86.239870,73.647940]), np.array([40.186511,38.756556,36.824392])]],
                "s2looking" : [[np.array([44.792358,53.015604,52.016244]), np.array([20.026240,20.554073,19.831728])], [np.array([52.518009,56.162529,54.177058]), np.array([18.663814,16.367386,15.324433])]],
                }




def mixDicts(dict_flair, dict_target, ratio):  #### ratio = target data proportion in training set (0.5 for equal proportions target / pretraining)
    n1, n2 = len(dict_flair["IMG_A"]), len(dict_target["IMG_A"])
    ratio = ratio/(1-ratio)
    needed_examples = int(ratio * n1)
    q = needed_examples//n2
    r = needed_examples%n2
    dictMix = {}
    for key in dict_flair:
        dictMix[key] = dict_target[key] * q + dict_target[key][:r] + dict_flair[key]
    print(f"Mixing datasets : {n1} images in FlairChange. Adding {needed_examples} images from target (with {q} repetitions)")
    print(f"Total size : {len(dictMix['IMG_A'])}")
    return dictMix

def loadPretrainData(args):
        file_ext = ".tif" if args.pretrain_name=="fsc" else ".png"
        data = {"train":{"IMG_A":[], "IMG_B":[], "MSK":[], "MSK_A":[]},
                    "val":{"IMG_A":[], "IMG_B":[], "MSK":[], "MSK_A":[]}}
        available_images = glob.glob(args.pretrain_path+"B/*"+file_ext)

        if args.restr_train_set:
            available_images = available_images[:200]

        n_images = len(available_images)
        print("Total Images : {}".format(n_images))
        imgB_train = available_images[:int(0.8*n_images)]
        data["train"]["IMG_B"] += imgB_train
        data["train"]["IMG_A"] += [x.replace("/B/","/A/") for x in imgB_train]
        data["train"]["MSK"] += [x.replace("/B/","/LBL/").replace("IMG_","LBL_") for x in imgB_train]
        data["train"]["MSK_A"] += [""] * len(imgB_train)
        data["train"]["DS"] = [args.pretrain_name]*len(data["train"]["IMG_A"])

        imgB_val = available_images[int(0.8*n_images):]
        data["val"]["IMG_B"] += imgB_val
        data["val"]["IMG_A"] += [x.replace("/B/","/A/") for x in imgB_val]
        data["val"]["MSK"] += [x.replace("/B/","/LBL/").replace("IMG_","LBL_") for x in imgB_val]
        data["val"]["MSK_A"] += [""] * len(imgB_val)
        data["val"]["DS"] = [args.pretrain_name]*len(data["val"]["IMG_A"])

        data["test"] = data["val"]
        return data


def loadSyntheWorld(data_path, restr_train_set=False):
    data = {"train":{"IMG_A":[], "IMG_B":[], "MSK":[]},
                        "val":{"IMG_A":[], "IMG_B":[], "MSK":[]},}
    for num in [1,2,3,1024]:
        available_images = glob.glob(f"{data_path}/{num}/images/*.png")
        if restr_train_set:
            available_images = available_images[:200]
        n_images = len(available_images)
        print("Total Images : {}".format(n_images))
        imgB_train = available_images[:int(0.8*n_images)]
        data["train"]["IMG_B"] += imgB_train
        data["train"]["IMG_A"] += [x.replace("images","pre_event") for x in imgB_train]
        data["train"]["MSK"] += [x.replace("images","cd_mask") for x in imgB_train]
        #data_flair["train"]["MSK_SEM"] += [x.replace("images","ss_mask") for x in imgB_train]
        data["train"]["DS"] = ["syntheworld"]*len(data["train"]["IMG_A"])
        imgB_val = available_images[int(0.8*n_images):]
        data["val"]["IMG_B"] += imgB_val
        data["val"]["IMG_A"] += [x.replace("images","pre_event") for x in imgB_val]
        data["val"]["MSK"] += [x.replace("images","cd_mask") for x in imgB_val]
        #data_flair["val"]["MSK_SEM"] += [x.replace("images","ss_mask") for x in imgB_val]
        data["val"]["DS"] = ["syntheworld"]*len(data["val"]["IMG_A"])
    data["test"] = data["val"]
    return data

def loadChangen(data_path, restr_train_set=False):
    data = {"train":{"IMG_A":[], "IMG_B":[], "MSK":[]},
                        "val":{"IMG_A":[], "IMG_B":[], "MSK":[]},}
    available_images = glob.glob(f"{data_path}/t2_images/*.tif")
    if restr_train_set:
        available_images = available_images[:200]
    n_images = len(available_images)
    print("Total Images : {}".format(n_images))
    imgB_train = available_images[:int(0.8*n_images)]
    data["train"]["IMG_B"] += imgB_train
    data["train"]["IMG_A"] += [x.replace("t2_images","t1_images") for x in imgB_train]
    data["train"]["MSK"] += [x.replace("t2_images","t1_masks") for x in imgB_train]
    #data["train"]["MSK_SEM"] += [x.replace("images","ss_mask") for x in imgB_train]
    data["train"]["DS"] = ["changen"]*len(data["train"]["IMG_A"])
    imgB_val = available_images[int(0.8*n_images):]
    data["val"]["IMG_B"] += imgB_val
    data["val"]["IMG_A"] += [x.replace("t2_images","t1_images") for x in imgB_val]
    data["val"]["MSK"] += [x.replace("t2_images","t1_masks") for x in imgB_val]
    #data["val"]["MSK_SEM"] += [x.replace("images","ss_mask") for x in imgB_val]
    data["val"]["DS"] = ["changen"]*len(data["val"]["IMG_A"])
    data["test"] = data["val"]
    return data

def loadFSC(data_path, fsc_versions=[], mix_fsc_versions=False, restr_train_set=False):
    data = {"train":{"IMG_A":[], "IMG_B":[], "MSK":[], "MSK_A":[], "DS":[]},
                "val":{"IMG_A":[], "IMG_B":[], "MSK":[], "MSK_A":[], "DS":[]},
                    }
    for num in fsc_versions:
        available_images = glob.glob(data_path+"IMG{}/*.tif".format(num), recursive=True)
        if restr_train_set:
            available_images = available_images[:200]

        n_images = len(available_images)
        print("Total Images : {}".format(n_images))
        imgB_train = available_images[:int(0.8*n_images)]
        data["train"]["IMG_B"] += imgB_train
        data["train"]["IMG_A"] += [x.replace("IMG{}".format(num),"IMG") for x in imgB_train]
        data["train"]["MSK"] += [x.replace("IMG{}".format(num),"LBL{}".format(num)).replace("IMG_","LBL_") for x in imgB_train]
        data["train"]["MSK_A"] += [""] * len(imgB_train)
        data["train"]["DS"] += ["fsc"]*len(data["train"]["IMG_A"])

        imgB_val = available_images[int(0.8*n_images):]
        data["val"]["IMG_B"] += imgB_val
        data["val"]["IMG_A"] += [x.replace("IMG{}".format(num),"IMG") for x in imgB_val]
        data["val"]["MSK"] += [x.replace("/IMG{}/".format(num),"/LBL{}/".format(num)).replace("IMG_","LBL_") for x in imgB_val]
        data["val"]["MSK_A"] += [""] * len(imgB_val)
        data["val"]["DS"] += ["fsc"]*len(data["val"]["IMG_A"])
    
    if mix_fsc_versions:
        for num in fsc_versions:
            for num2 in fsc_versions:
                if num2>num:
                    available_images = glob.glob(data_path+"IMG{}/*.tif".format(num), recursive=True)
                    if restr_train_set:
                        available_images = available_images[:200]
                    n_images = len(available_images)
                    print("Total Images : {}".format(n_images))
                    imgB_train = available_images[:int(0.8*n_images)]
                    data["train"]["IMG_B"] += imgB_train
                    imgsA = [x.replace("/IMG{}/".format(num),"/IMG{}/".format(num2)).replace("IMG2_","IMG_") for x in imgB_train]
                    data["train"]["IMG_A"] += imgsA
                    data["train"]["MSK"] += [x.replace("/IMG{}/".format(num),"/LBL{}/".format(num)).replace("IMG_","LBL_") for x in imgB_train]
                    data["train"]["MSK_A"] += [x.replace("/IMG{}/".format(num2),"/LBL{}/".format(num2)).replace("IMG_","LBL_") for x in imgsA]
                    data["train"]["DS"] += ["fsc"]*len(data["train"]["IMG_A"])

                    imgB_val = available_images[int(0.8*n_images):]
                    data["val"]["IMG_B"] += imgB_val
                    imgsA = [x.replace("/IMG{}/".format(num),"/IMG{}/".format(num2)).replace("IMG2_","IMG_") for x in imgB_val]
                    data["val"]["IMG_A"] += imgsA
                    data["val"]["MSK"] += [x.replace("IMG2_","LBL_").replace("/IMG{}/".format(num),"/LBL{}/".format(num)).replace("IMG_","LBL_") for x in imgB_val]
                    data["val"]["MSK_A"] += [x.replace("IMG2_","LBL_").replace("/IMG{}/".format(num2),"/LBL{}/".format(num2)).replace("IMG_","LBL_") for x in imgsA]
                    data["val"]["DS"] += ["fsc"]*len(data["val"]["IMG_A"])
    data["test"] = data["val"]
    return data


def DataDict(data_path = "", dataset_name=None, file_ext="png"):
    dict_train = {}
    print(data_path + f"train/A/*.{file_ext}")
    imgsA = glob.glob(data_path + f"train/A/*.{file_ext}", recursive=True)

    imgsB = [x.replace("/A/","/B/") for x in imgsA]
    lbls = [x.replace("/A/","/label/") for x in imgsA]
    dict_train["IMG_A"] = imgsA
    dict_train["IMG_B"] = imgsB
    dict_train["MSK"] = lbls

    dict_val = {}
    imgsA = glob.glob(data_path + f"val/A/*.{file_ext}", recursive=True)
    imgsA_ = glob.glob(data_path + f"test/A/*.{file_ext}", recursive=True)
    if len(imgsA)>0 and len(imgsA_)>0:
        print("Validation and test set available")
        specific_from = "test"
        imgsB = [x.replace("/A/","/B/") for x in imgsA]
        lbls = [x.replace("/A/","/label/") for x in imgsA]
        dict_val["IMG_A"] = imgsA
        dict_val["IMG_B"] = imgsB
        dict_val["MSK"] = lbls

        dict_test = {}
        imgsB = [x.replace("/A/","/B/") for x in imgsA_]
        lbls = [x.replace("/A/","/label/") for x in imgsA_]
        dict_test["IMG_A"] = imgsA_
        dict_test["IMG_B"] = imgsB
        dict_test["MSK"] = lbls
    elif len(imgsA_)>0:
        print("No validation set but 1 test set : validation set = test set")
        specific_from = "test"
        dict_test = {}
        imgsB = [x.replace("/A/","/B/") for x in imgsA_]
        lbls = [x.replace("/A/","/label/") for x in imgsA_]
        dict_test["IMG_A"] = imgsA_
        dict_test["IMG_B"] = imgsB
        dict_test["MSK"] = lbls
        dict_val = dict_test
    elif os.path.isfile(data_path+"test_filenames.csv"):
        print("Using test_filenames.csv to create test set !")
        toRemove = set(pd.read_csv(data_path+"test_filenames.csv")["filename"])
        dict_test = {}
        print(len(dict_train["IMG_A"]))
        for key in dict_train:
            dict_test[key] = [x for x in dict_train[key] if x.split("/")[-1] in toRemove]
            dict_train[key] = [x for x in dict_train[key] if x.split("/")[-1] not in toRemove]
        dict_val = dict_test
        print(len(dict_train["IMG_A"]))
                               
    else:   ## No val, No test
        print("No validation set, no test set")
        specific_from = "train"
        imgsA = dict_train["IMG_A"]
        n = len(imgsA)
        n1 = int(0.8*n)
        dict_val = {}
        dict_val["IMG_A"] = imgsA[n1:]
        dict_val["IMG_B"] = imgsB[n1:]
        dict_val["MSK"] = lbls[n1:]
        dict_test = {}
        dict_test["IMG_A"] = imgsA[n1:]
        dict_test["IMG_B"] = imgsB[n1:]
        dict_test["MSK"] = lbls[n1:]

    dict_visu = {}
    lbls = glob.glob(data_path + f"specific/label/*.{file_ext}", recursive=True)
    if len(lbls)==0:
        dict_visu = None
    else:
        lbls = [x.replace("/specific/","/{}/".format(specific_from)) for x in lbls]
        imgsA = [x.replace("/label/","/A/") for x in lbls]
        imgsB = [x.replace("/A/","/B/") for x in imgsA]
        dict_visu["IMG_A"] = imgsA
        dict_visu["IMG_B"] = imgsB
        dict_visu["MSK"] = lbls
        if dataset_name is not None:
            dict_visu["DS"] = [dataset_name]*len(dict_visu["IMG_A"])

    if dataset_name is not None:
        dict_train["DS"], dict_val["DS"], dict_test["DS"] = [dataset_name]*len(dict_train["IMG_A"]), [dataset_name]*len(dict_val["IMG_B"]), [dataset_name]*len(dict_test["IMG_A"])

    return {"train":dict_train,
            "val":dict_val,
            "test":dict_test,
            "visu":dict_visu}


class Fit_Dataset(Dataset):

    def __init__(self,dict_files,num_classes=2, isBinary=True, H=512, W=512, augment_first_image=False, use_augmentations=False, use_inversion=False, normalize=False, num_channels=3, dataset_name="fsc", class_mapping_to=None, args=None):
        self.list_imgsA = np.array(dict_files["IMG_A"])
        self.list_imgsB = np.array(dict_files["IMG_B"])
        self.list_msks = np.array(dict_files["MSK"])
        if "MSK_A" in dict_files:
            self.list_msksA = np.array(dict_files["MSK_A"])
        self.augment_first_image = augment_first_image
        self.use_target_classes = (class_mapping_to is not None)
        self.num_classes = num_classes
        self.binary = isBinary
        self.num_channels = num_channels
        self.use_augmentations = use_augmentations
        self.use_inversion = use_inversion or use_augmentations or augment_first_image
        if self.use_inversion:
            print("Using random inversions")
        self.normalize = normalize
        self.use_crop256 = args.crop256
        self.use_crop512 = args.crop512

        if "DS" in dict_files:
            self.list_dataset = np.array(dict_files["DS"])
        else:
            self.list_dataset = [dataset_name] * len(self.list_imgsA)        

        if self.use_target_classes:
            if args.pretrain_name=="syntheworld":
                print(f"Mapping from SyntheWorld to {class_mapping_to}")
                self.class_mapping = mappings_syntheworld[class_mapping_to]
            elif args.pretrain_name=="changen":
                print(f"Mapping from Changen2 to {class_mapping_to}")
                self.class_mapping = mappings_changen[class_mapping_to]
            elif args.pretrain_name=="fsc":
                print(f"Mapping from FSC to {class_mapping_to}")
                self.class_mapping = mappings_flair[class_mapping_to]
            self.getDict = lambda x:self.class_mapping.get(x, 0)
            self.inv_mapping = {val:key for (key,val) in self.class_mapping.items()}
        else:
            self.inv_mapping = None
        
        
        
        self.crop256 = A.Compose([A.RandomCrop(256, 256, p=1.0)], additional_targets={'imageB': 'image', 'msk': 'mask'}, is_check_shapes=False)
        self.crop512 = A.Compose([A.RandomCrop(512, 512, p=1.0)], additional_targets={'imageB': 'image', 'msk': 'mask'}, is_check_shapes=False)

        self.trfFirst = A.Compose([
            A.RandomShadow(num_shadows_lower=1, num_shadows_upper=6, p=0.7),
            A.RandomToneCurve(scale=0.1, p=0.5),
            A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3, p=0.5),
            A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.6, p=0.5),
            A.RandomGamma(gamma_limit=(80,120), p=0.5)
        ])

        self.trfSimple= A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.RandomResizedCrop(H,W, scale=(0.8, 1.0), p=0.8),
            ],
            additional_targets={'imageB': 'image', 'msk': 'mask'}
        )

        self.trfColor = A.Compose([   
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.2), 
        ])
        
        if self.normalize:
            print("Normalization of the inputs !!!")
            self.normA = A.Normalize(mean=list(norm_params[dataset_name][0][0]/255.), std=list(norm_params[dataset_name][0][1]/255.))
            self.normB = A.Normalize(mean=list(norm_params[dataset_name][1][0]/255.), std=list(norm_params[dataset_name][1][1]/255.))
   

    def read_img(self, raster_file: str) -> np.ndarray:
        try:
            return rasterio.open(raster_file).read()[:self.num_channels]
        except:
            print(f"Fail loading : {raster_file}")
            return None
        

    def read_msk(self, raster_file: str) -> np.ndarray:
        if self.binary:
            return (rasterio.open(raster_file).read()[:1]>0).astype(int)    
        else:
            msk = rasterio.open(raster_file).read()
            return msk.astype(int)[:3]

    def __len__(self):
        return len(self.list_imgsA)

    def __getitem__(self, index):
        ds = self.list_dataset[index]
        image_fileA = self.list_imgsA[index]
        image_fileB = self.list_imgsB[index]
        mask_file = self.list_msks[index]
        imgB = self.read_img(raster_file=image_fileB)
        if imgB is None:
            imgB = np.zeros([self.num_channels,512,512]).astype(np.uint8)
        if (ds=="syntheworld") & (imgB.shape[1]==1024):
            image_fileA = image_fileB.replace("images", "small_pre_images")
        imgA = self.read_img(raster_file=image_fileA)
        if imgA is None:
            imgA = np.zeros([self.num_channels,512,512]).astype(np.uint8)
        
        if ds=="syntheworld":
            msk = (rasterio.open(mask_file).read()>0).astype(int)
            msk_sem = (rasterio.open(mask_file.replace("cd_mask","ss_mask")).read())
            msk = np.concatenate([msk, np.zeros_like(msk), msk_sem]).astype(int)
        elif ds=="changen":
            mskA = (rasterio.open(mask_file).read()).astype(int)
            mskB = (rasterio.open(mask_file.replace("t1_masks","t2_masks")).read()).astype(int)
            msk = np.concatenate([(mskA!=mskB).astype(int), mskA, mskB], axis=0)
        else:
            try:
                msk = self.read_msk(raster_file=mask_file)
                nomask = False
            except:
                msk = np.zeros([3,512,512]).astype(np.uint8)
                nomask = True
        if ds=="hiucd":
            if not nomask:
                msk = msk[[2,0,1]]  ### put the change labels on first layer, semantics image 1 on second layer and semantics image 2 on third layer
                msk[0] -= 1   ###  use -1, 0, 1 as labels (-1 for unlabeled)
            
        elif ds=="fsc":
            mask_fileA = self.list_msksA[index]
            if mask_fileA != "":
                mskA = self.read_msk(raster_file=mask_fileA)
                msk = np.stack([((msk[0]+mskA[0]>0) * (msk[2]!=mskA[2])).astype(int), mskA[2], msk[2]]).astype(int)
            else:
                msk[0] = ((msk[0]>0) * (msk[1]!=msk[2])).astype(int)
        elif (ds=="levir") or (ds=="s2looking"):
            msk[0] = (msk[0]>0).astype(int)
            msk[1] = 0
            msk[2] = msk[0] * 1   # 1 corresponds to Flair building class
        
        if self.use_crop512:
            timgs = self.crop512(image=np.moveaxis(imgA,0,2), imageB=np.moveaxis(imgB,0,2), msk=np.moveaxis(msk,0,2))
            imgA = np.moveaxis(timgs["image"],2,0)
            imgB = np.moveaxis(timgs["imageB"],2,0)
            msk = np.moveaxis(timgs["msk"],2,0)

        if self.use_crop256:
            timgs = self.crop256(image=np.moveaxis(imgA,0,2), imageB=np.moveaxis(imgB,0,2), msk=np.moveaxis(msk,0,2))
            imgA = np.moveaxis(timgs["image"],2,0)
            imgB = np.moveaxis(timgs["imageB"],2,0)
            msk = np.moveaxis(timgs["msk"],2,0)
        ##################
        if (ds=="syntheworld") & (imgA.shape[1]==1024):
            timgs = self.crop512(image=np.moveaxis(imgA,0,2), imageB=np.moveaxis(imgB,0,2), msk=np.moveaxis(msk,0,2))
            imgA = np.moveaxis(timgs["image"],2,0)
            imgB = np.moveaxis(timgs["imageB"],2,0)
            msk = np.moveaxis(timgs["msk"],2,0)
        if (self.use_target_classes and (ds in ["fsc","syntheworld","changen"])):
            msk[1:] = np.vectorize(self.getDict)(msk[1:])
        if self.augment_first_image:
            timg = self.trfFirst(image = np.moveaxis(imgA,0,2))
            imgA = np.moveaxis(timg["image"],2,0)

        if self.use_augmentations:
            timgs = self.trfSimple(image=np.moveaxis(imgA,0,2), imageB=np.moveaxis(imgB,0,2), msk=np.moveaxis(msk,0,2))
            #timgs["image"] = self.trfColor(image=timgs["image"])
            #timgs["imageB"] = self.trfColor(image=timgs["imageB"])
            imgA = np.moveaxis(timgs["image"],2,0)
            imgB = np.moveaxis(timgs["imageB"],2,0)
            msk = np.moveaxis(timgs["msk"],2,0)
        
        if self.normalize:
                imgA = np.moveaxis(self.normA(image=np.moveaxis(imgA,0,2))["image"],2,0)
                imgB = np.moveaxis(self.normB(image=np.moveaxis(imgB,0,2))["image"],2,0)
        else:
            imgA = img_as_float(imgA)
            imgB = img_as_float(imgB)

        if self.use_inversion:
            if torch.rand(1)>0.5:
                imgA, imgB = imgB, imgA
                if len(msk)>1:
                    msk[1], msk[2] = msk[2], msk[1]
        
        return {"image1": torch.as_tensor(imgA, dtype=torch.float),
                        "image2": torch.as_tensor(imgB, dtype=torch.float), 
                        "mask": torch.as_tensor(msk, dtype=torch.long),
                        "path":image_fileB}
    


class LEVIR_DataModule(LightningDataModule):

    def __init__(self,dict_train=None,dict_val=None,dict_test=None,num_workers=1,batch_size=2,drop_last=True,num_classes=2,num_channels=3, augment_first_image=False, use_augmentations=False, use_inversion=False, normalize=False, isBinary=True, dataset_name="fsc", class_mapping_to=None, args=None):
        super().__init__()
        self.dict_train = dict_train
        self.dict_val = dict_val
        self.dict_test = dict_test
        self.batch_size = batch_size
        self.num_classes, self.num_channels = num_classes, num_channels
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.pred_dataset = None
        self.drop_last = drop_last
        self.use_augmentations = use_augmentations
        self.use_inversion = use_inversion
        self.isBinary = isBinary
        self.normalize = normalize
        self.dataset_name = dataset_name
        self.class_mapping_to = class_mapping_to
        self.augment_first_image = augment_first_image
        self.collate_fn = None
        self.args =args
        
    def prepare_data(self):
        pass
    def setup(self, stage="fit"):
        if stage == "fit" or stage == "validate":
            self.train_dataset = Fit_Dataset(
                dict_files=self.dict_train,
                num_classes=self.num_classes,
                isBinary = self.isBinary,
                use_augmentations=self.use_augmentations,
                augment_first_image = self.augment_first_image,
                use_inversion = self.use_inversion,
                normalize=self.normalize,
                dataset_name=self.dataset_name, class_mapping_to=self.class_mapping_to,num_channels=self.num_channels,
                args=self.args)

            self.val_dataset = Fit_Dataset(
                dict_files=self.dict_val,
                num_classes=self.num_classes,
                isBinary = self.isBinary,
                use_augmentations=False,
                augment_first_image = self.augment_first_image,
                normalize=self.normalize,
                dataset_name=self.dataset_name, class_mapping_to=self.class_mapping_to,num_channels=self.num_channels,
                args=self.args
            )
            self.pred_dataset = Fit_Dataset(
                dict_files=self.dict_test,
                num_classes=self.num_classes,
                isBinary = self.isBinary,
                use_augmentations=False,
                normalize=self.normalize,
                dataset_name=self.dataset_name, class_mapping_to=self.class_mapping_to,num_channels=self.num_channels,
                args=self.args
            )

        elif stage == "predict":
            self.pred_dataset = Fit_Dataset(
                dict_files=self.dict_test,
                num_classes=self.num_classes,
                isBinary = self.isBinary,
                use_augmentations=False,
                normalize=self.normalize,
                dataset_name=self.dataset_name, class_mapping_to=self.class_mapping_to,num_channels=self.num_channels,
                args=self.args
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn = self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=True,   ###########################################
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn = self.collate_fn,
        )
    
    def test_dataloader(self):
        return DataLoader(
            dataset=self.pred_dataset,
            batch_size=self.batch_size, 
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn = self.collate_fn,
        )