# Conect Segment-Anything with CLIP
We aim to classify the output masks of [segment-anything](https://github.com/facebookresearch/segment-anything) with the off-the-shelf [CLIP](https://github.com/openai/CLIP) models. The cropped image corresponding to each mask is sent to the CLIP model.
<img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/imgs/pipeline.png" width="100%" height="50%">

## Other Nice Works

#### Editing-Related Works
1. [sail-sg/EditAnything](https://github.com/sail-sg/EditAnything) 
2. [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) 
3. [geekyutao/Inpaint-Anything](https://github.com/geekyutao/Inpaint-Anything)
4. [Luodian/RelateAnything](https://github.com/Luodian/RelateAnything)

#### Nerf-Related Works 
1. [ashawkey/Segment-Anything-NeRF](https://github.com/ashawkey/Segment-Anything-NeRF)
2. [Anything-of-anything/Anything-3D](https://github.com/Anything-of-anything/Anything-3D)
3. [Jun-CEN/SegmentAnyRGBD](https://github.com/Jun-CEN/SegmentAnyRGBD)
4. [Pointcept/SegmentAnything3D](https://github.com/Pointcept/SegmentAnything3D)

#### Segmentation-Related Works
1. [maxi-w/CLIP-SAM](https://github.com/maxi-w/CLIP-SAM)  
2. [Curt-Park/segment-anything-with-clip](https://github.com/Curt-Park/segment-anything-with-clip)  
3. [kadirnar/segment-anything-video](https://github.com/kadirnar/segment-anything-video) 
4. [fudan-zvg/Semantic-Segment-Anything](https://github.com/fudan-zvg/Semantic-Segment-Anything) 
5. [continue-revolution/sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything) 
6. [RockeyCoss/Prompt-Segment-Anything](https://github.com/RockeyCoss/Prompt-Segment-Anything) 
7. [ttengwang/Caption-Anything](https://github.com/ttengwang/Caption-Anything)  
8. [ngthanhtin/owlvit_segment_anything](https://github.com/ngthanhtin/owlvit_segment_anything)   
9. [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything)
10. [helblazer811/RefSAM](https://github.com/helblazer811/RefSAM)  
11. [Hedlen/awesome-segment-anything](https://github.com/Hedlen/awesome-segment-anything)  
12. [ziqi-jin/finetune-anythin](https://github.com/ziqi-jin/finetune-anything)
13. [ylqi/Count-Anything](https://github.com/ylqi/Count-Anything)
14. [xmed-lab/CLIP_Surgery](https://github.com/xmed-lab/CLIP_Surgery)
15. [RockeyCoss/Prompt-Segment-Anything](https://github.com/RockeyCoss/Prompt-Segment-Anything)
16. [segments-ai/panoptic-segment-anything](https://github.com/segments-ai/panoptic-segment-anything)
17. [Cheems-Seminar/grounded-segment-any-parts](https://github.com/Cheems-Seminar/grounded-segment-any-parts)
18. [aim-uofa/Matcher](https://github.com/aim-uofa/Matcher)
19. [SysCV/sam-hq](https://github.com/SysCV/sam-hq)
20. [CASIA-IVA-Lab/FastSAM](https://github.com/CASIA-IVA-Lab/FastSAM)
21. [ChaoningZhang/MobileSAM](https://github.com/ChaoningZhang/MobileSAM)
22. [JamesQFreeman/Sam_LoRA](https://github.com/JamesQFreeman/Sam_LoRA)
23. [UX-Decoder/Semantic-SAM](https://github.com/UX-Decoder/Semantic-SAM)
24. [cskyl/SAM_WSSS](https://github.com/cskyl/SAM_WSSS)
25. [ggsDing/SAM-CD](https://github.com/ggsDing/SAM-CD)
26. [yformer/EfficientSAM](https://github.com/yformer/EfficientSAM)

#### Labelling-Related Works
1. [vietanhdev/anylabeling](https://github.com/vietanhdev/anylabeling)
2. [anuragxel/salt](https://github.com/anuragxel/salt)


#### Tracking-Related Works
1. [gaomingqi/track-anything](https://github.com/gaomingqi/track-anything) 
2. [z-x-yang/Segment-and-Track-Anything](https://github.com/z-x-yang/Segment-and-Track-Anything)
3. [achalddave/segment-any-moving](https://github.com/achalddave/segment-any-moving)

#### Medical-Related Works
1. [bowang-lab/medsam](https://github.com/bowang-lab/medsam)
2. [hitachinsk/SAMed](https://github.com/hitachinsk/SAMed)
3. [cchen-cc/MA-SAM](https://github.com/cchen-cc/MA-SAM#ma-sam-modality-agnostic-sam-adaptation-for-3d-medical-image-segmentation)
4. [OpenGVLab/SAM-Med2D](https://github.com/OpenGVLab/SAM-Med2D)

## Todo
1. We plan to connect segment-anything with [MaskCLIP](https://github.com/chongzhou96/MaskCLIP).  
2. We plan to finetune on the COCO and LVIS datasets.  


## Run Demo
Download the [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) model from the SAM repository and put it at ```./SAM-CLIP/```. Follow the instructions to install segment-anything and clip packages using the following command. 
```
cd SAM-CLIP; pip install -e .
pip install git+https://github.com/openai/CLIP.git
```
Then run the following script:
```
sh run.sh
```

## Example 
Input an example image and a point (250, 250) to the SAM model. The input image and output three masks as follows:
<center><img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/imgs/ADE_val_00000001.jpg" width="50%" height="50%"></center>

The three masks and corresponding predicted category are as follows:
<center>
<img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/outs/ADE_val_00000001/outs.png" width="100%" height="50%"> 
</center>
<center>
<img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/outs/ADE_val_00000001/logits.png" width="100%" height="50%"> 
</center>

You can change the point location at L273-274 of ```scripts/amp_points.py```.
```
## input points 
input_points_list = [[250, 250]]
label_list = [1]
```
