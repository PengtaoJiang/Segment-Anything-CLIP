# Conect Segment-Anything with CLIP
We aim to classify the output masks of [segment-anything](https://github.com/facebookresearch/segment-anything) with the off-the-shelf [CLIP](https://github.com/openai/CLIP) models. The cropped image corresponding to each mask is sent to the CLIP model.
<img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/imgs/pipeline.png" width="100%" height="50%">

## Other Nice Works

#### Editing-Related Works
1. [sail-sg/EditAnything](https://github.com/sail-sg/EditAnything) 
2. [IDEA-Research/Grounded-Segment-Anything](https://github.com/IDEA-Research/Grounded-Segment-Anything) 

#### Segmentation-Related Works
3. [maxi-w/CLIP-SAM](https://github.com/maxi-w/CLIP-SAM)  
4. [Curt-Park/segment-anything-with-clip](https://github.com/Curt-Park/segment-anything-with-clip)  
5. [kadirnar/segment-anything-video](https://github.com/kadirnar/segment-anything-video) 
6. [fudan-zvg/Semantic-Segment-Anything](https://github.com/fudan-zvg/Semantic-Segment-Anything) 
7. [continue-revolution/sd-webui-segment-anything](https://github.com/continue-revolution/sd-webui-segment-anything) 
8. [RockeyCoss/Prompt-Segment-Anything](https://github.com/RockeyCoss/Prompt-Segment-Anything) 
9. [ttengwang/Caption-Anything](https://github.com/ttengwang/Caption-Anything)  
10. [ngthanhtin/owlvit_segment_anything](https://github.com/ngthanhtin/owlvit_segment_anything)   
11. [lang-segment-anything](https://github.com/luca-medeiros/lang-segment-anything)
12. [helblazer811/RefSAM](https://github.com/helblazer811/RefSAM)  
13. [Hedlen/awesome-segment-anything](https://github.com/Hedlen/awesome-segment-anything)  
14. [ziqi-jin/finetune-anythin](https://github.com/ziqi-jin/finetune-anything)


## Todo
1. We plan connect segment-anything with [MaskCLIP](https://github.com/chongzhou96/MaskCLIP).  
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
