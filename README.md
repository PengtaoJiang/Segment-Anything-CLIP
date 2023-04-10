# Conect Segment-Anything with CLIP
We aim to classify the output masks of [segment-anything](https://github.com/facebookresearch/segment-anything) with the off-the-shelf [CLIP](https://github.com/openai/CLIP) models. The cropped image corresponding to the image is sent to the CLIP model.
<img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/imgs/pipeline.png" width="100%" height="50%">

## Run Demo
Download the [sam_vit_h_4b8939.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) model from the SAM repository. Then run the following script:
```
sh run.sh
```

## Example 
Input an example image and a point (int(w\*0.2), int(h\*0.5)) to the SAM model. The input image and output three masks as follows:
<center><img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/imgs/ADE_val_00000001.jpg" width="50%" height="50%"></center>

The three masks and corresponding predicted category are as follows:
<center>
<img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/outs/ADE_val_00000001/outs.png" width="100%" height="50%"> 
</center>
We crop each image patch and send to the clip model to get top-5 predicted categories:
