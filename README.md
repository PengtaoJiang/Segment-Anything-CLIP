# Conect Segment-Anything with CLIP
We aim to classify the output masks of [segment-anything](https://github.com/facebookresearch/segment-anything) with the off-the-shelf [CLIP](https://github.com/openai/CLIP) models. The cropped image corresponding to the image is sent to the CLIP model.
<img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/imgs/pipeline.png" width="100%" height="50%">

## Run Demo
```
sh run.sh
```

## Example 
Input an example image and a point to the SAM model. The input image and output three masks as follows:
<center><img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/imgs/ADE_val_00000001.jpg" width="50%" height="50%"></center>
The three masks and corresponding predicted category are as follows:
<div align=center><img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/outs/ADE_val_00000001/0.jpg" width="50%" height="50%"></div>
<div align=center><img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/outs/ADE_val_00000001/1.jpg" width="50%" height="50%"></div>
<div align=center><img src="https://github.com/PengtaoJiang/SAM-CLIP/blob/main/outs/ADE_val_00000001/2.jpg" width="50%" height="50%"></div>