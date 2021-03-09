# ResNet_35 layers
 Modify ResNet with 35 layers.
 Training on Cifar10.
## Architecture
### Residual block(full pre-activation)
<p align="left">
    <img src="/Residual block.png" width="400" height="680"/>
</p>

### Model Overall
See this [model](https://github.com/moddent/ResNet_customize/blob/main/model.png).


## Training
 Run the following script from the directory:
 
    python cifar10_rec_elu+relu.py
## Result
### Loss
<p align="left">
    <img src="/loss.png" width="640" height="480"/>
</p>

### Accuracy
<p align="left">
    <img src="/acc.png" width="640" height="480"/>
</p>

### Testing

<p align="left">
    <img src="/testing.png" width="600" height="330"/>
</p>
