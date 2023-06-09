# S6 Assignment Part 1
Bakpropagation in a neural network demo:

![Back Propagation](https://github.com/sunpau/ERA_V1_S6/blob/main/images/BakPropagation.png)

# Effects of Learning Rate on Loss
-  As seen from the below plots, When the learning rate is small, the loss function takes a long time to decrease and reach a minimum point.
-  As we increase the learning rate, the incremental jump of the loss decrease is more and so, the loss function reaches the minimum point in a smaller number of steps.
-  But when the learning rate is too high, the incremental jump of the loss overshoots the minimum and the loss changes drastically (oscillates) indiacting the divergent behaviour.
![LR1](https://github.com/sunpau/ERA_V1_S6/blob/main/images/LR1.png)
![LR2](https://github.com/sunpau/ERA_V1_S6/blob/main/images/LR2.png)
![LR3](https://github.com/sunpau/ERA_V1_S6/blob/main/images/LR3.png)

# Excel File
In the Excel file, the Plots tab contains the loss plot. The effect of learning rate can be observed by changing the cell N36 in tab Data
