# Coarser data and more examples

We down samples the ECMWF data from 0.05 to 0.2. 
In previous experiments we used a 0.1 resolution, as this is the same as the live ECMWF data.

By reducing the resolution we can increase the number of samples we have to train on.
We used 41408 number of samples to train, and 10352 samples to validate
This is approximately 5 times more samples than the previous experiments.

## Experiments


### b8_s1
Batche size 8, with 0.2 degree NWP data. 
https://wandb.ai/openclimatefix/india/runs/w85hftb6


### b8_s2
Batch size 8, different seed, with 0.2 degree NWP data. 
https://wandb.ai/openclimatefix/india/runs/k4x1tunj

### b32_s3
Batch size 32, with 0.2 degree NWP data. Also kept the learning rate a bit higher
https://wandb.ai/openclimatefix/india/runs/ktale7pa

### epochs
We set the early stopping epochs from 10 to 15. This should mean model will train a bit more
https://wandb.ai/openclimatefix/india/runs/8hfc83uv

### small model
We made the model about 50% of the size by reduce the reducing the channels in the NWP encoder fomr 256 to 64 and reducing the hidden features in the output network fomr 1024 to 256
https://wandb.ai/openclimatefix/india/runs/sk5ek3pk


### early stopping on MAE/val
Changing from quantile_loss to MAE/val to stop early on. This should mean the model does more training epochs, and the results we are interested int.  
https://wandb.ai/openclimatefix/india/runs/a5nkkzj6


### old
Old experiment with 0.1 degree NWP data. 
https://wandb.ai/openclimatefix/india/runs/m46wdrr7.
Note the validation batches are different that the experiments above.

Interesting the GPU memory did not increase much better experiments 2 and 3. 
Need to check that 32 batches were being passed through. 

## Results

The coarsening data does seem to improve the experiments results in the first 10 hours of the forecast.
DA forecast looks very similar. Note the 0 hour forecast has a large amount of variation. 

Still spike results in the individual runs. 

| Timestep | b8_s1 MAE % | b8_s2 MAE % | b32_s3 MAE % | epochs MAE % | small MAE % | mae/val MAE % | old MAE % |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0-0 minutes | 0.052 | 0.047 | 0.027 | 0.030 | 0.041 | 0.041 | 0.066 |
| 15-15 minutes | 0.052 | 0.049 | 0.031 | 0.033 | 0.041 | 0.041 | 0.064 |
| 30-45 minutes | 0.052 | 0.051 | 0.037 | 0.039 | 0.043 | 0.043 | 0.063 |
| 45-60 minutes | 0.053 | 0.052 | 0.040 | 0.043 | 0.044 | 0.044 | 0.063 |
| 60-120 minutes | 0.056 | 0.054 | 0.048 | 0.052 | 0.048 | 0.048 | 0.063 |
| 120-240 minutes | 0.061 | 0.060 | 0.060 | 0.064 | 0.057 | 0.057 | 0.065 |
| 240-360 minutes | 0.061 | 0.062 | 0.063 | 0.065 | 0.061 | 0.061 | 0.065 |
| 360-480 minutes | 0.062 | 0.062 | 0.062 | 0.063 | 0.063 | 0.063 | 0.066 |
| 480-720 minutes | 0.063 | 0.063 | 0.062 | 0.064 | 0.064 | 0.064 | 0.065 |
| 720-1440 minutes | 0.065 | 0.066 | 0.065 | 0.067 | 0.066 | 0.066 | 0.066 |
| 1440-2880 minutes | 0.069 | 0.070 | 0.071 | 0.071 | 0.071 | 0.071 | 0.071 |


![](mae_step.png "mae_steps")

![](mae_step_smooth.png "mae_steps")

I think its worth noting the model traing MAE is around `3`% and the validation MAE is about `7`%, so there is good reason to believe that the model is over fit to the trianing set. 
It would be good to plot some of the trainin examples, to see if they are less spiky. 