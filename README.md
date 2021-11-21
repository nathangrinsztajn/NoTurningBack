This is a Pytorch implementation of the paper: Neurips 2021 - There Is No Turning Back: A Self-Supervised Approach for 
Reversibility-Aware Reinforcement Learning.
  
  
## Requirements  
  
 * Python 3.8
 * For the other packages, please refer to the requirements.txt, or do
 
```
pip install -r requirements.txt
```

### Training RAC

RAC can be trained on Cartpole using rac_cartpole.py, or on Turf using rac_turf.py.
The parameter n_traj_classifier controls the number of trajectories used to train psi, 
the parameter *epoch_classifier* controls the number of training epochs of psi, and *steps_action_model* the number of
 training examples given to phi.
 
```  
python rac_cartpole.py
python rac_turf.py --epoch_classifier 100 --steps_action_model 100000 --n_traj_classifier 50000
```

### Training RAE

RAE can be trained on Cartpole using rae_cartpole.py, or on Turf using rae_turf.py. The parameter *threshold* is denoted 
beta in the paper. The online training of psi frequency is fixed using *train_freq*. The window *w* is controled using 
*d_max*. 

```  
python rae_cartpole.py
python rae_turf.py --threshold 0.8 --train_freq 500 --d_max 50000
```
