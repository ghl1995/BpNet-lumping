# BpNet-lumping

Biological events often occur at microseconds timescale, and while simulation tools and computation resources had a large advancement in the past few decades. Getting a reliable sampling of these long-timescale events is still challenging.
Markov State Model is a mathematical framework that allows the estimation of timescales and other long timescale properties through a large number of simulations. By splitting the phase space into “states” and consider the dynamics within microstate to be memoryless, one can extrapolate long timescale dynamics through repeated propagation.
p(mτ)=T(τ)^m p(0)
However, the splitting of phase space itself is a tricky task, as one has to strike a balance between “relaxation of memory” and “ease of interpretation”.  If we split the phase space into very small regions, the dynamics within each “state” (phase space volume) would be more homogeneous, the “memory” of the state will relax much faster and it is easier for the overall model to be Markovian. But splitting into very small states will result in a large number of states, and although this would favor MSM construction, it at the same time hinders interpretation of the model, as it is challenging to handle or visualize a large number of states.
Often, for the sake of understanding, we would make a compromise and lump these microstates into larger states (only a handful of states, say less than 10), to facilitate visualization and understanding. However, larger states means that are more inhomogeneity within each state (or “memory”), and so it is harder to make the model Markovian, and thus an accurate state assignment is necessary as poor state asignment boundary will often result in non-Markovian evolutions.

In this work, we have proposed a new lumping method that is based on an objective function with a much stronger physical basis. Inspired by the projection operator framework used in statistical mechanics to analyze the dynamics of a complex system, we have formulated a reverse projection scheme to allow comparison between the dynamics of lumped macrostates and original microstates, which allows quantification of the quality of lumping. To efficiently find a good lumping, we have also constructed a neural network and take the use the reverse projection score as the loss, which we called BPnet. We will demonstrate that our proposed scheme indeed perform well from simple numerical system to real biological systems. We have offered a different viewpoint on the lumped dynamics of Markovian systems, and we also foresee that our approach is applicable to the study of many biological systems, by opening up the possibility of constructing kinetic network model with a few states without the need of long simulation trajectories.

# Implementation
python main_train.py --data_path --save_data_path --lr 0.01 --bs 2000

python main_GMRQ.py --data_path --save_data_path --lr 0.01 --bs 2000


