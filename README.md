# The Impact of Multidimensional Opinions and Recommendation Algorithms on Social Network Polarization

*Author: Isabela Tellechea Coluccio*

Code for the investigation regarding my Master's thesis in Computer Science at Universidad de Chile.

## Repository Structure

- `md_network.py` - MDNetwork class and matrix operations
- `exp1_convergence.py` - Experiment 1: Convergence analysis
- `exp2_init_conds.py` - Experiment 2: Initial conditions
- `exp3_min_dsg.py` - Experiment 3: Recommendation algorithm
- `exp3_preprocessing.py` - Creation of a network from Reddit data
- `utils.py` - Helper functions
- `REDDIT_data/` - Original data from Reddit 
- `paper_code/` - Code for the paper, where Experiments 2 and 3 are repeated with a controlled cross-topic coupling
  - `exp_coupling.py` - Experiments 2 and 3 (random networks and Reddit) for different coupling values
  - `res_coupling.py` - Summaries and figures from the results of the experiments
  - `star_counterexample.py` - Counterexample for the convergence in Euclidean norm
  - `coupling_sweep.py` - Numerical example of uniform coupling between topics
  - `results/` - Results (JSON) and logs of the experiments

## Resources

- [Experiments (Colab Notebook)](https://colab.research.google.com/drive/1jxKRrGIeockdARAMv2-IczzrcJXREI6A?usp=sharing)
- [Figures and Data (Google Drive)](https://drive.google.com/drive/folders/13tYcMjCMIf5J3XI_sHIzHiG7beeU-U8g?usp=share_link)
- [Reference Code (Github Repository)](https://github.com/uthsavc/filter-bubbles-polarization)
- [Reddit Data Source (Github Repository)](https://github.com/arnestc/political-compass)
