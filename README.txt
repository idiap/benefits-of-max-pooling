
This folder contains the supplementary material for the paper
'Benefits of Max Pooling in Neural Networks: Theoretical and Experimental Evidence'
TMLR 2023

Kyle Matoba
Machine Learning Group, Idiap
Computer Science Department, École polytechnique fédérale de Lausanne

Nikolaos Dimitriadis
Computer Science Department, École polytechnique fédérale de Lausanne

François Fleuret
Computer Science Department, University of Geneva
Machine Learning Group, Idiap
Computer Science Department, École polytechnique fédérale de Lausanne

`statistical_sorting_plot.py` generates Figure 1.

`empirical_error.py` does the main experiments, with
 `--multirun "++prng.seed=range(0, 10)" ++model.model_name=bigapprox,mediumapprox,smallapprox`
for all of the following 
 ` ++experiment.ident=initialization-xavier ++training.initialization_name=xavier`
 ` ++experiment.ident=initialization-kaiming ++training.initialization_name=kaiming`
 ` ++experiment.ident=criterion-l2 ++training.optimize_criterion_name=l2`
 ` ++experiment.ident=optimizer-adamw ++optimizer.optimizer_name=adamw`
 ` ++experiment.ident=data-sobol ++data.dataset_name=unitcube_sobol`
 ` ++experiment.ident=data-dirichlet ++data.dataset_name=unitcube_dirichlet`
 ` ++experiment.ident=dataset_size100 ++data.num_rows=100`
 ` ++experiment.ident=dataset_size500 ++data.num_rows=500`
 ` ++experiment.ident=dataset_size1_000 ++data.num_rows=1_000`
 ` ++experiment.ident=dataset_size5_000 ++data.num_rows=5_000`
 ` ++experiment.ident=dataset_size10_000 ++data.num_rows=10_000`
 ` ++experiment.ident=dataset_size20_000 ++data.num_rows=20_000`
 ` ++experiment.ident=dataset_size50_000 ++data.num_rows=50_000`
 ` ++experiment.ident=dataset_size100_000 ++data.num_rows=100_000`

(see the Hydra documentation for this calling syntax: https://hydra.cc/docs/advanced/override_grammar/basic/#modifying-the-config-object)
`plot_empirical_errors.py` produces Figures 2, 3, and 4 given the outputs of this experiment.
`num_parameters_plot.py` produces Figures 5 and 6.
`average_difference_plot.py` produces Figure 7

The code to replicate the results on adversarial robustness are at

https://github.com/nik-dim/maxpooling

and are separately documented there.

'networks.py' implements in idiomatic pytorch the algorithm
described in Appendix E to write a general $R$-estimator as a
pytorch sequential of alternating linear and relu layers.

Some paths will need to be set in an obvious way. One high level
entrypoint is suggested by `utils/path_config.py`

Gurobi is required, an academic license is available at
https://www.gurobi.com/downloads/end-user-license-agreement-academic/
