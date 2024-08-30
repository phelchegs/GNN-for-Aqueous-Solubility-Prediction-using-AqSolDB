# GNN-for-Aqueous-Solubility-Prediction-using-AqSolDB
A graph neural network based on torch geometric for aqueous solubility prediction
* Predict aqueous solubility of organic molecules using graph neural network, i.e., node (atom), edge (bond), and stacking layers, instead of chem descriptors
* Descriptors have been presented in the repo aqueous solubility prediction. Would like to compare and draw conclusion
* Challenges:
  - multiclass classificaiton
  - message passing includes not only node features but also edge's
  - use wandb for hyperparameters sweeping
** Dependencies **
