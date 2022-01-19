# Model-Based Counterfacual Synthesizer Using Tabular GAN and a Comparison to Algorithmic Based Counterfactuals


This is a Github repository for the code associated with Arne Rustad's project thesis.  More information abot scripts and structure will be available soon. Below we describe briefly what each code file's purpose is and give the abstract for the written project thesis titled *"Model-Based Counterfacual Synthesizer Using Tabular GAN and a Comparison to Algorithmic Based Counterfactuals"*


## Code file explanations





## Project thesis abstract

> Counterfactual explanations are an emerging method for explaining predictions from black box
models by utilizing "what-if" scenarios. In this thesis, we define a model-based counterfactual synthesizer, tabGAN-cf, utilizing generative adversarial networks (GAN) and a post-processing step consisting of Monte Carlo sampling in combination with filtering. During the creation of tabGAN-cf, we also invent three different GAN based data synthesizers: tabGAN, tabGAN-qt and tabGAN-qtr. The two latter methods rely on quantile transformation and for the case of tabGAN-qtr, a novel stochastic version of quantile transformation that we introduce in this thesis. We compare the datasynthesizers against three state-of-the-art methods (TGAN, CTGAN and tabFairGAN). The results suggest that tabGAN, tabGAN-qt and tabGAN-qtr perform on par with or better than state-of-the-art methods for the experiments in this thesis. The implementation of the newly defined data synthesizers also run substantially faster than TGAN and CTGAN. We informally compare the newly created model-based counterfactual method, tabGAN-cf, to a well-known algorithmic-based method, MOC. The informal evaluation reveals a substantial gap between the two methods in favor of tabGAN-cf with respect to time efficiency and the desired prediction outcome, whilst MOC appears superior with respect to generating sparse counterfactual explanations.

