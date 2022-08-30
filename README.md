# Master thesis - Tabular GAN for data synthesis and counterfactual explanations - Arne Rustad


This is a Github repository for the code associated with Arne Rustad's master thesis. Below we describe briefly the github project layout and give the abstract for the written master thesis titled *"tabGAN: A Framework for UtilizingTabular GAN for Data Synthesizing and Generation of Counterfactual Explanations"* in first english and then norwegian.


## Github project layout explanation

1. **data**: This folder contains mostly unprocessed versions of the datasets used in the analysis
3. **helpers**: This folder contains a bunch of helper functions
    1. **compare**: This folder contains scripts with functions for comparing the marginal histograms of a synthesized dataset against the marginal histograms of a real dataset.
    2. **compare**: This folder contains scripts for comparing data synthesizers.
        - *synthesize_datasets_for_ml_comparison.py*: This file contains a single helper function that as input a data synthesizer and for each desired real dataset, creates a specific number of synthetic datasets that is saved where desired. Is used as a helper function when comparing different data synthesizers.
        - *eval_ml_efficacy_comparison*: Contains one function definition for fitting machine learning models to each of the synthesized datasets and real datasets, and one function definition for tidying up the returned results from the previous function as desired. Is split into two functions since the first function can be quite time consuming, whilst the second one is quite fast, but might be desirable to be called multiple times for different overviews.
        - **setup**: This folder contains a single file with a single function for creating and saving different train/test splits of the different datasets. This is done to decrease randomability and ensure a fair setup when comparing the different data synthesizing methods.
    3. **eval**: This folder contains files with different functions used for evaluation purposes. For instance evaluating synthesizing speed, fitting and evaluating a machine learning to a dataset or evaluating a data synthesizing method.
    4. **hp_tuning**: This folder contains different helper functions used during hyperparameter tuning of the tabGAN data synthesizing method presented in this master thesis. Contains helper functions both for both running the data synthesis for different hyperparameter combinations as well as helper functions for plotting and summarizing the results visually.
    5. **nmi_matrix**: This folder contains functions for computing and evaluating NMI-matrices.
        - *compute_nmi_matrix.py*: Contains a function that computes the NMI matrix of dataset.
        - *compare_nmi_matrices.py*: Contains a function for visually plotting different NMI matrices side-by-side in a grid format. Also allows for plotting the difference between the NMI matrices of the synthetic datasets compared to the original dataset.
    6. *tabGAN_gen_multiple_datasets.py*: File with helper function for synthesizing and saving multiple datasets using the tabGAN synthesizer. Due to the creation of other more general helper functions that can be used on any data synthesizer, this is mostly used by the hyperparameter tuning helper functions.
3. **hp_tuning_scripts**: This folder contains jupyter notebook files for visualizing the different iterations of hyperparameter tuning of the tabGAN framework. The notebooks are labelled with tabGAN or ctabGAN if they contain the hyperparameter tuning for respectively regular or conditional tabular GAN.
4. **images**: This folder contains images. The saved images corresponding to each hyperparameter search iteration is stored in its own folder.
5. **notebooks**: This folder contains a range of jupyter notebooks for doing all sorts of things.
6. **slurm_jobs**: This folder contains SLURM scripts for running jobs on the Idun high performing computer group
7. **src**: This folder contains various Python and R scripts. Additionally, it contains a folder called **constants** which is used to set or infer various global constants.
8. **tabGAN**: This folder contains the code for the TabGAN data synthesizing class and the TabGANcf class used to create counterfactual explanations. The TabGANcf class inherits from the TabGAN class.
9. **utils**: This folder contains lots of different utility functions.

## Master thesis abstract

> Counterfactual explanations is an emerging method for explaining predictions from black-box models by utilizing "what-if" scenarios. In this thesis, we create a Wasserstein Generative Adversarial Network (WGAN) based tabular data synthesizing framework, tabGAN, and later we modify this WGAN framework to create a model-based counterfactual synthesizer framework, which we call tabGANcf. The counterfactual framework is more a proof-of-concept, while the data synthesizing framework is more complete, with a lot of customization available and default values based on extensive hyperparameter tuning. During the creation of the data synthesizing framework tabGAN, we also create a new type of transformation, which we include as a preprocessing option for the numerical variables in a dataset. The novel transformation is a stochastic version of quantile transformation, which we in this thesis name the Randomized Quantile Transformation. In addition to a regular WGAN implementation, the data synthesizing framework tabGAN also implements a WGAN with a conditional generator inspired by the CTGAN data synthesizer. The conditional architecture and training process aim to provide more representation for rare categories in imbalanced columns. <br/> <br/>
We compare six data synthesizing methods from the tabGAN framework against the state-of-the-art data synthesizer methods CTGAN, TVAE, CopulaGAN, GaussianCopula, and TabFairGAN. The comparison includes an evaluation of the recreated marginal and joint distributions of a real dataset, as well as a comparison of machine learning efficacy on four real datasets. The methods from the tabGAN framework consistently outperform the other data synthesizing methods. Additionally, the methods from the tabGAN framework run substantially faster than the other GAN based data synthesizers in the evaluation, around 3-4 times faster than CTGAN and CopulaGAN for the real dataset used in the training time comparison. The comparison also indicates that the novel transformation method, the Randomized Quantile Transformation, is very beneficial for dataset variables with many repeated values. We visually verify that methods from the tabGANcf framework are able to generate counterfactual explanations that change the predictions of a black-box classifier whilst not making unnecessary changes to the discrete variables. Sparsity of proposed changes in the numerical variables is, however, still an issue. We propose a potential solution for this that can be investigated in future research. We also provide a list of other extensions that can be implemented in future GAN based counterfactual synthesizers.

## Sammendrag av masteroppgave

> Kontrafaktiske forklaringer er en metode for å forklare prediksjoner fra black-box modeller gjennom henvising til alternative lignende virkeligheter der et annet utfall skjer. Denne eksempel-baserte forklaringsmetoden er i ferd med å bli et populært hjelpemiddel innen forklaring av avanserte AI-modeller. I denne avhandlingen presenterer vi et rammeverk for å syntetisere tabelldata ved hjelp av Wasserstein Generative Adversarial Networks (WGAN) som vi navngir tabGAN. I tillegg modifiserer vi dette rammeverket til også å kunne lage modell-baserte kontrafaktiske forklaringer. Det nye rammeverket for generering av kontrafaktiske forklaringer kaller vi for tabGANcf, selv om det i praksis er mer et pilotprosjekt for å synliggjøre at man kan lage kontrafaktiske forklaringer på denne måten. Rammeverket for syntetisering av tabelldata, tabGAN, er imidlertid langt mer komplett. Det tilrettelegger for stor individuell brukstilpasning, og de anbefalte standardverdiene i rammeverket er basert på et omfattende hyperparametersøk. Gjennom arbeidet med rammeverket tabGAN fikk vi ideen til en ny type transformasjon som kan brukes som preprosesseringsmetode for numeriske variabler. Vi kaller den for randomisert kvantiltransformasjon (the Randomized Quantile Transformation), ettersom den er en stokastisk versjon av kvantiltransformasjon. I tillegg til en mer standard WGAN versjon implementerer vi et WGAN med betinget generator, inspirert av datasyntetiseringsmetoden CTGAN. Tanken bak den betingede arkitekturen og spesialtilpassede treningsprosessen er å være bedre i stand til å representere de sjeldne kategoriene for ubalanserte diskrete kolonner, slik at det blir lettere for generatoren å lære seg å gjenskape de også riktig. <br/> <br/>
Vi sammenligner seks datasyntetiseringsmetoder fra rammeverket tabGAN mot de beste datasyntetiseringsmetodene innen dette feltet, som CTGAN, TVAE, CopulaGAN, GaussianCopula og TabFairGAN. I sammenligningen evaluerer vi hvor godt datasyntetisererne er i stand til å gjenskape både marginal- og simultanfordelinger fra et reelt datasett. I tillegg evaluerer vi hvor godt maskinlæringsmodeller trent på syntetisk data fra hver datasyntetiserer presterer på ett reelt testdatasett, sammenlignet med hvor godt samme modell trent på det originale treningsdatasettet gjør det. Dette gjentar vi for fire forskjellige reelle datasett. Metodene fra tabGAN-rammeverket gjør det konsekvent bedre i disse evalueringene sammenlignet med de andre datasyntetiserermetodene. I tillegg til dette er metodene fra tabGAN-rammeverket raskere å trene enn de andre datasyntetisererne som også er basert på GAN. Resultatene indikerer videre at randomisert kvantiltransformasjon er svært nyttig for numeriske datasettvariable med mange repeterte verdier. For det kontrafaktiske rammeverket tabGANcf utfører vi en visuell verifisering av at metodene fra rammeverket er i stand til å generere kontrafaktiske forklaringer som faktisk endrer prediksjonene til en black-box klassifiseringsmodell. Vi observer at de ulike metodene fra rammeverket fint klarer å endre prediksjonene uten å gjøre unødvendige endringer på diskrete variable, men at dette ikke er tilfellet for de numeriske variablene. I denne avhandlingen foreslår vi en mulig løsning både på dette problemet og andre potensielle utfordringer vi identifiserer. Grunnet tidsbegrensninger overlater vi testing av disse forslagene til fremtidig forskning.
