# Inside Amsterdam’s high-stakes experiment to create fair welfare AI
This repository includes our analysis into Amsterdam's Slimme Check algorithm. Slimme Check was an attempt by the city of Amsterdam to predict which welfare applications were likely to be fraudulent or contain errors. Stories based on this analysis have been published by 
- [Trouw](https://www.trouw.nl/verdieping/amsterdam-wilde-met-ai-de-bijstand-eerlijker-en-efficienter-maken-het-liep-anders~b2890374/)
- [MIT Tech]()
We also published an extensive [methdology]() on our website.

# Repo Overview

## bias_analysis
This contains confusion our bias analysis of the Slimme Check system. The city made confusion matrices available to us along a number of dimensions: version of the model (before and after reweighing), dataset (train, test, prepilot, and pilot), and applicant characteristic (gender, age, parenthood, migration-background). You can find the full dataset provided to us by the city [here](https://github.com/Lighthouse-Reports/amsterdam_fairness/blob/main/bias_analysis/input/Results_LHR/Output/20240308_CMs_LHR_SlimmeCheck.xlsx).
The script we used for the analysis is called [bias_analysis](https://github.com/Lighthouse-Reports/amsterdam_fairness/blob/main/bias_analysis/code/lhr_bias_analysis.Rmd) and the outputs are located in this [folder](erdam_fairness/tree/main/bias_analysis/output).

## TODO: model analysis

## City code
The city also shared the [repo](https://github.com/Lighthouse-Reports/amsterdam_fairness/tree/main/wpi_uitkeringsfraude) they used for model [training](https://github.com/Lighthouse-Reports/amsterdam_fairness/blob/main/wpi_uitkeringsfraude/wpi_onderzoekswaardigheid_aanvraag/entrypoints/train_model.py) and scoring[https://github.com/Lighthouse-Reports/amsterdam_fairness/blob/main/wpi_uitkeringsfraude/wpi_onderzoekswaardigheid_aanvraag/entrypoints/score_applications.py]. This repo also contains the city's extensive bias evaluation pipeline.
Since we do not have access to individual level data, we could not test the code in this repo ourselves.
