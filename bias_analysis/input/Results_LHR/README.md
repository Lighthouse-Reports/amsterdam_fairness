# 20240308 Results LHR for bias analysis results Slimme Check

## Background
- This output is created for Lighthouse Reports (LHR) by de Gemeente Amsterdam (GA).
- It concerns results of de 'Slimme Check' project, a project aimed at developing a model that aids in the process of determining which welfare applications are to be subjected to more thorough research by enforcements specialists.
- For information about the project see the Algoritmeregister by GA: https://algoritmeregister.amsterdam.nl/onderzoekswaardigheid-slimme-check-levensonderhoud/

## Purpose
- The purpose of the output is for LHR to conduct their own (bias) analysis on the data, methodology, and results of the project and has a journalistic purpose.
- The purpose is not to support the GA regarding the Slimme Check.

## Current status Slimme Check
- The alderman of the GA has decided that the Slimme Check will NOT go to production, nor will development on the model continue.
- For more information see the Algoritmeregister by GA: https://algoritmeregister.amsterdam.nl/onderzoekswaardigheid-slimme-check-levensonderhoud/
- The lessons learned will be documented and used to further improve best practices in future comparable projects.

## Caveats/Limitations
- There are some noteworthy caveats in these results that should be taken into account.
- The most important reasons for these caveats are:
    - The Slimme Check model was initially developed in a VAO (Veilige Analyse Omgeving/Safe Analysis Environment) and the project was later migrated to Azure.
        - This VAO was removed as it was no longer necessary for development.
        - While the migration was successful, this made it more difficult to recreate the work from this period.
    - The Slimme Check was developed in Azure SDK v1. Since, the Azure SDK v2 was released and v1 is no longer supported within the Slimme Check workspace.
        - In order to recreate the results as they were created during the pilot, the codebase will have to be rewritten to SDK v2. 
        - Given the purpose of the LHR output, and the fact that there will be no further development of the Slimme Check, this was deemed too demanding.
        - The output was recreated outside of Azure, but due to this, it is plausible there are minor differences in methodology compared to the Azure pipelines, causing minor differences in the results.
            - These are most likely due to filtering/ordering/rounding differences.
- These caveats have been discussed with LHR and deemed not to be problematic for their analysis.

### Group size < 10
- Wherever the group size of any group is < 10, it is reduced to 0.
- This is in accordance with guidelines that prevent results being traced back to individuals
- For the confusion matrix, if there was only 1 metrics <10, the 2 lowest metrics were reduced to prevent being able to deduct the smallest group size from the total group size.
    - See code Final_Preprocessing_LHR_Data.ipynb for method used.

### Train/test split
- The order of the dataset is of importance for the resulting split, but since this split was made outside of the Azure pipeline, the ordering before the split is unlikely to be the same as used when training the model.
    - This means the split is not equal to the one used in the project.

### Model scores
- The applications are scored by both the model before and after reweighing.
- When comparing the scores to the scores that were found in e.g. the pilot results, they had a slight deviation (<1%).
    - The most plausible explanation was rounding differences.

### Bias metric results
- For the prepilot dataset, the bias metrics results in this output deviate from the results in the bias analysis conducted by the GA.
    - The most plausible explanation was a difference in filtering for the dataset.

### Most important features, dataset selection
- For the most important features, only the applications where these were actually calculated and communicated to enforcement during the project are calculated for this output.
- The most important features were communicated in order of importance to enforcement.
    - This order is removed in these results because it led to too small group sizes.
    - The result is the amount of times a feature was 1st, 2nd, or 3rd most important for a specific prediction.