# Vaccination Classification

![Vaccination Bottles](https://github.com/cschneck7/phase_3_project/blob/main/images/vaccine_bottles_shrunk.jpg)

#### Links

[Notebook](H1N1_Vaccine_Classification_Notebook.ipynb)<br/>
[Presentation](Presentation.pdf)<br/>
[Data](https://www.drivendata.org/competitions/66/flu-shot-learning/data/)<br/>
[Feature Descriptions](https://github.com/cschneck7/phase_3_project/blob/main/data/H1N1_and_Seasonal_Flu_Vaccines_Feature_Information.txt)

## Overview

Pandemics have occured throughout history, each time affecting a large portion of the population. The most recent case being the Covid-19 2020 outbreak which resulted in a prolonged change in peoples day to day life, a lack or resources, and ultimately many deaths all around the world. If possible governments will do their best to try and prevent any future outbreaks by observing and learning from past information.

Therefore our stakeholder, a government agency, is trying to plan for future pandemic prevention and awareness by using the 2009 H1N1 pandemic as an example. They would like us to observe which features of a survey completed at the time appear to hold the highest importance in people who did not recieve the H1N1 vaccine. With this information they are hoping to be able to concentrate their efforts to provide vaccination information and flu prevention methods to a group that was otherwise more succeptible to contracting the virus. These efforts will be performed in hopes of increasing the vaccination rates and to help limit the spread of future viruses.

## Business Understanding

Our stakeholders goal is to help in the prevention of future pandemics by spreading vaccination and prevention information. To increase the value of their efforts they would like to target the population that are less inclined to receive the vaccination. The most successful outcome would be the prevention of a widespread outbreak resulting in a pandemic, but more realisticly an increase in vaccination awareness that results in a higher rate of people receiving vaccinations in case another outbreak occurs.

Therefore this projects requirements are to define the target audience who are less inclined to receive a vaccination. In order to do so a dataset on people who have both received and refrained from receiving vaccinations in the past is required. Available to us is a survey conducted for the 2009 H1N1 pandemic. The H1N1 outbreak was first detected in the United States and quickly spread across the rest of the world resuliting in between 151,000 and 575,000 deaths worldwide. Unlike prior strains of the H1N1 virus, people under 65 were more affected than the older population. Around 80 percent of the deaths assumed caused by this strain of H1N1 were people under the age of 65. Since this strain differed from previous strains the seasonal flu vaccinations didn't offer protection from the virus causing a late production of a vaccine that would be affective. An affective vaccination didn't get mass produced until after a second wave of the virus had come and gone in the United States <a href="#h1n1_cdc_article">[1]</a>.

Due to these factors this dataset may stray from a typical case, reason being:

1. <strong>The late emergence of the mass produced vaccine.</strong> It wasn't until after the second outbreak had passed that it was available, possibly causing people to assume the worst was over and a vaccination wasn't required and lowering the number of vaccinations.

2. <strong>The age group most affected were people under 65.</strong> This isn't typical for outbreaks and may have caused the number of vaccinations in this age group to be inflated.

Though our dataset may not reflect these two points, they should be kept in mind while analyzing the results. As far as metrics to qualify our models performance, I believe accuracy and precision should be used. A result of having high precision will ensure that our feature importance will strongly relate to true positive entries. Though there should be a fine balance and recall shouldn't be completely forgone. While we don't want our predicted positives to contain many false positives, we would like to predict a good portion of our true positive entries. Approaching our problem by utilizing our metrics in this manner can assure that our stakeholder is targeting the correct audience.

## Data Understanding

As mentioned earlier the data used for this analysis will be a 2009 survey conducted for the H1N1 outbreak. This survey was performed by the CDC in order to monitor and evaluate the flu vaccination efforts of adults and children in randomly selected US households. The questions asked of the participants were related to age, race, education, H1N1 vaccination status, flu-related behaviors, opinions about vaccine safety and effectivenss, recent respiratory illness, and pneumococcal vaccination status <a href="#About the National Immunization Survery">[2]</a>.

The dataset started with 26,707 entries and 35 features. There was a good split between vaccinated and unvaccinated entries with 21% of the entries being vaccinated.

## Data Preparation

The dataset contained many missing values. This was handled by omitting features and rows which were missing the majority of their entries. The leftover missing values were then filled using an iterative decision tree classifier assisted with random imputation. After this was performed high collinearity between features was dealt with by dropping features of similar questions. Lastly the categorical features were one hot encoded and scaling was performed.

## Modeling

The final model utilized a gradient boosting classifier called XGBoost. Our model prioritized precision, with the resulting scores for predicting unvaccinated respondent's:

<ul>
  <li>Precision: 86%</li>
  <li>Accuracy: 84%</li>
  <li>Recall: 95%</li>
  <li>F1-score: .90</li>
</ul>

Therefore our model predicted 84% of the test set correctly, had 86% accuracy for respondent's predicted unnvaccinated, and correctly predicted 95% of all unvaccinated respondent's.

The features holding the most importance in order were `doctor_recc_h1n1`, `opinion_h1n1_vacc_effective`, and `opinion_h1n1_risk`. This agreed with our inital exploratory data analysis where 53.3% of those who were recommended the vaccine received the vaccination. Also for those who believed strongly in the vaccines effectiveness and H1N1 virus risk had a 40.5% and 51% vaccination rate relatively.

## Conclusion

Observing the features that hold the most impact on the respondent's vaccination status, the conclusion that general practitioners hold the most weight in one's decision to be vaccinated can be obtained. As stated 53.3% of people with a doctors recommendation received the vaccine, compared to 13.6% of people who were not. Also it may be assumed that people receive information on vaccine effectiveness and virus risk from their personal health care physicians. Therefore I recommend reaching out to general practicioners with general information about vaccines or viruses to then transfer to their patients. This could possibly be in the form of pamphlets, posters or other sources of media. It should also be shown to the general practicioners how important their recommendation is in the outcome of their patient's decision. Only 22% of the respondant's in the survey were recommended the vaccine, yet more than half of those who were vaccinated were a part of that group.

## Future Improvements

<strong>Take a deeper look into the features that showed great importance.</strong> This includes finding subgroups in those features to help narrow down the target audience. For example we found out that those who believed the vaccination was effective had a much lower chance of being vaccinated, but could this be narrowed down to find the age group or other features that filled these sub groups more than others in order to find another group to target.

<strong>Fill out missing data about health insurance.</strong> A lot of entries were missing information on respondant's having access to health insurance so this feature was ommited. This feature may take a very important role in someone's decision to be vaccinated. As found getting a doctor's reccomendatin greatly improved the rate of someone recieving this vaccination, therefore if more people had access to primary physicians the population that recieves recommendations will increase and in turn the vaccination rate.

## References

[1] <a id='h1n1_cdc_article' href='https://www.cdc.gov/flu/pandemic-resources/2009-h1n1-pandemic.html'>https://www.cdc.gov/flu/pandemic-resources/2009-h1n1-pandemic.html</a>

[2] <a id='About the National Immunization Survery' href="https://webarchive.loc.gov/all/20140511031000/http://www.cdc.gov/nchs/nis/about_nis.htm#h1n1">https://webarchive.loc.gov/all/20140511031000/http://www.cdc.gov/nchs/nis/about_nis.htm#h1n1</a>

[3] <a href='https://www.drivendata.org/competitions/66/flu-shot-learning/data/'>https://www.drivendata.org/competitions/66/flu-shot-learning/data/</a>

[4] <a id='Missing Data Imputation using Regression' href='https://www.kaggle.com/code/shashankasubrahmanya/missing-data-imputation-using-regression'>https://www.kaggle.com/code/shashankasubrahmanya/missing-data-imputation-using-regression</a>

## Repository Structure

├── data<br/>
│ ├── models<br/>
│ └── H1N1_and_Seasoned_Flu_Vaccines_Feature_Information.txt<br/>
├── images<br/>
├── python_code<br/>
│ ├── **init**.py<br/>
│ ├── my_classes.py<br/>
│ └── my_functions.py<br/>
├── H1N1_Vaccine_Classification_Notebook.ipynb<br/>
├── Presentation.pdf<br/>
├── gitignore<br/>
└── README.md<br/>
