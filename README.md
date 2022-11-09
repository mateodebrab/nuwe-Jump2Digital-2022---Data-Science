## background
#### Air quality classification
The Paris Agreement is an international treaty on climate change that was adopted by 196 Parties at COP21 in Paris. Its goal is to limit global warming to well below 2, preferably 1.5 degrees Celsius, compared to pre-industrial levels.  
To reach this long-term temperature goal, countries aim to peak greenhouse gas emissions as soon as possible to achieve a climate-neutral planet by mid-century.  
That is why the European Union is allocating large amounts of resources to the development of new technologies that allow the improvement of the fight against pollution. One of these is a new type of sensor based on laser technology that allows air quality to be detected based on different sensors.
***
## problem
The objective is to carry out a predictive model that allows knowing the type of air quality based on the measurements of the sensors.  
0 corresponds to Good air quality  
1 corresponds to Moderate air quality  
2 corresponds to Dangerous air quality  
We face a multiclass classification problem.
***
## analysis
We have 8 features, all numerical, corresponding to the parameters measured by the different sensors. The training dataset has 2100 records and the test dataset 900. The distribution between the 3 labels is balanced: 33% of the records for each label. The features are standardized. We normalize them too to put them all on the same scale from -1 to 1. 3 features have almost no correlation with the target: feature7, feature8, feature4. The feature importance graph shows us the same thing. The decision is made not to remove them because 2 percentage points of f1_score are lost by removing them. 
***
## solution

***
## results
***
## license
***
