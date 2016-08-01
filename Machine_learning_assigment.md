# Machine Learning Assigment
Harby Ariza  
17 July 2016  



## Background


Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: <http://groupware.les.inf.puc-rio.br/har> (see the section on the Weight Lifting Exercise Dataset).

When choosing predicting models , one of the most important topics to consider is how we predict errors. Predicting errors can be defined into two main subcomponents that we should care about: errors due to "bias" and errors due to "variance".There is a tradeoff between a model's ability to minimize bias and variance. Errors due to Bias is defined as the difference between the expected prediction of our model and the correct value we are trying to predict. Errors due to Variance is defined as the variability of a model prediction for a given data set. This two main subcomponents will be taken into account in order to choose our "Best Model". In this exercise we'll be executing a variety of classification algorithms such as trees , SVM , Random forest and Boosting. The last two in particular seems to be the top performers and due to this reason they are highly use in prediction competitions like Kaggle and others. Random Forest rely on bagging (Bootstrap Aggregating) and resampling techniques that are commonly use to reduce the variance in model predictions.


 

##Data

The training data for this project are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv>

The test data are available here:

<https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv>.





```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", destfile = 'D:\\R\\pml-training.csv')
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", destfile = 'D:\\R\\pml-testing.csv')
```




## Loading the Data and Cleanup 

One of the more important things to do before execute the classifier algorithm is to identify what columns in the data set are relevant and can become good predictors. So first we remove all columns containing NA's because they are usefull at all. Secondly something I did learn testing with these differents classiffiers algorithms is that some of them have some good capability/features that can be used to perform some sort of exploratory Analysis. For instance now I'll be executing a Random Forest classifier to the trainset but I want to list on the variable importance list provided by this model. Now as you can see below in the output of the varImp function there are a bunch of irrelevant variables that appears at the top of the list. For instance the variable X plus others containing Time-Stamp information. They offer no value so I'll remove them from the trainset dataset.



```r
pmlTrain<-read.csv("D:\\R\\pml-training.csv", header=T, na.strings=c("NA","#DIV/0!"))
pmlTest<-read.csv("D:\\R\\pmltesting.csv", header=T, na.string=c("NA", "#DIV/0!"))

pmlTrainNoNa<-pmlTrain[, apply(pmlTrain, 2, function(x) !any(is.na(x)))]       ## remove columns with NA's 
TidyTrain<-pmlTrainNoNa
inTrain <- createDataPartition(y=TidyTrain$classe,p=0.6, list=FALSE)
trainset <- TidyTrain[inTrain,]
testset <- TidyTrain[-inTrain,]
dim(trainset);dim(testset)
```

```
## [1] 11776    60
```

```
## [1] 7846   60
```


```r
fitControl <- trainControl(method = "cv",number = 5,allowParallel = TRUE)
forestDataTest.fit <- train(classe~ .,data=trainset,method="rf",prox=TRUE,allowParallel = TRUE,trControl = fitControl,importance=TRUE)
varImp(forestDataTest.fit)
```

```
## rf variable importance
## 
##   variables are sorted by maximum importance across the classes
##   only 20 most important variables shown (out of 81)
## 
##                                      A       B        C       D       E
## X                              40.5934 50.2892 100.0000 49.8582 14.9689
## roll_belt                       1.8715  3.8822   4.6947  4.1700  2.1193
## raw_timestamp_part_1            1.2110  1.3334   1.8507  2.0296  2.5504
## pitch_forearm                   1.0208  1.4167   1.9882  2.0948  1.8781
## accel_belt_z                    0.9463  1.3751   1.9162  1.8146  1.0365
## num_window                      0.6699  1.5726   1.8063  1.7564  1.8334
## cvtd_timestamp02/12/2011 14:57  0.3216  0.5380   1.6849  1.6988  1.6639
## roll_dumbbell                   0.3285  0.5487   1.3769  1.6362  1.1094
## magnet_dumbbell_y               0.9307  0.7915   1.4305  1.3711  1.1215
## accel_forearm_x                 0.4405  0.5585   0.7869  1.1177  1.2836
## magnet_belt_y                   0.9043  1.1143   1.2649  1.1513  1.1885
## cvtd_timestamp30/11/2011 17:12  0.4920  1.0468   1.2424  0.8324  0.8737
## cvtd_timestamp28/11/2011 14:15  0.3216  0.9151   1.2385  0.8885  0.8617
## yaw_belt                        0.6620  0.7338   0.9915  1.1346  1.1562
## cvtd_timestamp02/12/2011 13:33  0.6131  0.7339   0.8432  0.8066  1.1131
## pitch_belt                      0.5658  0.6774   0.9931  1.0793  1.0527
## cvtd_timestamp30/11/2011 17:11  0.3216  0.3900   0.7375  1.0732  0.9262
## cvtd_timestamp05/12/2011 14:23  0.3216  0.6068   0.5985  0.8809  1.0413
## total_accel_belt                0.4913  0.7132   1.0302  0.9917  0.7034
## accel_dumbbell_y                0.3858  0.4854   0.8444  1.0250  0.7569
```



```r
TidyTrain<-pmlTrainNoNa[,-c(1:7)]   ## remove unnecessary variables
inTrain <- createDataPartition(y=TidyTrain$classe,p=0.6, list=FALSE)
trainset <- TidyTrain[inTrain,]
testset <- TidyTrain[-inTrain,]
table(trainset$class)
```

```
## 
##    A    B    C    D    E 
## 3348 2279 2054 1930 2165
```

```r
table(testset$class)
```

```
## 
##    A    B    C    D    E 
## 2232 1518 1368 1286 1442
```

```r
dim(trainset);dim(testset)
```

```
## [1] 11776    53
```

```
## [1] 7846   53
```


A decision tree model was generated using the code below. A cross-validation of K=5 was applied to the trainset dataset.



```r
set.seed(1234)
start.time <- Sys.time()
dtree.fit <- rpart(classe ~ ., data=trainset, method="class",control=rpart.control(xval=5),parms=list(split="information"))
end.time <- Sys.time();time.taken <- end.time - start.time;time.taken
```

```
## Time difference of 2.299 secs
```

```r
dtree.pred <- predict(dtree.fit, testset, type="class")
dtree.perf <- table(testset$classe, dtree.pred,
                    dnn=c("Actual", "Predicted"))
```





Here we can see the Accuracy of the decision tree model.

```r
confusionMatrix(dtree.perf)
```

```
## Confusion Matrix and Statistics
## 
##       Predicted
## Actual    A    B    C    D    E
##      A 1993  100   59   51   29
##      B  312  944  205   50    7
##      C   14  146 1094  110    4
##      D  114   40  160  938   34
##      E   94  201  281   87  779
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7326          
##                  95% CI : (0.7227, 0.7424)
##     No Information Rate : 0.3221          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6604          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7887   0.6597   0.6081   0.7589  0.91325
## Specificity            0.9551   0.9105   0.9547   0.9474  0.90519
## Pos Pred Value         0.8929   0.6219   0.7997   0.7294  0.54022
## Neg Pred Value         0.9049   0.9230   0.8912   0.9546  0.98844
## Prevalence             0.3221   0.1824   0.2293   0.1575  0.10872
## Detection Rate         0.2540   0.1203   0.1394   0.1196  0.09929
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639  0.18379
## Balanced Accuracy      0.8719   0.7851   0.7814   0.8531  0.90922
```

```r
performance(dtree.perf)
```

```
## Sensitivity = 0.75159
## Specificity = 0.95222
## Positive Predictive Value = 0.90421
## Negative Predictive Value = 0.86464
## Accuracy = 0.87698
```

A SVM model was generated using the code below. Cross-Validation of k=10 was applied to the trainset dataset.

```r
set.seed(1234)
start.time <- Sys.time()
svm.fit <- svm(classe~., data=trainset,cross=10)
end.time <- Sys.time();time.taken <- end.time - start.time;time.taken
```

```
## Time difference of 2.01825 mins
```

```r
svm.pred <- predict(svm.fit, na.omit(testset))
svm.perf <- table(na.omit(testset)$class,svm.pred, dnn=c("Actual", "Predicted"))
```

Here we can see the Accuracy of the SVM model.

```r
confusionMatrix(svm.perf)
```

```
## Confusion Matrix and Statistics
## 
##       Predicted
## Actual    A    B    C    D    E
##      A 2226    3    3    0    0
##      B  128 1321   64    1    4
##      C    4   44 1295   21    4
##      D    2    4  122 1157    1
##      E    0   17   39   31 1355
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9373          
##                  95% CI : (0.9317, 0.9426)
##     No Information Rate : 0.3008          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9205          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9432   0.9510   0.8503   0.9562   0.9934
## Specificity            0.9989   0.9695   0.9885   0.9806   0.9866
## Pos Pred Value         0.9973   0.8702   0.9466   0.8997   0.9397
## Neg Pred Value         0.9761   0.9893   0.9648   0.9919   0.9986
## Prevalence             0.3008   0.1770   0.1941   0.1542   0.1738
## Detection Rate         0.2837   0.1684   0.1651   0.1475   0.1727
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9711   0.9603   0.9194   0.9684   0.9900
```

```r
performance(svm.perf)
```

```
## Sensitivity = 0.91166
## Specificity = 0.99865
## Positive Predictive Value = 0.99773
## Negative Predictive Value = 0.94562
## Accuracy = 0.96438
```

A Random Forest model was generated using the code below. Cross-validation using k=5 was performed. 


```r
set.seed(1234)
start.time <- Sys.time()
fitControl <- trainControl(method = "cv",number = 5,allowParallel = TRUE)
forest.fit <- train(classe~ .,data=trainset,method="rf",prox=TRUE,allowParallel = TRUE,trControl = fitControl,importance=TRUE)
end.time <- Sys.time();time.taken <- end.time - start.time;time.taken
```

```
## Time difference of 20.14681 mins
```

```r
forest.pred <- predict(forest.fit, testset)
forest.perf <- table(testset$class, forest.pred,dnn=c("Actual", "Predicted"))
forest.perf
```

```
##       Predicted
## Actual    A    B    C    D    E
##      A 2230    2    0    0    0
##      B   19 1492    7    0    0
##      C    0    9 1354    5    0
##      D    0    0   11 1273    2
##      E    0    0    0    1 1441
```


```r
varImp(forest.fit)
```

```
## rf variable importance
## 
##   variables are sorted by maximum importance across the classes
##   only 20 most important variables shown (out of 51)
## 
##                            A     B     C     D     E
## yaw_belt             100.000 72.52 73.88 76.97 59.65
## pitch_forearm         66.566 80.76 98.32 55.53 73.95
## pitch_belt            28.777 96.76 60.25 47.10 48.66
## magnet_dumbbell_z     81.916 64.94 81.63 60.49 67.99
## magnet_dumbbell_y     65.497 55.88 71.62 55.70 51.07
## gyros_belt_z          27.862 41.02 34.19 30.74 44.41
## accel_forearm_x       21.352 33.61 29.98 43.09 32.02
## yaw_arm               34.300 27.96 25.95 27.84 22.67
## accel_dumbbell_y      27.814 23.78 33.80 25.56 29.93
## roll_dumbbell         21.259 33.23 27.53 28.75 30.23
## roll_forearm          32.630 28.51 33.00 18.69 31.02
## accel_belt_z          27.354 32.68 31.96 32.52 25.51
## gyros_dumbbell_y      30.434 22.31 31.79 22.04 20.25
## magnet_belt_z         23.273 29.12 19.83 31.45 24.55
## accel_dumbbell_z      25.150 29.31 22.95 26.25 31.22
## magnet_belt_x          9.333 27.81 27.01 18.72 30.70
## magnet_arm_z          20.314 29.86 21.85 18.97 16.50
## total_accel_dumbbell  14.732 22.43 19.43 19.39 28.73
## gyros_arm_y           23.295 27.66 18.10 24.96 16.51
## pitch_arm              5.347 26.92 18.70 17.55 13.79
```


```r
confusionMatrix(forest.perf)
```

```
## Confusion Matrix and Statistics
## 
##       Predicted
## Actual    A    B    C    D    E
##      A 2230    2    0    0    0
##      B   19 1492    7    0    0
##      C    0    9 1354    5    0
##      D    0    0   11 1273    2
##      E    0    0    0    1 1441
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9929          
##                  95% CI : (0.9907, 0.9946)
##     No Information Rate : 0.2866          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.991           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9916   0.9927   0.9869   0.9953   0.9986
## Specificity            0.9996   0.9959   0.9978   0.9980   0.9998
## Pos Pred Value         0.9991   0.9829   0.9898   0.9899   0.9993
## Neg Pred Value         0.9966   0.9983   0.9972   0.9991   0.9997
## Prevalence             0.2866   0.1916   0.1749   0.1630   0.1839
## Detection Rate         0.2842   0.1902   0.1726   0.1622   0.1837
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9956   0.9943   0.9924   0.9967   0.9992
```

```r
performance(forest.perf)
```

```
## Sensitivity = 0.98743
## Specificity = 0.9991
## Positive Predictive Value = 0.99866
## Negative Predictive Value = 0.99155
## Accuracy = 0.99439
```
A Boosting model was generated using the code below. 
Cross-validation is performed implicity by this model as well using
bagging and resampling techniques but anyway Cross-validation using k=5 was performed.The only drawback using this approach that might affect the run-time of the model during the classification process but the accuracy of the model might improve as more learning is achieved by the model. 

```r
set.seed(1234)
start.time <- Sys.time()
fitControl <- trainControl(method = "cv",number = 5,allowParallel = TRUE)
boost.fit <- train(classe ~ ., method="gbm",data=trainset,verbose=FALSE,trControl = fitControl)
end.time <- Sys.time();time.taken <- end.time - start.time;time.taken
```

```
## Time difference of 2.145583 mins
```

```r
boost.pred <- predict(boost.fit, testset)
boost.perf <- table(testset$class, boost.pred,dnn=c("Actual", "Predicted"))
```


```r
confusionMatrix(boost.perf)
```

```
## Confusion Matrix and Statistics
## 
##       Predicted
## Actual    A    B    C    D    E
##      A 2194   26    6    5    1
##      B   55 1399   59    1    4
##      C    0   48 1300   18    2
##      D    1    7   34 1230   14
##      E    3   14   18   17 1390
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9576          
##                  95% CI : (0.9529, 0.9619)
##     No Information Rate : 0.2872          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9463          
##  Mcnemar's Test P-Value : 3.667e-07       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9738   0.9364   0.9174   0.9677   0.9851
## Specificity            0.9932   0.9813   0.9894   0.9915   0.9919
## Pos Pred Value         0.9830   0.9216   0.9503   0.9565   0.9639
## Neg Pred Value         0.9895   0.9850   0.9819   0.9937   0.9967
## Prevalence             0.2872   0.1904   0.1806   0.1620   0.1798
## Detection Rate         0.2796   0.1783   0.1657   0.1568   0.1772
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9835   0.9588   0.9534   0.9796   0.9885
```



```r
performance(boost.perf)
```

```
## Sensitivity = 0.96217
## Specificity = 0.98829
## Positive Predictive Value = 0.98175
## Negative Predictive Value = 0.97554
## Accuracy = 0.97795
```

## Analysis / Review and Conclusions

Now we'll be looking at the measures of predictive accuracy displayed by each executed model.Each of these model/classifiers performed quite well but in particular.
the random forest model outperformed the others. For instance Sensitivity was 98% of which is the probability of getting a positive classification when the true outcome is positive, then Specificity was 99% which is the probability of getting a negative classification when the true outcome is negative.Positive Predictive Value of 99% which is the probability that an observation with a positive classification is correctly identified as positive. Negative Predictive Value of 99% which is the probability that an observation with a negative classification is correctly identified as negative.Accuracy of 99% which is the proportion of observations correctly identified.Also with a Kappa coeficient of 99%  which is basically telling us that there is a high level of aggreetments between the observers involve in the classification. Furthermore I executed cross-Validation using K=100 K=50 K=20 K=10 and K=5 in order to verify that changing the size of the folds might impact the results of the predictions and practically I got the exactly the same results therefore for the Efficiency or best run-time (elapse time taken by the classifier to execute and generate the model) I've selected the cross-validation with K=5 which completed in less time. It looks like that when it comes to clasifying qualitative data these models can perform well but another factor/variable to consider is the computational time taken by the classifier to complete. This is why I collected the star.time and end.time for each exeuction because when ones of these algorithms is implemented in a live/production system efficieny really matters.
  

