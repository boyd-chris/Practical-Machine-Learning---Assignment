# Summary

This project is an attempt to predict how correctly a person did the
exercise. The data set contains data pertaining to measurements of body
movement during exercise. The data was collected using wearable devises
during exercise routine. The model will predict using the variable
‘classe’. Using this model we will also predict 20 sample data provided.

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now
possible to collect a large amount of data about personal activity
relatively inexpensively. These type of devices are part of the
quantified self movement – a group of enthusiasts who take measurements
about themselves regularly to improve their health, to find patterns in
their behavior, or because they are tech geeks. One thing that people
regularly do is quantify how much of a particular activity they do, but
they rarely quantify how well they do it. In this project, the goal will
be to use data from accelerometers on the belt, forearm, arm, and
dumbbell of 6 participants. The participants were asked to perform
barbell lifts correctly and incorrectly in 5 different ways. The data
for this project comes from the source:
<http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har>.

# Building Prediction Model

### Initialize

Load required packages and set environment options  
Load data. The data has been already downloaded from the links provided
and stored under the folder ‘Course8’ within the working directory  
Training data is loaded into dataframe ‘training’  
Testing data is loaded into dataframe ‘testing’  
Make a copy of training data into ‘trainclean’. This data frame is used
to clean data by selecting only the required columns to build prediction
model

    suppressMessages(library(caret))
    suppressMessages(library(dplyr))
    set.seed(12321)
    options(scipen = 999)
    training <- read.csv("Course8/pml-training.csv")
    testing <- read.csv("Course8/pml-testing.csv")
    trainclean <- training

### Cleaning Data

    dim(trainclean)

    ## [1] 19622   160

    dim(testing)

    ## [1]  20 160

We can take a quick look at the distribution of Class(classe) values

    plot(as.factor(training$classe), 
    main = "How 'classe' is distributed within Training data", 
    xlab = "classe (Class)", ylab = "Frequency", col = "green",
    col.axis = "darkgreen", col.lab = "blue", col.main = "purple")

![](Practical-Machine-Learning---Project_files/figure-markdown_strict/class-frequency-1.png)

Here is the definition of Class (classe) values  
Class A : Exactly according to the specification  
Class B : Throwing the elbows to the front  
Class C : Lifting the dumbbell only halfway  
Class D : Lowering the dumbbell only halfway  
Class E : Throwing the hips to the front

summary(trainclean) (the result is not displayed here as it is very
big)  
shows that the data frame contains 19622 rows and 160 columns  
The following columns have only maximum of 406 valid values. Some does
not even have any valid values  
Column names staring with ‘max’, ‘min’, ‘avg’, ‘amplitude’, ‘var’,
‘stddev’, ‘kurtosis’ and ‘skewness’  
These columns can be removed as they do not have any impact in the
prediction model  
Also, we can remove first 7 columns from the data frame as they also
will not have any positive impact to the model  
‘X’, ‘user\_name’, ‘raw\_timestamp\_part\_1’, ‘raw\_timestamp\_part\_2’,
‘cvtd\_timestamp’, ‘new\_window’ and ‘num\_window’

    trainclean <- select(trainclean, 
    !starts_with("max") & 
    !starts_with("min") & 
    !starts_with("avg") &
    !starts_with("amplitude") & 
    !starts_with("var") & 
    !starts_with("stddev") & 
    !starts_with("kurtosis") & 
    !starts_with("skewness") &
    !(X:num_window)
    )

    dim(trainclean)

    ## [1] 19622    53

Now the cleaned data only has 53 columns with valid values required for
prediction

## Model Selection

Here we will be comparing 2 different models  
1. Model using the full training data (cleaned)  
2. Model using only partitioned data

We will also compare 2 different methods and apply the most accurate
one  
1. Tree  
2. Random Forest

### Model with full data

First we will apply these 2 methods to the cleaned full data

    ModFitTR1 <- train(classe ~ ., data = trainclean, method = "rpart")
    # ModFitTR1 contains the Model using Tree method
    controlvar <- trainControl(method = "cv", 5)
    ModFitRF1 <- train(classe ~ ., data = trainclean, method = "rf", trControl = controlvar, ntree = 250)
    # ModFitRF1 contains the Model using Random Forest method

We can see the accuracy of these 2 methods using confusionMatrix and
select the most accurate method

    confusionMatrix(ModFitTR1)

    ## Bootstrapped (25 reps) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 25.5  8.8  8.4  7.7  3.1
    ##          B  0.4  5.9  0.6  2.7  2.3
    ##          C  1.9  3.7  7.8  4.1  3.7
    ##          D  0.4  1.0  0.8  1.8  0.9
    ##          E  0.1  0.0  0.0  0.1  8.5
    ##                             
    ##  Accuracy (average) : 0.4947

    confusionMatrix(ModFitRF1)

    ## Cross-Validated (5 fold) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 28.4  0.1  0.0  0.0  0.0
    ##          B  0.0 19.2  0.0  0.0  0.0
    ##          C  0.0  0.0 17.3  0.2  0.0
    ##          D  0.0  0.0  0.1 16.2  0.0
    ##          E  0.0  0.0  0.0  0.0 18.3
    ##                             
    ##  Accuracy (average) : 0.9941

We can clearly note that Random Forest method is far better with 0.9941
accuracy

### Model with Partitioned data

Use the cleaned full data and make 2 partitions  
1. 70% data for training  
2. The rest for testing

Once we have the models, we can compare their accuracy using partitioned
test data and calculate Out of Sample Error also

    inTrain <- createDataPartition(trainclean$classe, p=0.70, list=FALSE)
    trainpartclean <- trainclean[inTrain, ]
    testpartclean <- trainclean[-inTrain, ]
    ModFitTR2 <- train(classe ~ ., data = trainpartclean, method = "rpart")
    ModFitRF2 <- train(classe ~ ., data = trainpartclean, method = "rf", trControl = controlvar, ntree = 250)
    confusionMatrix(ModFitTR2)

    ## Bootstrapped (25 reps) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 25.4  7.7  7.9  7.0  2.5
    ##          B  0.5  6.5  0.5  2.5  2.2
    ##          C  2.0  3.8  7.8  3.3  3.9
    ##          D  0.4  1.1  1.3  3.5  1.0
    ##          E  0.3  0.1  0.0  0.0  8.7
    ##                             
    ##  Accuracy (average) : 0.5182

    confusionMatrix(ModFitRF2)

    ## Cross-Validated (5 fold) Confusion Matrix 
    ## 
    ## (entries are percentual average cell counts across resamples)
    ##  
    ##           Reference
    ## Prediction    A    B    C    D    E
    ##          A 28.4  0.2  0.0  0.0  0.0
    ##          B  0.0 19.1  0.1  0.1  0.0
    ##          C  0.0  0.1 17.3  0.2  0.0
    ##          D  0.0  0.0  0.1 16.1  0.1
    ##          E  0.0  0.0  0.0  0.0 18.3
    ##                             
    ##  Accuracy (average) : 0.9906

Here also we can see that Random Forest is more accurate

Now we can use the cleaned test data to compare these two methods

    predictresultTR <- predict(ModFitTR2, testpartclean)
    confusionMatrix(table(testpartclean$classe, predictresultTR))

    ## Confusion Matrix and Statistics
    ## 
    ##    predictresultTR
    ##        A    B    C    D    E
    ##   A 1534   20  116    0    4
    ##   B  489  359  291    0    0
    ##   C  465   38  523    0    0
    ##   D  440  162  362    0    0
    ##   E  164  147  281    0  490
    ## 
    ## Overall Statistics
    ##                                           
    ##                Accuracy : 0.4938          
    ##                  95% CI : (0.4809, 0.5067)
    ##     No Information Rate : 0.5254          
    ##     P-Value [Acc > NIR] : 1               
    ##                                           
    ##                   Kappa : 0.338           
    ##                                           
    ##  Mcnemar's Test P-Value : NA              
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.4961   0.4945  0.33249       NA  0.99190
    ## Specificity            0.9499   0.8488  0.88335   0.8362  0.89019
    ## Pos Pred Value         0.9164   0.3152  0.50975       NA  0.45287
    ## Neg Pred Value         0.6300   0.9227  0.78391       NA  0.99917
    ## Prevalence             0.5254   0.1234  0.26729   0.0000  0.08394
    ## Detection Rate         0.2607   0.0610  0.08887   0.0000  0.08326
    ## Detection Prevalence   0.2845   0.1935  0.17434   0.1638  0.18386
    ## Balanced Accuracy      0.7230   0.6716  0.60792       NA  0.94105

    predictresultRF <- predict(ModFitRF2, testpartclean)
    confusionMatrix(table(testpartclean$classe, predictresultRF))

    ## Confusion Matrix and Statistics
    ## 
    ##    predictresultRF
    ##        A    B    C    D    E
    ##   A 1671    2    1    0    0
    ##   B   10 1122    7    0    0
    ##   C    0    5 1017    4    0
    ##   D    0    0   16  945    3
    ##   E    0    0    3    0 1079
    ## 
    ## Overall Statistics
    ##                                                
    ##                Accuracy : 0.9913               
    ##                  95% CI : (0.9886, 0.9935)     
    ##     No Information Rate : 0.2856               
    ##     P-Value [Acc > NIR] : < 0.00000000000000022
    ##                                                
    ##                   Kappa : 0.989                
    ##                                                
    ##  Mcnemar's Test P-Value : NA                   
    ## 
    ## Statistics by Class:
    ## 
    ##                      Class: A Class: B Class: C Class: D Class: E
    ## Sensitivity            0.9941   0.9938   0.9741   0.9958   0.9972
    ## Specificity            0.9993   0.9964   0.9981   0.9962   0.9994
    ## Pos Pred Value         0.9982   0.9851   0.9912   0.9803   0.9972
    ## Neg Pred Value         0.9976   0.9985   0.9944   0.9992   0.9994
    ## Prevalence             0.2856   0.1918   0.1774   0.1613   0.1839
    ## Detection Rate         0.2839   0.1907   0.1728   0.1606   0.1833
    ## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
    ## Balanced Accuracy      0.9967   0.9951   0.9861   0.9960   0.9983

    accuTR <- confusionMatrix(table(testpartclean$classe, predictresultTR))$overall[1]
    accuRF <- confusionMatrix(table(testpartclean$classe, predictresultRF))$overall[1]

    ooseTR <- 1 - as.numeric(confusionMatrix(table(testpartclean$classe, predictresultTR))$overall[1])
    ooseRF <- 1 - as.numeric(confusionMatrix(table(testpartclean$classe, predictresultRF))$overall[1])

    print(paste('Accuracy of Tree method:', accuTR))

    ## [1] "Accuracy of Tree method: 0.493797790994053"

    print(paste('Accuracy of Random Forest method:', accuRF))

    ## [1] "Accuracy of Random Forest method: 0.991333899745115"

    print(paste('Out of Sample Error of Tree method:', ooseTR))

    ## [1] "Out of Sample Error of Tree method: 0.506202209005947"

    print(paste('Out of Sample Error of Random Forest method:', ooseRF))

    ## [1] "Out of Sample Error of Random Forest method: 0.00866610025488534"

The results clearly show that Random Forest method is much better  
We will use Random Forest method to predict test data (‘testing’)
provided

## Prediction

We have 2 models  
1. Using full cleaned data (ModFitRF1)  
2. Using partitioned training data (ModFitRF2)

We can apply both models and we can see that both gives the same result

    Result <- data.frame()
    for (i in 1:20) { 
       Result[i,1] <- testing$problem_id[i]
       Result[i,2] <- predict(ModFitRF1, newdata = testing[i,])
       Result[i,3] <- predict(ModFitRF2, newdata = testing[i,])
    }
    colnames(Result) <- c("Problem_Id", "Model_1_Result", "Model_2_ Result")

    Result

    ##    Problem_Id Model_1_Result Model_2_ Result
    ## 1           1              B               B
    ## 2           2              A               A
    ## 3           3              B               B
    ## 4           4              A               A
    ## 5           5              A               A
    ## 6           6              E               E
    ## 7           7              D               D
    ## 8           8              B               B
    ## 9           9              A               A
    ## 10         10              A               A
    ## 11         11              B               B
    ## 12         12              C               C
    ## 13         13              B               B
    ## 14         14              A               A
    ## 15         15              E               E
    ## 16         16              E               E
    ## 17         17              A               A
    ## 18         18              B               B
    ## 19         19              B               B
    ## 20         20              B               B

# Conclusion

Random Forest method is more accurate comparing to Tree method  
Fitted 2 models with full data and partitioned data  
Result of these 2 models give same prediction for this set of data

– The End –
