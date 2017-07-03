# Course Project: Weight Lifting Exercise
library(plyr)
library(dplyr)
library(caret)
library(kernlab)
library(randomForest)
library(ggplot2)
library(ggthemes)
options(digits = 3,max.print = 200)

urlTrain<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
urlTest<-"https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(urlTrain,destfile="./data/pml-training.csv", method="curl")
download.file(urlTest,destfile ="./data/pml-testing.csv", method = "curl")
training = read.csv("./data/pml-training.csv", na.strings=c('', 'NA'))
testing = read.csv("./data/pml-testing.csv", na.strings=c('', 'NA'))
dim(training);dim(testing)
# [1] 19622   160
# [1]  20 160
str(training,list.len=160)
# 'data.frame':	19622 obs. of  160 variables:
summary(training)

# Cleaning, Problem 1: huge NAs in most features (N=100)
trainingIsNA<-is.na.data.frame(training)
table(apply(trainingIsNA,2,sum))
#        0 19216 
#       60   100 
trainingAnyNAs<-apply(training,2,anyNA)
table(trainingAnyNAs)
#      trainingAnyNAs
#       FALSE  TRUE 
#       60     100

# subset data to drop 100 features riddled with NAs
# NOTE: conveniently, this also drops all features with #DIV/0!
trainingNoNAs<-training[,trainingAnyNAs==FALSE]
dim(trainingNoNAs)
# [1] 19622    60

# subset to drop 1st 7 non-numeric features (retaining non-numeric response)
trainingCleanVars<-trainingNoNAs[,-c(1:7)]
dim(trainingCleanVars)
# [1] 19622    53
head(trainingCleanVars)

# select same features in testing data set, too
testingNoNAs<-testing[,trainingAnyNAs==FALSE]
dim(testingNoNAs)
# [1] 20 60
testingCleanVars<-testingNoNAs[,-c(1:7)]
dim(testingCleanVars)
# [1] 20 53

# finally, doublecheck, no near zero variance features left
NearZeroClean<-nzv(trainingCleanVars,saveMetrics=TRUE)
sum(NearZeroClean$nzv)
# [1] 0

# split into final train, valid sets (with feature, response subsets)
set.seed(34567)
inTrain<-createDataPartition(trainingCleanVars$classe,p=0.75,list=FALSE)
trainCleanVars<-trainingCleanVars[inTrain,]
validCleanVars<-trainingCleanVars[-inTrain,]
trainFeatures<-as.matrix(trainCleanVars[,-53])
trainResponse<-as.matrix(trainCleanVars[,53])
validFeatures<-validCleanVars[,-53];validResponse<-validCleanVars[,53]
dim(trainFeatures); str(trainResponse)
# [1] 14718    52
# Factor w/ 5 levels "A","B","C","D",..
dim(validFeatures); str(validResponse)
# [1] 4904   52
# Factor w/ 5 levels "A","B","C","D",..
table(trainResponse)
#   A    B    C    D    E 
# 4185 2848 2567 2412 2706 
table(validResponse)
#   A    B    C    D    E 
# 1395  949  855  804  901 

# set up 10-fold Repeated Cross Validation for lda, gbm & svm 
# models in caret (but not rf)
ctrl<-trainControl(method = "cv", number = 10)

# NOTE: scale and center features in each model algorithm call


# 1. fit lda with caret (runtime = 1:00)
set.seed(34567)
fitLda<-train(classe ~ .,data=trainCleanVars, method="lda", 
              preProc = c("center","scale"), trControl=ctrl)
fitLda
# Pre-processing: centered (52), scaled (52) 
# Resampling: Cross-Validated (10 fold, repeated 3 times) 
# Summary of sample sizes: 13246, 13245, 13245, 13247, 13248, 13246, ... 
# Resampling results:
# Accuracy  Kappa 
# 0.701     0.6217
predictLda<-predict(fitLda,validFeatures)
confusionMatrix(predictLda,validResponse)$table
#               Reference
# Prediction    A    B    C    D    E
#           A 1140  128   69   56   40
#           B   35  623   98   25  155
#           C  115  123  576   94   81
#           D  101   33   98  606   87
#           E    4   42   14   23  538
confusionMatrix(predictLda,validResponse)$overall[1]
# Accuracy 
# 0.71 
table(predictLda==validResponse)
# FALSE  TRUE 
# 1421  3483


# 2. fit Boosted Trees with caret (runtime = 16:00)
set.seed(34567)
fitGbm<-train(classe ~ ., data = trainCleanVars, method="gbm", 
              preProc = c("scale","center"), trControl = ctrl, verbose = FALSE)
fitGbm
# Pre-processing: scaled (52), centered (52) 
# Resampling: Cross-Validated (10 fold)
# Summary of sample sizes: 13246, 13245, 13245, 13247, 13248, 13248, ...
# Resampling results across tuning parameters:
#         interaction.depth  n.trees  Accuracy  Kappa 
#         1                   50      0.7540    0.6881
#         2                   50      0.8553    0.8166
#         3                   50      0.8994    0.8727
#         3                  100      0.9449    0.9303
#         3                  150      0.9640    0.9544
# Tuning parameter 'shrinkage' was held constant at a value of 0.1
# parameter 'n.minobsinnode' was held constant at a value of 10
# Accuracy was used to select the optimal model using  the largest value.
# The final values used for the model were n.trees = 150, interaction.depth = 3,
# shrinkage = 0.1 and n.minobsinnode = 10.
plot(fitGbm,main="Why caret set Maximum Tree Depth parameter to 3",ylab="Accuracy",
     xlab="Boosting Iterations")
# FIG 1: Accuracy (cross-validation) x boosting iterations with lines for 
# Max Tree Depths of 1<2<3?
predictGbm<-predict(fitGbm,validCleanVars)
confusionMatrix(predictGbm,validCleanVars$classe)$table
#           Reference
# Prediction    A    B    C    D    E
#          A 1361   27    0    0    1
#          B   24  898   27    3    4
#          C    7   23  818   26    7
#          D    3    0   10  765   16
#          E    0    1    0   10  873
confusionMatrix(predictGbm,validCleanVars$classe)$overall[1]
# Accuracy 
# 0.961 
table(predictGbm==validCleanVars$classe)
# FALSE  TRUE 
#   189  4715  


# 3. fit SVM model, scaled & 10-fold cross validation
set.seed(34567)

#with ksvm in kernlab, which already scales and centers by default (runtime = 2:00)
#       [NOTE: only plots binary classifiers]
fitSVMrbf <- ksvm(classe ~ ., data = trainCleanVars, kernel = "rbfdot",
                  scaled = TRUE, C = 16, cross = 10)
fitSVMrbf
# parameter : cost C = 16 
# Hyperparameter : sigma =  0.0137325880957421  
# Number of Support Vectors : 3837  
# Training error : 0.01223 
# Cross validation error : 0.01821
predictSVMrbf<-predict(fitSVMrbf,validCleanVars)
confusionMatrix(predictSVMrbf,validCleanVars$classe)$table
#           Reference
# Prediction    A    B    C    D    E
#          A 1391   12    0    0    0
#          B    4  934    1    0    0
#          C    0    0  852   35    1
#          D    0    1    2  769    9
#          E    0    2    0    0  891
confusionMatrix(predictSVMrbf,validCleanVars$classe)$overall[1]
# Accuracy 
# 0.986
table(predictSVMrbf==validCleanVars$classe)
# FALSE  TRUE 
#    67  4837 


# 4. fit Random Forest bagged trees with randomForrest runtime = 25 + 2 = 27:00)
# choose no. of var.s with rfcv using 10-fold cross validation (runtime = 25:00)
set.seed(34567)
fitRandomFcv500<-rfcv(trainFeatures, trainResponse, cv.fold = 10)
str(fitRandomFcv500)
# List of 3
# $ n.var    : num [1:6] 52 26 13 6 3 1
# $ error.cv : Named num [1:6] 0.00679 0.00836 0.01135 0.04545 0.10932 ...
# ..- attr(*, "names")= chr [1:6] "52" "26" "13" "6" ...
# $ predicted:List of 6
# ..$ 52: Factor w/ 5 levels "A","B","C","D",..: 1 1 1 1 1 1 1 1 1 1 ...
names(fitRandomFcv500)
# [1] "n.var"     "error.cv"  "predicted"
error<-as.numeric(fitRandomFcv500$error.cv);variables<-as.numeric(fitRandomFcv500$n.var)
errorRates500<-data.frame(cbind(variables,error))
errorRates500
#       variables   error
# 1            52 0.00550
# 2            26 0.00666
# 3            13 0.00931
# 4             6 0.04253
# 5             3 0.10871
# 6             1 0.58846
# FIG 2: 
ggplot(errorRates500)+aes(variables,error)+geom_point(color="blue2")+theme_few()+
        labs(y = "Out of sample error rate",x = "Variables at each split",
             title = "Tuning Random Forest mtry parameter:",
             subtitle = "setting it close to 10 quickly drops OOB error")

# (runtime = 1:00)
set.seed(34567)
fitRandomF<-randomForest(classe ~ ., trainCleanVars, ntree = 100, mtry = 13)
fitRandomF
# Number of trees: 100
# No. of variables tried at each split: 13
# OOB estimate of  error rate: 0.54%
# Confusion matrix:
#      A    B    C    D    E class.error
# A 4179    6    0    0    0     0.00143
# B   15 2824    9    0    0     0.00843
# C    0   12 2551    4    0     0.00623
# D    0    0   17 2392    3     0.00829
# E    0    0    4    9 2693     0.00480
# FIG 3
plot(fitRandomF,main="Training error drops quickly as forest approaches 100 trees", 
     sub = "Note: separate colored line for each predicted response")
predictRf<-predict(fitRandomF,validFeatures) 
confusionMatrix(predictRf,validResponse)$table
#           Reference
# Prediction    A    B    C    D    E
#          A 1390    3    0    0    0
#          B    5  944    2    0    0
#          C    0    2  853    7    1
#          D    0    0    0  797    1
#          E    0    0    0    0  899
confusionMatrix(predictRf,validResponse)$overall[1]
# Accuracy 
# 0.996
table(predictRf==validResponse)
# FALSE  TRUE 
#    21  4883

# re-estimate OOB error with cross validation for ntree = 100?
set.seed(34567)
fitRandomFcv100<-rfcv(trainFeatures, trainResponse, ntree = 100, cv.fold = 10)
error<-as.numeric(fitRandomFcv100$error.cv);variables<-as.numeric(fitRandomFcv100$n.var)
errorRates100<-data.frame(cbind(variables,error))
errorRates100[variables==13,]
#       variables   error
#     3        13 0.00951


# final 20 predicted values:
# by 1. Lda
predictFinalLda<-predict(fitLda,testingCleanVars)
# by 2. Gbm
predictFinalGbm<-predict(fitGbm,testingCleanVars)
# by 3. SVM
predictFinalSVM<-predict(fitSVMrbf,testingCleanVars)
# by 4. RF
predictFinalRF<-predict(fitRandomF,testingCleanVars)

# 3 of 4 models agree!
sum(predictFinalLda==predictFinalSVM)
# [1] 13
sum(predictFinalGbm==predictFinalSVM)
# [1] 20
sum(predictFinalSVM==predictFinalRF)
# [1] 20

# table comparing models by out of sample accuracy
modelAccuracy<-c(RF=confusionMatrix(predictRf,validResponse)$overall[1],
                 SVM=confusionMatrix(predictSVMrbf,validCleanVars$classe)$overall[1],
                 GBM=confusionMatrix(predictGbm,validCleanVars$classe)$overall[1],
                 LDA=confusionMatrix(predictLda,validResponse)$overall[1])
data.frame(modelAccuracy,row.names = c("RF","SVM","GBM","LDA"))
#       modelAccuracy
# RF          0.996
# SVM         0.986
# GBM         0.961
# LDA         0.710

# Conclusion: Only LDA fails to predict all 20 testing values. And, since Gbm, 
# SVM and RF all predict all of them, the choice between those 3 models should
# be on runtime.

# print predictions of chosen model (SVM), if necessary?
as.matrix(predictFinalRF)

