##############################################################################
#Cleveland Heart Disease Dataset
#Classification - Healthy and Sick - heart disease 
#CART,Random Forest and Neural Network

#Modified by: Anand
#Modified Date: 12/16/2020
#############################################################################

rm(list=ls())

set.seed(748)

library("rpart")
library("caret")
library("randomForest")
library("neuralnet")
library("nnet")

load("cleveland.RData")
dim(cleveland)
names(cleveland)
table(cleveland$diag1)
table(cleveland$diag2)

#As we are not interested in the other response variable diag2 - 
#which signifies the stage of the heart disease so we will ignore from the dataset

#Create train and test data
trainidx<-sample(1:nrow(cleveland),nrow(cleveland)*0.7,replace=FALSE)
traindat<-cleveland[trainidx,-15]
testdat<-cleveland[-trainidx,-15]

table(traindat$diag1)
table(testdat$diag1)

#CART Model
model.control<-rpart.control(minsplit = 5,cp=0,maxdepth = 30,xval = 10)
c.model<-rpart(diag1~.,data = traindat,method = "class", control = model.control)

x11()
plot(c.model, uniform = T, compress = T)
text(c.model, cex = 0.5)

c.model$cptable

x11()
plotcp(c.model, minline = TRUE, lty = 3, col = 1,
       upper = c("size", "splits", "none"))


c.model$cptable
c.model$variable.importance
summary(c.model)
printcp(c.model)

min_cp = which.min(c.model$cptable[,4])
pruned_fit <- prune(c.model, cp = c.model$cptable[min_cp,1])


## plot the full tree and the pruned tree
x11()
plot(pruned_fit, branch = .3, compress=T, main = "Pruned Tree")
text(pruned_fit, cex = .6)

pruned_fit$variable.importance

#############Test before pruning and after pruning - plot confusion matrix##########
trainorigpred<-predict(c.model,newdata = traindat,type="class")
testorigpred<-predict(c.model,newdata = testdat,type="class")
trainorigcmat<-confusionMatrix(trainorigpred,traindat$diag1)
testorigcmat<-confusionMatrix(testorigpred,testdat$diag1)

trainoptpred<-predict(pruned_fit,newdata = traindat,type="class")
testoptpred<-predict(pruned_fit,newdata = testdat,type="class")
trainoptcmat<-confusionMatrix(trainoptpred,traindat$diag1)
testoptcmat<-confusionMatrix(testoptpred,testdat$diag1)


trainorigcmat
testorigcmat
trainoptcmat
testoptcmat

##################################################################################
##Random Forest 
##Optimal value of m and n trees
cland.rf<-randomForest(diag1~.,data = traindat,mtry=3,ntree=700)
cland.rf$confusion
cland.rf$importance

ypred<-predict(cland.rf,testdat)
test.conf<-confusionMatrix(ypred,testdat$diag1)
test.conf

#################################################################################
##Neural Network
#Data Conversion as dataset contains the categorical values - model matrix
dummy <- dummyVars(" ~.", fullRank = TRUE, data=cleveland[,-15]) 
ncland <- data.frame(predict(dummy, newdata = cleveland[,-15])) 
head(ncland)
names(ncland)
table(ncland$diag1.sick) 
#0 - bugg - healthy - no heart disease
#1 - sick - heart disease

traindat<-ncland[trainidx,]
testdat<-ncland[-trainidx,]

table(traindat$diag1.sick)
table(testdat$diag1.sick)

#######################################################################
# train a neural network
nnet1 <- neuralnet(diag1.sick ~ ., data = traindat, hidden = 1, err.fct = "ce", linear.output = FALSE)

x11()
plot(nnet1)

trainpred <- round(predict(nnet1, newdata = traindat))
train_acc <- length(which(traindat$diag1.sick == trainpred))/length(trainpred)
train_acc

testpred <- round(predict(nnet1, newdata = testdat))
test_acc <- length(which(testdat$diag1.sick == testpred))/length(testpred)
test_acc

######################################################################
#Tuning neural network to find optimal - number of hidden layers
#based on test and train accuracy

trainacclst<-c()
testacclst<-c()
nhiddenlayers<-c()

for(i in seq(1,10)){
  
  nhiddenlayers<-append(nhiddenlayers,i)
  nnet <- neuralnet(diag1.sick ~ ., data = traindat, hidden = i, 
                    stepmax = 10^9,err.fct = "ce", linear.output = FALSE)
  
  trainpred <- round(predict(nnet, newdata = traindat))
  train_acc <- length(which(traindat$diag1.sick == trainpred))/length(trainpred)
  trainacclst<-append(trainacclst,train_acc)
  
  testpred <- round(predict(nnet, newdata = testdat))
  test_acc <- length(which(testdat$diag1.sick == testpred))/length(testpred)
  testacclst<-append(testacclst,test_acc)
}


X11()
plot(x=nhiddenlayers,y=trainacclst,type="o",lty=2,col = "brown1",xlab = "number of nodes in hidden layers",main="Neural Network- Tuning",ylab="Accuracy")
lines(testacclst,type="o",lty=1,col="dodgerblue")
legend("bottomright",c("training.accuracy", "test.accuracy"),lty=c(2,1),col=c("brown1","dodgerblue"))

trainacclst
testacclst

########################################################################################
#Optimal hidden layers - 3
finalnet <- neuralnet(diag1.sick ~ ., data = traindat, hidden = 3, 
                      stepmax = 10^9,err.fct = "ce", linear.output = FALSE)

x11()
plot(finalnet)


ftrainpred <- round(predict(finalnet, newdata = traindat))
ftrain_acc <- length(which(traindat$diag1.sick == ftrainpred))/length(ftrainpred)

ftestpred <- round(predict(finalnet, newdata = testdat))
ftest_acc <- length(which(testdat$diag1.sick == ftestpred))/length(ftestpred)

ftrain_acc
ftest_acc

########################################################################################
#Deep Neural Network - how it performs
########################################################################################

deepnet <- neuralnet(diag1.sick ~ ., data = traindat, hidden = c(5,2), 
                      stepmax = 10^9, err.fct = "ce", linear.output = FALSE)

x11()
plot(deepnet)

trainpred <- round(predict(deepnet, newdata = traindat))
train_acc <- length(which(traindat$diag1.sick == trainpred))/length(trainpred)
train_acc

testpred <- round(predict(deepnet, newdata = testdat))
test_acc <- length(which(testdat$diag1.sick == testpred))/length(testpred)
test_acc



