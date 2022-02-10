####### cheese data #######
cheese.df <- read.csv("tinydata.csv", stringsAsFactors = TRUE)
cheese.df
summary(cheese.df)

####### neural network for classification #######
install.packages("neuralnet")
library(neuralnet)

cheese.nn <- neuralnet(Acceptance ~ Fat + Salt,      
                       data = cheese.df,                
                       linear.output = FALSE,       
                       hidden = 3)                  

# display weights
cheese.nn$weights
# display predictions
prediction(cheese.nn)
# plot network
plot(cheese.nn, 
     rep = "best")   

## confusion matrix
library(caret)
predict.cheese <- predict(cheese.nn, cheese.df[, 2:3])
predicted.class <- apply(predict.cheese,       
                         1,                    
                         which.max) - 1        

confusionMatrix(as.factor(ifelse(predicted.class == 1, "like", "dislike")), 
                cheese.df$Acceptance, 
                positive = "like")

####### data pre-processing for neural net for bank data #######
bank.df <- read.csv("UniversalBank.csv")
bank.df <- bank.df[, -c(1, 5)] # drop ID and ZIP.Code
bank.df$Education <- as.factor(bank.df$Education)
bank.df$Personal.Loan <- as.factor(bank.df$Personal.Loan)
library(fastDummies)
bank.df <- dummy_cols(bank.df, select_columns = "Education", remove_first_dummy = TRUE, remove_selected_columns = TRUE)
str(bank.df)

# check for highly skewed predictors
library(e1071)
skewness(bank.df$Age)
skewness(bank.df$Experience)
skewness(bank.df$Income)
skewness(bank.df$Family)
skewness(bank.df$CCAvg)
skewness(bank.df$Mortgage)

# apply a log transformation to highly skewed predictors
bank.df$CCAvg <- log(bank.df$CCAvg + 1)
bank.df$Mortgage <- log(bank.df$Mortgage + 1)

# partition the data
set.seed(1)
train.index <- sample(rownames(bank.df), nrow(bank.df)*0.6)
bank.train <- bank.df[train.index, ]
valid.index <- setdiff(rownames(bank.df), train.index)
bank.valid <- bank.df[valid.index, ]

# convert all predictors to a 0-1 scale
bank.train.norm <- bank.train
bank.valid.norm <- bank.valid
cols <- colnames(bank.train[, -7])
for (i in cols) {
  bank.valid.norm[[i]] <- 
    (bank.valid.norm[[i]] - min(bank.train[[i]])) / (max(bank.train[[i]]) - min(bank.train[[i]]))
  bank.train.norm[[i]] <- 
    (bank.train.norm[[i]] - min(bank.train[[i]])) / (max(bank.train[[i]]) - min(bank.train[[i]]))
}
summary(bank.train.norm)
summary(bank.valid.norm)

####### neural net with 1 hidden layer of 3 nodes #######
bank.nn.3 <- neuralnet(Personal.Loan ~ .,            
                       data = bank.train.norm,          
                       linear.output = FALSE,       
                       hidden = 3)                 

# plot the neural net model
plot(bank.nn.3, rep = "best")

predict.nn.3 <- predict(bank.nn.3, bank.valid.norm)
predicted.class.3 <- apply(predict.nn.3,         
                           1,                    
                           which.max) - 1        

confusionMatrix(as.factor(predicted.class.3), 
                bank.valid.norm$Personal.Loan, 
                positive = "1")

####### neural net with 2 hidden layers of 2 nodes each #######
bank.nn.2.2 <- neuralnet(Personal.Loan ~ ., data = bank.train.norm, linear.output = FALSE,
                         hidden = c(2,2),       
                         stepmax = 1e+07)       
plot(bank.nn.2.2, rep = "best")

predict.nn.2.2 <- predict(bank.nn.2.2, bank.valid.norm)
predicted.class.2.2 <- apply(predict.nn.2.2,         
                             1,                      
                             which.max) - 1          

confusionMatrix(as.factor(predicted.class.2.2), 
                bank.valid.norm$Personal.Loan, 
                positive = "1")

####### data pre-processing for neural net for toyota data #######
toyota.df <- read.csv("ToyotaCorolla.csv", stringsAsFactors = TRUE)
t(t(names(toyota.df)))
toyota.df <- toyota.df[toyota.df$CC != 16000, -c(1:2, 6, 15)]
toyota.df$Mfg_Month <- as.factor(toyota.df$Mfg_Month)
str(toyota.df)
t(t(names(toyota.df)))
toyota.df <- dummy_cols(toyota.df, 
                        select_columns = c("Mfg_Month", "Fuel_Type", "Color"),
                        remove_first_dummy = TRUE,
                        remove_selected_columns = TRUE)

# check for highly skewed predictors
skewness(toyota.df$Age_08_04)
skewness(toyota.df$KM)
skewness(toyota.df$HP)
skewness(toyota.df$CC)
skewness(toyota.df$Doors)
skewness(toyota.df$Gears)
skewness(toyota.df$Quarterly_Tax)
skewness(toyota.df$Weight)
skewness(toyota.df$Guarantee_Period)

# apply a log transformation to highly skewed predictors
toyota.df$KM <- log(toyota.df$KM + 1)
toyota.df$Gears <- log(toyota.df$Gears + 1)
toyota.df$Quarterly_Tax <- log(toyota.df$Quarterly_Tax + 1)
toyota.df$Weight <- log(toyota.df$Weight + 1)
toyota.df$Guarantee_Period <- log(toyota.df$Guarantee_Period + 1)

# partition the data
set.seed(7)
train.index <- sample(rownames(toyota.df), nrow(toyota.df) * 0.7)
toyota.train <- toyota.df[train.index, ]
valid.index <- setdiff(rownames(toyota.df), train.index)
toyota.valid <- toyota.df[valid.index, ]

# convert all variables (INCLUDING QUANTITATIVE OUTCOME) to a 0-1 scale
toyota.train.norm <- toyota.train
toyota.valid.norm <- toyota.valid

cols <- colnames(toyota.train)
for (i in cols) {
  toyota.valid.norm[[i]] <- 
    (toyota.valid.norm[[i]] - min(toyota.train[[i]])) / (max(toyota.train[[i]]) - min(toyota.train[[i]]))
  toyota.train.norm[[i]] <- 
    (toyota.train.norm[[i]] - min(toyota.train[[i]])) / (max(toyota.train[[i]]) - min(toyota.train[[i]]))
}
summary(toyota.train.norm)
summary(toyota.valid.norm)

####### neural net with 1 hidden layer of 3 nodes #######
toyota.nn.3 <- neuralnet(Price ~ .,                   
                         data = toyota.train.norm,       
                         linear.output = FALSE,      
                         hidden = 3)                 

predict.nn.3 <- predict(toyota.nn.3, toyota.valid.norm)
head(predict.nn.3)

# convert back to original scale
minprice <- min(toyota.train$Price)
maxprice <- max(toyota.train$Price)
actpred <- data.frame(actual = toyota.valid$Price, 
           predicted = minprice + predict.nn.3*(maxprice - minprice))
head(actpred)
RMSE(actpred$predicted, actpred$actual)

####### neural net with 2 hidden layers of 3 nodes #######
toyota.nn.3.3 <- neuralnet(Price ~ .,                  
                           data = toyota.train.norm,   
                           linear.output = FALSE,      
                           hidden = c(3,3))            
plot(toyota.nn.3.3, rep = "best")

predict.nn.3.3 <- predict(toyota.nn.3.3, toyota.valid.norm)

# convert back to original scale
actpred <- data.frame(actual = toyota.valid$Price, 
                      predicted = minprice + predict.nn.3.3*(maxprice - minprice))
RMSE(actpred$predicted, actpred$actual)

####### neural net with 1 hidden layer of 10 nodes #######
toyota.nn.10 <- neuralnet(Price ~ .,                  
                           data = toyota.train.norm,   
                           linear.output = FALSE,      
                           hidden = 10)            
plot(toyota.nn.10, rep = "best")

predict.nn.10 <- predict(toyota.nn.10, toyota.valid.norm)

# convert back to original scale
actpred <- data.frame(actual = toyota.valid$Price, 
                      predicted = minprice + predict.nn.10*(maxprice - minprice))
RMSE(actpred$predicted, actpred$actual)

####### loop through different hidden layer sizes #######
RMSE.df <- data.frame(n = seq(1, 20, 1), RMSE.k = rep(0, 20))
for (i in 1:20) {
  toyota.nn <- neuralnet(Price ~ .,                  
                            data = toyota.train.norm,   
                            linear.output = FALSE,      
                            hidden = i)            
  predict.nn <- predict(toyota.nn, toyota.valid.norm)
  
  # convert back to original scale
  actpred <- data.frame(actual = toyota.valid$Price, 
                        predicted = minprice + predict.nn*(maxprice - minprice))
  RMSE.df[i,2] <- RMSE(actpred$predicted, actpred$actual)
}
RMSE.df
