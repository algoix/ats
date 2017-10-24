###https://www.r-bloggers.com/machine-learning-stock-market-data-part-1-logistic-regression/
### https://github.com/pakinja/Data-R-Value/blob/master/MachineLearning_Classification_StockMarketData_LDA/stock_market_data_LDA.R
### install and load required packages
library(ISLR)
library(psych)
### explore the dataset
names(Smarket)
dim(Smarket)
summary(Smarket)
### correlation matrix
cor(Smarket[,-9])
### correlations between th lag variables and today
### returns are close to zero
### the only substantial correlation is between $Year
### and $Volume
plot(Smarket$Volume,main= "Stock Market Data", ylab = "Volume")
### scatterplots, distributions and correlations
pairs.panels(Smarket)
### fit a logistic regression model to predict $Direction
### using $Lag1 through $Lag5 and $Volume
### glm(): generalized linear model function
### family=binomial => logistic regression
glm.fit <- glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,data = Smarket, family = binomial)
summary(glm.fit)
### the smallest p_value is associated with Lag1
### the negative coefficient for this predictor suggests
### that if the market had a positive return yesterday,
### then it is less likely to go up today
### at a value of 0.15, the p-value is still relatively large,
### and so there is no clear evidence of a real association
### between $Lag1 and $Direction
### explore fitted model coefficients
coef(glm.fit)
summary(glm.fit)$coef
summary(glm.fit)$coef[ ,4]
### predict the probability that the market will go up,
### given values of the predictors
glm.probs <- predict(glm.fit, type="response")
glm.probs[1:10]
contrasts(Smarket$Direction)
### these values correspond to the probability of the market
### going up, rather than down, because the contrasts()
### function indicates that R has created a dummy variable with
### a 1 for Up

### create a vector of class predictions based on whether the
### predicted probability of a market increase is greater than
### or less than 0.5
glm.pred <- rep ("Down", 1250)
glm.pred[glm.probs > .5] <- "Up"
### confusion matrix in order to determine how many observations
### were correctly or incorrectly classified
table(glm.pred, Smarket$Direction)
mean(glm.pred == Smarket$Direction)
### model correctly predicted that the market would go up on 507
### days and that it would go down on 145 days, for a total of
### 507 + 145 = 652 correct predictions
### ogistic regression correctly predicted the movement of the
### market 52.2 % of the time
### to better assess the accuracy of the logistic regression model
### in this setting, we can fit the model using part of the data,
### and then examine how well it predicts the held out data
train <- (Smarket$Year < 2005)
Smarket.2005 <- Smarket[!train, ]
dim(Smarket.2005)

Direction.2005 <- Smarket$Direction[!train]

glm.fit <- glm(Direction~Lag1+Lag2+Lag3+Lag4+Lag5+Volume,
               data = Smarket, family = binomial, subset = train)
glm.probs <- predict(glm.fit, Smarket.2005, type = "response")
### compute the predictions for 2005 and compare them to the
### actual movements of the market over that time period
glm.pred <- rep("Down", 252)
glm.pred[glm.probs > 0.5] <- "Up"
table(glm.pred, Direction.2005)

mean(glm.pred == Direction.2005)
mean(glm.pred != Direction.2005)
### not generally expect to be able to use previous days returns ### to predict future market performance

### refit the logistic regression using just $Lag1 and $Lag2,
### which seemed to have the highest predictive power in the
### original logistic regression model
glm.fit <- glm(Direction ~ Lag1 + Lag2 , data = Smarket,
               family = binomial, subset = train)
glm.probs <- predict(glm.fit, Smarket.2005 , type = "response")
glm.pred <- rep("Down", 252)
glm.pred[glm.probs > 0.5] <- "Up"
table(glm.pred, Direction.2005)
mean(glm.pred == Direction.2005)
### if we want to predict the returns associated with particular
### values of $Lag1 and $Lag2
predict(glm.fit, newdata = data.frame(Lag1 = c (1.2 ,1.5),
                                      Lag2 = c(1.1, -0.8)) , type = "response")


### "An Introduction to Statistical Learning.
### With applications in R" by Gareth James,
### Daniela Witten, Trevor Hastie and Robert Tibshirani.
### Springer 2015.


### install and load required packages
library(ISLR)
library(psych)
library(MASS)

### perform linear discriminant analysis LDA on the stock
### market data
train <- (Smarket$Year < 2005)
Smarket.2005 <- Smarket[!train, ]
Direction.2005 <- Smarket$Direction[!train]

lda.fit <- lda(Direction ~ Lag1 + Lag2 , data = Smarket , subset = train )
lda.fit

### LDA indicates that 49.2% of training observations
### correspond to days during wich the market went down

### group means suggest that there is a tendency for the
### previous 2 days returns to be negative on days when the
### market increases, and a tendency for the previous days
### returns to be positive on days when the market declines

### coefficients of linear discriminants output provides the
### linear combination of Lag1 and Lag2 that are used to form
### the LDA decision rule

### if (−0.642 * Lag1 − 0.514 * Lag2) is large, then the LDA
### classifier will predict a market increase, and if it is
### small, then the LDA classifier will predict a market decline

### plot() function produces plots of the linear discriminants,
### obtained by computing (−0.642 * Lag1 − 0.514 * Lag2) for
### each of the training observations
plot (lda.fit)

### predictions
lda.pred <- predict(lda.fit, Smarket.2005)
names(lda.pred)

lda.class <- lda.pred$class
table(lda.class, Direction.2005)
mean(lda.class == Direction.2005)

### the LDA and logistic regression predictions are almost identical

### apply a 50% threshold to the posterior probabilities allows
### us to recreate the predictions in lda.pred$class
sum(lda.pred$posterior [ ,1] >= 0.5)
sum(lda.pred$posterior [ ,1] < 0.5)

### posterior probability output by the model corresponds to
### the probability that the market will decrease
lda.pred$posterior[1:20 ,1]
lda.class[1:20]

### use a posterior probability threshold other than 50 % in order
### to make predictions

### suppose that we wish to predict a market decrease only if we
### are very certain that the market will indeed decrease on that
### day-say, if the posterior probability is at least 90%
sum(lda.pred$posterior[ ,1] > 0.9)

### No days in 2005 meet that threshold! In fact, the greatest
### posterior probability of decrease in all of 2005 was 52.02%