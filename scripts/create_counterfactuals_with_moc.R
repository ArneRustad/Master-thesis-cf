# Importing packages
library(counterfactuals)
library(iml)
library(ggplot2)
library(data.table)
library(tictoc)
library(dplyr)


df_toy = read.csv("Datasets//df_toy.csv")

# Create prediction function for test dataset
sigmoid = function(x) {
  1 / (1 + exp(-x))
}

pred_func = function(model, newdata) {
  x1 = newdata[, "x1"]
  x2 = newdata[, "x2"]
  value = (0.1 * (x1 - 5)^2 + (x2)^2 - 0.5 + 0.2 * sqrt(x1) -
             0.1*x1 + 0.005*x1^3 * x2 + 0.5*sin(x1) + 0.3*cos(x2) 
           )
  value = sign(value) * (abs(value))^(1/2)
  return(sigmoid(value))
}

# Visualize prediction function
df_plot_pred_func = data.frame(expand.grid(x1 = seq(0,10,length.out=100),
                                           x2 = seq(-1,1,length.out=100)))
df_plot_pred_func$func = pred_func(NULL, newdata=df_plot_pred_func)
head(df_plot_pred_func)
ggplot(df_plot_pred_func, aes(x = x1, y = x2)) + geom_tile(aes(fill = func)) +
  geom_point(data=df_toy, aes(x = x1, y=x2, color =x3))

# Change discrete columns to be of type object
df_toy = df_toy %>% mutate(across(where(is.character), as.factor))

# Create function for automating the counterfactual generating and saving step
create_counterfactual = function(x_obs_nr, data, pred_func, generations = 200,
                                 mu = 50, initialization = "icecurve",
                                 epsilon = 0, cf_dir = ".//Counterfactuals//",
                                 dataset_name = NULL, track.infeas = TRUE) {
  x.interest = data[x_obs_nr + 1,]
  pred = Predictor$new(data = data, predict.function = pred_func)
  
  if (pred_func(NULL, x.interest) < 0.5) {
    target = c(0.5, 1)
  } else {
    target = c(0, 0.5)
  }
  tic()
  cf = Counterfactuals$new(predictor = pred, 
                           x.interest = x.interest, 
                           target = target,
                           epsilon = epsilon, generations=generations, mu=mu,
                           initialization = initialization,
                           track.infeas = TRUE)
  toc()
  if (! is.null(dataset_name)) {
    dir.create(cf_dir, showWarnings = FALSE)
    df_cf = cf$subset_results(10)$counterfactuals
    fwrite(df_cf, paste0(cf_dir, sprintf("%s_obs%d.csv", dataset_name, x_obs_nr)))
  }
  return (cf)
}

# Run the methods three times. One time for each observation from the toy dataset to be
# explained
cf1 = create_counterfactual(89, data=df_toy, pred_func=pred_func, dataset_name = "Syn2D_cf")
cf1$subset_results(10)

cf2 = create_counterfactual(243, data=df_toy, pred_func=pred_func, dataset_name = "Syn2D_cf")
cf2$subset_results(10)

cf3 = create_counterfactual(11, data=df_toy, pred_func=pred_func, dataset_name = "Syn2D_cf")
cf2$subset_results(10)


########### Adult dataset ###### Create counterfactuals for the single observation from
# the adult dataset
library(dplyr)
library(tidyverse)
library(caret)
library(data.table)
library(xgboost)
library(xtable)
# Fetch adult data and perform some preprocessing to be able to use xgboost
df_adult_train = read.csv("Datasets//df_adult_edited_train.csv")
df_adult_train = df_adult_train %>% mutate(across(where(is.character), as.factor))
df_adult_train_wo_income = df_adult_train[colnames(df_adult_train) != "income"]

oh_encoder <- dummyVars("~ .", data = df_adult_train_wo_income)
matrix_adult_train <- predict(oh_encoder, newdata = df_adult_train)
label_train = ifelse(df_adult_train$income == "<=50K", 0, 1)

df_adult_test = read.csv("Datasets//df_adult_edited_test.csv")
matrix_adult_test = predict(oh_encoder, newdata = df_adult_test)
label_test = ifelse(df_adult_test$income == "<=50K", 0, 1)

# Create train and test matrices for input into the xgboost ecosystem
xgb.train = xgb.DMatrix(data=matrix_adult_train,label=label_train)
xgb.test = xgb.DMatrix(data=matrix_adult_test,label=label_test)
# Fit xgboost model
xgb.fit=xgb.train(
  data=xgb.train,
  nrounds=1000
)
# Test that xgboost model in R performs likewise as the xgboost model in Python
xgb.pred = predict(xgb.fit,xgb.test,reshape=T)
xgb.pred = round(xgb.pred)
mean(xgb.pred == label_test)

# Create xgboost prediction function that will be used in the MOC framework
pred_func_xgboost = function(model, newdata) {
  matrix_data = predict(oh_encoder, newdata = newdata)
  xgb.matrix = xgb.DMatrix(data=matrix_data)
  xgb.pred = predict(xgb.fit,xgb.matrix,reshape=T)
  return (xgb.pred)
}

# Run MOC for the first observation in the Adult train dataset
cf_adult = create_counterfactual(0, data=df_adult_train_wo_income[1:2000,],
                                 pred_func=pred_func_xgboost,
                                 generations = 100, initialization = "icecurve", mu = 20,
                                 track.infeas = TRUE)
# print counterfactuals
cf_adult$results
# print counterfactuals for easy insertion into latex document
xtable(t(cf_adult$subset_results(3)$counterfactuals))