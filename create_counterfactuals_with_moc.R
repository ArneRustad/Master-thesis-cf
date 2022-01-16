#remotes::install_git(url = "https://github.com/susanne-207/moc", ref = "moc_without_iml", subdir = "counterfactuals")
library(counterfactuals)
library(iml)
library(ggplot2)
library(data.table)
library(tictoc)

df_toy = read.csv("Datasets//df_toy.csv")
df_toy

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

df_plot_pred_func = data.frame(expand.grid(x1 = seq(0,10,length.out=100), x2 = seq(-1,1,length.out=100)))
df_plot_pred_func$func = pred_func(NULL, newdata=df_plot_pred_func)
head(df_plot_pred_func)
ggplot(df_plot_pred_func, aes(x = x1, y = x2)) + geom_tile(aes(fill = func)) +
  geom_point(data=df_toy, aes(x = x1, y=x2, color =x3))

x.interest_obs245 = df_toy[245,]
x.interest_obs245


?Counterfactuals

pred_func(NULL, x.interest_obs245)

cf$results

tic()
cf = Counterfactuals$new(predictor = pred, 
                         x.interest = x.interest_obs245, 
                         target = ifelse(pred_func(NULL, x.interest_obs245) < 0.5, c(0.5, 1), c(0, 0.5)),
                         epsilon = 0, generations=2, mu=40, initialization = "icecurve")
toc()

create_counterfactual = function(x_obs_nr, data, pred_func, generations = 200, mu = 50, initialization = "icecurve",
                                 epsilon = 0, cf_dir = ".//Counterfactuals//",
                                 dataset_name = NULL) {
  x.interest = data[x_obs_nr + 1,]
  pred = Predictor$new(data = df_toy, predict.function = pred_func)
  tic()
  cf = Counterfactuals$new(predictor = pred, 
                           x.interest = x.interest, 
                           target = ifelse(pred_func(NULL, x.interest) < 0.5, c(0.5, 1), c(0, 0.5)),
                           epsilon = epsilon, generations=generations, mu=mu, initialization = initialization)
  toc()
  dir.create(cf_dir, showWarnings = FALSE)
  df_cf = cf$subset_results(10)$counterfactuals
  fwrite(df_cf, paste0(cf_dir, sprintf("%s_obs%d.csv", dataset_name, x_obs_nr)))
  return (cf)
}

cf1 = create_counterfactual(89, data=df_toy, pred_func=pred_func, dataset_name = "Syn2D_cf")
cf1$subset_results(10)

cf2 = create_counterfactual(243, data=df_toy, pred_func=pred_func, dataset_name = "Syn2D_cf")
cf2$subset_results(10)

cf3 = create_counterfactual(11, data=df_toy, pred_func=pred_func, dataset_name = "Syn2D_cf")
cf2$subset_results(10)




########### Adult dataset
df_adult_train = read.csv("Datasets//df_adult_edited_train.csv")
cf_adult = create_counterfactual(0, data=df_adult_train, pred_func=pred_func, dataset_name = "Adult_cf")

install.packages("drat", repos="https://cran.rstudio.com")
drat:::addRepo("dmlc")
install.packages("xgboost", repos="http://dmlc.ml/drat/", type = "source")
