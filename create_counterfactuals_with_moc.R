remotes::install_git(url = "https://github.com/susanne-207/moc", ref = "moc_without_iml", subdir = "counterfactuals")
library(counterfactuals)
library(iml)
library(ggplot2)
library(data.table)

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

x.interest_obs0 = df_toy[1,]
x.interest_obs0
pred = Predictor$new(data = df_toy, predict.function = pred_func)
pred$predict(newdata=x.interest)


library(tictoc)
tic()
cf = Counterfactuals$new(predictor = pred, 
                         x.interest = x.interest_obs0, 
                         target = c(0.5, 1), epsilon = 0, generations=175, mu=40, initialization = "icecurve")
toc()

cf_dir = ".//Counterfactuals//"
dir.create(cf_dir, showWarnings = FALSE)
df_cf_obs0 = cf$subset_results(10)$counterfactuals
fwrite(df_cf_obs0, paste0(cf_dir, "moc_cf_toy_dataset_obs0.csv"))
cf$subset_results(10)

x.interest_obs1 = df_toy[2,]
tic()
cf$explain(x.interest_obs1, target = c(0.5,1))
toc()
df_cf_obs1 = cf$subset_results(10)$counterfactuals
fwrite(df_cf_obs1, paste0(cf_dir, "moc_cf_toy_dataset_obs1.csv"))
