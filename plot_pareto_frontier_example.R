library(ggplot2)
install.packages('latex2exp')
library(latex2exp)
x1 = seq(0,4, length.out = 10000)

df = data.frame(x1 = x1, x2_upper_bound = rep(4,length(x1)), x2_pareto_frontier = 3-x1^2)
ggplot(df, aes(x = x1)) + geom_ribbon(aes(ymin = x2_pareto_frontier, ymax = x2_upper_bound), fill = "lightgrey", alpha = 0.5) +
  theme_minimal() + geom_line(aes(y = x2_pareto_frontier), size = 1.5) + xlab(TeX("x_1")) + ylab(TeX("x_2"))
ggsave("pareto_frontier_example.jpg", path = "Images", height = 3, width = 4)
