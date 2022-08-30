library(dplyr)
library(tidyverse)
library(data.table)
library(ggplot2)
library(stringr)
library(xtable)

# Importing dataset
df.adult.full = read.csv("Datasets//adult.csv", strip.white = TRUE)
df.adult_train = read.table("Datasets//adult_train.data", sep = ",",
                            strip.white = TRUE, col.names = colnames(df.adult.full))
df.adult_train$train_test_split = "train"
df.adult_test = read.table("Datasets//adult_test.data", sep = ",",
                           strip.white = TRUE, col.names = colnames(df.adult.full))
df.adult_test$income = str_remove(df.adult_test$income, "[:punct:]")
df.adult_test$train_test_split = "test"
df.adult = rbind(df.adult_train, df.adult_test)

# Changing factor columns to character columns
df.adult = df.adult %>% mutate(across(where(is.factor), as.character))

for (col in 1:ncol(df.adult)) {
  df.adult[, col] = ifelse(df.adult[, col] == "?", NA, df.adult[, col])
}

df.adult = na.omit(df.adult)
nrow(df.adult)
nrow(filter(df.adult, train_test_split == "train"))
nrow(filter(df.adult, train_test_split == "test"))

ncol(df.adult)

View(df.adult)
sapply(df.adult, class)

# Function for visualizing dataset in latex
print_latex_table_of_column_values_adult = function(df.adult) {
  list_discrete_classes_count = sapply(df.adult[, sapply(df.adult, class) == "character"],
                                       table)
  list_discrete_classes_count
  vec_classes_min = sapply(df.adult[, sapply(df.adult, class) %in% c("numeric", "integer")],
                           min)
  vec_classes_max = sapply(df.adult[, sapply(df.adult, class) %in% c("numeric", "integer")],
                           max)
  df_classes_count = data.frame("")
  
  df.variables = data.frame(Column = colnames(df.adult), Values = "", stringsAsFactors = FALSE)
  for (name in names(list_discrete_classes_count)) {
    row = df.variables$Column == name
    classes = list_discrete_classes_count[[name]]
    temp_categories = ""
    for (i in seq_along(classes)) {
      if (i > 1) temp_categories = paste0(temp_categories, ", ")
      temp_categories = paste0(temp_categories, names(classes)[i],
                               " (", signif(classes[i] / nrow(df.adult) * 100, 2), "%)")
    }
    df.variables[row, "Values"] = temp_categories
  }
  for (i in seq_along(vec_classes_min)) {
    row = df.variables$Column == names(vec_classes_min)[i]
    df.variables[row, "Values"] = paste0("[", vec_classes_min[i], ", ",
                                         vec_classes_max[i], "]")
  }
  print(xtable(df.variables[df.variables$Column != "train_test_split",]),
        include.rownames = FALSE)
}

print_latex_table_of_column_values_adult(df.adult)

ggplot(df.adult, aes(x = fnlwgt, y = ..density.., group = income, fill = income)) +
  geom_histogram(alpha = 0.5)

table(df.adult$workclass, df.adult$income) / diag(table(df.adult$workclass,
                                                        df.adult$workclass))

# Editing dataset. Merging categories etc
df.adult.rv = df.adult
df.adult.rv$native.country = ifelse(as.character(df.adult$native.country) ==
                                      "United-States", "US", "Non-US")
df.adult.rv$marital.status = ifelse(df.adult$marital.status %in% c("Married-AF-spouse",
                                                                   "Married-civ-spouse",
                                                                   "Married-spouse-absent"),
                                    "Married", df.adult$marital.status)
df.adult.rv$workclass = ifelse(df.adult$workclass %in% c("Local-gov",
                                                         "Federal-gov",
                                                         "State-gov"),
                               "Government", df.adult$workclass)
df.adult.rv$education[df.adult.rv$education %in%
                        c("10th", "11th", "12th", "1st-4th", "5th-6th",
                          "7th-8th", "9th", "Preschool")] = "<=12th"

df.adult.rv = df.adult.rv[, ! colnames(df.adult.rv) %in% c("relationship")]
sapply(df.adult.rv[, sapply(df.adult.rv, class) == "character"], table)
View(df.adult.rv)

print_latex_table_of_column_values_adult(df.adult.rv)

df.adult.rv.train = filter(df.adult.rv, train_test_split == "train")
df.adult.rv.train = df.adult.rv.train[, colnames(df.adult.rv.train) != "train_test_split"]

df.adult.rv.test = filter(df.adult.rv, train_test_split == "test")
df.adult.rv.test = df.adult.rv.test[, colnames(df.adult.rv.test) != "train_test_split"]

# Save the edited train and test datasets
fwrite(df.adult.rv.train, "Datasets//df_adult_edited_train.csv")
fwrite(df.adult.rv.test, "Datasets//df_adult_edited_test.csv")
