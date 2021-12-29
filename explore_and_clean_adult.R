library(dplyr)
library(data.table)
library(ggplot2)
df.adult = read.csv("Datasets//adult.data")#, strip.white = TRUE)

df.adult = df.adult %>% mutate(across(where(is.factor), as.character))
View(df.adult)
sapply(df.adult, class)
sapply(df.adult[, sapply(df.adult, class) == "character"], table)

ggplot(df.adult, aes(x = fnlwgt, y = ..density.., group = Salary, fill = Salary)) + geom_histogram(alpha = 0.5)


df.adult.rv = df.adult
#df.adult.rv$Race = ifelse(df.adult$Race == "White", "White", "Non-white")
df.adult.rv$Country = ifelse(as.character(df.adult$Country) == "United-States", "US", "Non-US")
df.adult.rv$Relationship = ifelse(df.adult$Relationship %in% c("Husband", "Wife"), "Husband/Wife", "Non-Husband/Wife")
df.adult.rv$Occupation = df.adult$Occupation
df.adult.rv$Marital.Status = ifelse(df.adult$Marital.Status %in% c("Married-AF-spouse", "Married-civ-spouse", "Married-spouse-absent"),
                                    "Married", df.adult$Marital.Status)
df.adult.rv$Workclass = ifelse(df.adult$Workclass == "Private", "Private",
                               ifelse(df.adult$Workclass %in% c("Local-gov", "Federal-gov"), "Government",
                                      "Other"))
df.adult.rv$Education[df.adult.rv$Education %in%
                        c("10th", "11th", "12th", "1st-4th", "5th-6th", "7th-8th", "9th", "Preschool")] = "<=12th"

df.adult.rv = df.adult.rv[, ! colnames(df.adult.rv) %in% c("Relationship", "Capital.Loss")]
sapply(df.adult.rv[, sapply(df.adult.rv, class) == "character"], table)
View(df.adult.rv)

fwrite(df.adult.rv, "df_adult_edited.csv")