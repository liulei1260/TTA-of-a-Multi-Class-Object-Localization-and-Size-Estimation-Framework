install.packages("idefix")
install.packages("reshape2")
install.packages("dplyr")
library(dplyr)
library(reshape2)
library(idefix)
library(survival)

levels <- c(3,3,3,3,3,3,3,3,3,3,3,3,3)
coding <-c("E","E","E","E","E","E","E","E","E","E","E","E","E")

Profiles (lvls=levels, coding=coding)

data<-data[2:ncol(data)]
personid <- rownames(data)
personid <- as.integer(personid)
data <- cbind(personid, data)

data <- melt(data, id.vars=c("personid"))

data <-rbind(data, data, data)
data <- data[order(data$personid, data$variable),]
x <- nrow(data)/3
alt <- rep(1:3, x)
data <- cbind(data, alt)
cs <- rep(1:x, each= 3)
cs <- sort(cs)
data <- cbind(data,cs)

data <- mutate(data, choice=ifelse(value == "Bundle 1" & alt=="1" | value== "Bundle 2" & alt=="2" | value== "Bundle 3" & alt=="3", 1, 0))


resultsCLM <- clogit(choice~ x11 + x12 + x21 + x22 + x31 + x32 + x41 + x42 + x51 +x52 + x61 + x62 + x71 + x72 +
                       x81  + x82 + x91 + x92 + x101 + x102 + x111 + x112 + x121 + x122 + x131 + x132 + strata (cs), data=final,method="approximate")
summary(resultsCLM)

resultsCLM1 <- clogit(choice~ x51 +x52 + x61 + x62 + x71 + x72 +
                       x81  + x82 + x91 + x92 + x101 + x102 + x121 + x122 + x131 + x132 + strata (cs), data=final)
summary(resultsCLM1)

D <- dfidx(final, choice="choice", idx = list(c("cs", "personid"), "alt"), idnames= c("cs", "alt"))
resultsXLM <- mlogit(choice ~  x11 + x12 + x21 + x22 + x31 + x32 + x41 + x42 + x51 +x52 + x61 + x62 + x71 + x72 +
      x81  + x82 + x91 + x92 + x101 + x102 + x111 + x112 + x121 + x122 + x131 + x132 | 0, data=final, rpar=c(x11="n", x12="n", x21="n", x22="n",
      x31="n", x32="n", x41="n", x41="n", x51="n", x52="n", x61="n", x62="n", x71="n", x72="n", x81="n", x82="n", x91="n", x92="n", x101="n", x102="n", x111="n", x112="n", x121="n", x122="n", x131="n", x132="n"), R=100, halton=NA, panel=TRUE)
summary(resultsXLM)

