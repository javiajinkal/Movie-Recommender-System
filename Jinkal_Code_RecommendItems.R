#######################################################################
# ANALYTICAL DATA MINING PROJECT                                      #
# TITLE         :  MOVIE RECOMMENDER SYSTEM BY RECOMMENDING TOP ITEMS #
# NAME          : JINKAL ARVIND JAVIA (SUID: 425325424)               #
#######################################################################

# Set data path as per your data file (for example: "c://abc//" )
setwd("~/Desktop/data_mining_project/working")

# If not installed, first install following three packages in R
library(recommenderlab)
library(reshape2)
library(ggplot2)

cat("Processing...")

# Read training file along with header
tr<-read.csv("u_data.csv",header=FALSE)

# Data preprocessing
tr<-subset(tr,select=-4)

# Using acast to convert above data as follows:
#       m1  m2   m3   m4
# u1    3   4    2    5
# u2    1   6    5
# u3    4   4    2    5
g<-acast(tr, tr[,1] ~ tr[,2])

# Convert it as a matrix
R<-as.matrix(g)

# Convert R into realRatingMatrix data structure
# realRatingMatrix is a recommenderlab sparse-matrix like data-structure
r <- as(R, "realRatingMatrix")

#MODEL EVALUATION
e <- evaluationScheme(r, method="split", train=0.90,
                      given=15, goodRating=5)

# Create a recommender object (model)
#       They pertain to four different algorithms.
#        UBCF: User-based collaborative filtering
#        IBCF: Item-based collaborative filtering
#        Popular
#        Random
#      Parameter 'method' decides similarity measure
rec_UBCF=Recommender(getData(e,"train"),method="UBCF", param=list(normalize = "Z-score", method="Pearson",nn=7))
rec_IBCF=Recommender(getData(e,"train"),method="IBCF", param=list(normalize = "Z-score", method="Jaccard"))
rec_P=Recommender(getData(e,"train"),method="POPULAR")
rec_R=Recommender(getData(e,"train"),method="RANDOM")

############Create predictions#############################
p1 <- predict(rec_UBCF, getData(e, "known"),type = "topNList", n = 10)
p2 <- predict(rec_IBCF, getData(e, "known"),type = "topNList", n = 10)
p3 <- predict(rec_P, getData(e, "known"), type = "topNList",n = 10)
p4 <- predict(rec_R, getData(e, "known"),type = "topNList", n = 10)

#Error calculation
error <- rbind(
  calcPredictionAccuracy(p1, getData(e, "unknown"), given = 15, goodRating = 5),
  calcPredictionAccuracy(p2, getData(e, "unknown"), given = 15, goodRating = 5),
  calcPredictionAccuracy(p3, getData(e, "unknown"), given = 15, goodRating = 5),
  calcPredictionAccuracy(p4, getData(e, "unknown"), given = 15, goodRating = 5))

rownames(error) <- c("UBCF","IBCF","POPULAR","RANDOM")

#Comparison of Algorithms
algorithms <- list(
  "random items" = list(name="RANDOM", param=NULL),
  "popular items" = list(name="POPULAR", param=NULL),
  "user-based CF" = list(name="UBCF", param=list(normalize = "Z-score",method="Pearson",nn = 165)),
  "item-based CF" = list(name="IBCF", param=list(normalize = "Z-score",method="Jaccard",k = 2)))

# run algorithms
results <- evaluate(e, algorithms, type = "topNList", n=c(1, 3, 5, 10, 15, 20))

#Plot comparison results
plot(results, annotate=TRUE)
