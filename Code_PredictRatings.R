#######################################################################
# ANALYTICAL DATA MINING PROJECT                                      #
# TITLE         :  MOVIE RECOMMENDER SYSTEM BY PREDICTING RATINGS     #
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

# Data Preprocessing
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
e <- evaluationScheme(r, method="split", train=0.67,
                      given=15, goodRating=3.5)

# Create a recommender object (model)
#       They pertain to four different algorithms.
#        UBCF: User-based collaborative filtering
#        IBCF: Item-based collaborative filtering
#        Popular
#        Random
#      Parameter 'method' decides similarity measure
rec_UBCF=Recommender(getData(e,"train"),method="UBCF", param=list(normalize = "Z-score", method="Pearson",nn=5))
rec_IBCF=Recommender(getData(e,"train"),method="IBCF", param=list(normalize = "Z-score",method="Jaccard"))
rec_P=Recommender(getData(e,"train"),method="POPULAR")
rec_R=Recommender(getData(e,"train"),method="RANDOM")

############Create predictions#############################
# This prediction does not predict movie ratings for test.
#   But it fills up the user 'X' item matrix so that
#    for any userid and movieid, we can find predicted rating
#      'type' parameter decides whether you want ratings or top-n items
#         get top-10 recommendations for a user, as:
#             predict(rec, r[1:nrow(r)], type="topNList", n=10)
p1 <- predict(rec_UBCF, getData(e, "known"), type="ratings")
p2 <- predict(rec_IBCF, getData(e, "known"), type="ratings")
p3 <- predict(rec_P, getData(e, "known"), type="ratings")
p4 <- predict(rec_R, getData(e, "known"), type="ratings")

# Error calculation
error <- rbind(
  calcPredictionAccuracy(p1, getData(e, "unknown")),
  calcPredictionAccuracy(p2, getData(e, "unknown")),
  calcPredictionAccuracy(p3, getData(e, "unknown")),
  calcPredictionAccuracy(p4, getData(e, "unknown")))

rownames(error) <- c("UBCF","IBCF","POPULAR","RANDOM")

#Comparison of Algorithms
algorithms <- list(
  "random items" = list(name="RANDOM", param=NULL),
  "popular items" = list(name="POPULAR", param=NULL),
  "user-based CF" = list(name="UBCF", param=list(normalize = "Z-score", method="Pearson",nn=5)),
  "item-based CF" = list(name="IBCF", param=list(normalize = "Z-score", method="Jaccard")))

# run algorithms
results <- evaluate(e, algorithms, type = "ratings")

# Plot the comparison
plot(results, annotate = TRUE, col = rainbow(4), ylim = c(0,5))
