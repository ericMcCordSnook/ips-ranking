# Nikhil Bhaip

# First time running the program uncomment line below:
# install.packages("Rankcluster")
library(Rankcluster)

# From documentation:
# link: https://cran.r-project.org/web/packages/Rankcluster/Rankcluster.pdf

############################################################################
# APA dataset
# This dataset contains the 5738 full rankings resulting from the American Psychological Association
# (APA) presidential election of 1980. For this election, members of APA had to rank five candidates
# in order of preference.
# 5738 by 5

data("APA")
write.csv(APA$data, file="APA.csv")
############################################################################
# Soccer (football?) dataset 
# Multidimensional rank data

# Good for comparing between two different but related rankings for 21 seasons
# UEFA ranks vs Premier league ranks (from 1 to 4)
# 21 by 8 
# Contains a couple ties

data("big4")
write.csv(big4$data, file="soccer.csv")
############################################################################
# Quizzes dataset 
# Multidimensional rank data
# Has an actual ground truth! 

# This dataset contains the answers of 70 students from Polytech’Lille (statistics engineering 
# school, France) to the four rank-based quizzes in literature, soccer, math, and film.
# See documentation for information about quizzes

# 70 by 16
# The student’s answers are in row and the 16 columns correspond to
# the 4 rankings (for the 4 quizzes) of size 4 (ranking representation).

data("quiz")
write.csv(quiz$data, file="quiz.csv")

############################################################################
# Sports dataset

# 130 students at the University of Illinois ranked seven sports according to their 
# preference in participating: 
# A = Baseball, B = Football,  C = Basketball, D = Tennis, E = Cycling, F = Swimming, G = Jogging

# 130 by 7 

data("sports")
write.csv(sports$data, file="sports.csv")

############################################################################

# Words dataset

# A sample of 98 college students were asked to rank five words according to strength of 
# association (least to most associated) with the target word "Idea": 
# A = Thought, B = Play, C = Theory, D = Dream and E = Attention

# 98 by 5
data("words")
write.csv(words$data, file="words.csv")


