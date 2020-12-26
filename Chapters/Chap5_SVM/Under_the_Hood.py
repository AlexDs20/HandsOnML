# How it works:
#
# Notations: b=bias
#            w=weights
#       (therefore we do not add a x_0=1)

# Decision function and Predictions
#
#   LinearSVM predicts the class of an instance x
#
#       y^\hat = 0 if h=w^Tx+b < 0
#       y^\hat = 1 if h=w^Tx+b >= 0
#
#   The decision boundary corresponds to the points where h=0
#
#       margin : h = +-1
#           --> as wide as possible without (with limited) margin violation: hard margin (soft margin)

# Training objectives:
#
#   If the slope of h is smaller (||w|| smaller) -> where h=+-1 (margins) get farther from the decision boundary -> wider "road"
#
#       --> the smaller the weight vector w -> the larger the margin
#
#
