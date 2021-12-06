import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz

fig, ax = plt.subplots()

#Inspecting the data
#print(aaron_judge.columns)
#print(aaron_judge.description.unique())
#We’re interested in looking at whether a pitch was a ball or a strike. That information is stored in the type feature
#Looking at how balls and strikes are recorded, that is the labels
print(aaron_judge.type.unique())
#Changing the values of the labels from strings to numbers
#Creating a function to calculate SVM scores for all three players
def calculate_svm_score(player):
  player["type"] = player.type.map({"S": 1, "B": 0})
  #print(player.type.unique())
  #We want to predict whether a pitch is a ball or a strike based on its location over the plate, which can be found in plate_x and plate_z
  #print(aaron_judge["plate_x"])
  #Dropping every row that has a NaN in any of the three columns we are goin to work with
  player = player.dropna(subset = ["plate_x", "plate_z", "type"])
  #print(aaron_judge.head())
  #Plotting the plates x and y
  plt.scatter(player.plate_x, player.plate_z, c = player.type, cmap = plt.cm.coolwarm, alpha = 0.25)
  #Creating an SVM to create a decision boundary
  #Splitting the data into a training set and validation set
  training_set, validation_set = train_test_split(player, random_state = 1)
  classifier = SVC(kernel = "rbf", gamma = 3, C = 1)
  classifier.fit(player[["plate_x", "plate_z"]], player.type)
  #ax.set_ylim(-2, 2)
  draw_boundary(ax, classifier)
  plt.show()
  #Finding the accuracy of the classifier
  return (classifier.score(validation_set[["plate_x", "plate_z"]], validation_set.type))
#Changing some of the parameters of the SVM like gamma and C to see if the score will be better

print(calculate_svm_score(aaron_judge))
print(calculate_svm_score(jose_altuve))
print(calculate_svm_score(david_ortiz))