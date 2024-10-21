from Decision_Tree.decision_tree import decision_tree
from CNN.cnn import cnn
from Naive_Bayes.naive_bayes import naive_bayes
from Multi_Layer_Perceptron.mlp import mlp
from data_management import get_dataset, resnet_18
from numpy import *

def main():
    return 0

if __name__ == "__main__":
    train_set, test_set = get_dataset(500, 100)

    train_features, test_features = resnet_18(train_set, test_set)

    print("\nTrain features shape:", train_features.shape)
    print("Test features shape:", test_features.shape)

    exit(0)

