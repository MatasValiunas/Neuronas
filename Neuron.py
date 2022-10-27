import random
import math

# Class for storing a single line of data
# Atributes [x₁, x₂, ...] and label 0/1
class DataLine:
    def __init__(self, Attributes, label):
        self.Attributes = Attributes
        self.label = label                  


# Reading Data from a file and returning a list
def FileReading(fileName):
    Data = []
    with open(fileName) as file:
        for line in file:
            Data.append(DataLine([float(elem) for elem in line.split(",")[:-1]],  int(line.split(",")[-1])))
    
    return Data

# Net input function which is a weighted sum of all the inputs to the neuron
def NetInput(Inputs):
    sum = Weights[-1]
    for i in range(len(Inputs)):
        sum += Inputs[i] * Weights[i]
    
    return sum

def StepFunction(Inputs):
    if NetInput(Inputs) > 0:
        return 1
    else:
        return 0

def SigmoidFunction(Inputs):
    return 1 / (1 + pow(math.e, -1 * NetInput(Inputs)))

# Function which shows how bad is the neuron
def LossFunction(Data):
    sum = 0
    for data in Data:
        sum += pow(ActivasionFunction(data.Attributes) - data.label, 2)

    return sum / 2

# Function which trains a neuron by changing weights and returns a list of loss function results of each epoch
def Training(Data):
    iterations = 0
    Loss = []
    while LossFunction(Data) > 0 and iterations < generations:
        Loss.append(LossFunction(Data))
        for data in Data:
            activasion = ActivasionFunction(data.Attributes)
            if activasion != data.label:
                Weights[-1] += learningRate * (data.label - activasion)
                for i in range(len(Weights) - 1):
                    Weights[i] += learningRate * (data.label - activasion) * data.Attributes[i]
        iterations += 1

    return Loss

# Function which tests how many % of data the neuron guessed correctly
def Testing(Data):
    passed = 0
    for data in Data:
        if round(ActivasionFunction(data.Attributes)) == data.label:
            passed += 1

    return str(round(passed / len(Data) * 100, 2)) + "%"


if __name__ == "__main__":
    learningRate = 0.1           # How fast the weights change (0, 1]
    generations = 10             # How many cycles the learning happens through the same data
    trainingDataSize = 0.8       # How much of the data is assigned for training (0.8 = 80% for training / 20% for testing)

    #if input("Which function should be used for neuron activation: Step [1] / Sigmoid [2] - ") == "1":
    #    ActivasionFunction = StepFunction
    #else:
    #    ActivasionFunction = SigmoidFunction

    #if input("Which data should be used: Iris [1] / Breast Cancer [2] - ") == "1":
    #    Data = FileReading("iris.data")
    #else:
    #    Data = FileReading("breast-cancer-wisconsin.data")

    ActivasionFunction = StepFunction
    #ActivasionFunction = SigmoidFunction
    #Data = FileReading("iris.data")
    Data = FileReading("breast-cancer-wisconsin.data")

    random.shuffle(Data)         # Shuffles the data

    TrainingData, TestingData = Data[:int(len(Data) * trainingDataSize)], Data[int(len(Data) * trainingDataSize):]      # Assigns data to training and testing

    # Weights [w₁, w₂, ..., w₀], w₀ being bias
    Weights = [random.random() for w in range(len(Data[0].Attributes) + 1)]          # Generates a list of random weights 

    Loss = Training(TrainingData)

    print("Svoriai: ", [round(w, 2) for w in [Weights[-1]] + Weights[:-1]])          # [w₀, w₁, w₂, ...]
    print("Tikslumas mokymo duomenims:", Testing(TrainingData))
    print("Tikslumas testavimo duomenims:", Testing(TestingData))
    print("Kiekvienos epochos paklaidos:", [round(l, 2) for l in Loss])
    print("Paklaida testavimo duomenims:", round(LossFunction(TestingData), 2))