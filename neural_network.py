import numpy
#scipy.special for the sigmod function expit(),S函数
import scipy.special
#neural network class definition
class neuralNetwork :

    #initialise the neural network
    def __init__(self,inputnodes,hiddennodes,outputnoddes,
                 learningrate) :
        #set number of nodes in each input,hidden,outputlayer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnoddes

        #link weight matrices ,wih and who
        self.wih = numpy.random.normal(0.0,pow(self.hnodes,-0.5),
                                       (self.hnodes,self.inodes))
        self.who = numpy.random.normal(0.0,pow(self.onodes,-0.5),
                                        (self.onodes,self.hnodes))
        #定义次函数为S函数，方便修改为其他函数
        self.activation_function = lambda x:scipy.special.expit(x)
        #learning rate
        self.lr = learningrate
        pass

    #train the neural network
    def train(self,input_list,targets_list) :
        #convert inputs list to 2d array
        inputs = numpy.array(input_list,ndmin =2).T
        targets = numpy.array(targets_list,ndmin = 2).T
        #calculate signals into hidden layer
        hidden_inputs = numpy.dot(self.wih,inputs)
        #calculate signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into outputs layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #calculate signal emerging from outputs layer
        final_outputs = self.activation_function(final_inputs)

        #error is the (target -actual)
        outputs_error = targets-final_outputs
        #error in hidden
        hidden_error = numpy.dot(self.who.T,outputs_error)

        #update the weights for the links between the hidden and output layers
        self.who +=self.lr*numpy.dot(outputs_error*final_outputs*(1.0-final_outputs),
                                      numpy.transpose(hidden_outputs))
        #update the weights for the links between the input and hidden layers
        self.wih +=self.lr*numpy.dot(hidden_error*hidden_outputs*(1.0-hidden_outputs),
                                      numpy.transpose(inputs))
        pass

    #query the neural network
    def query(self,input_list) :
        #convert inputs list to 2d array
        inputs = numpy.array(input_list,ndmin=2).T
        #print(inputs)
        #calculate signals into hidden layers
        hidden_inputs = numpy.dot(self.wih, inputs)
        #calculate the signals emerging from hidden layer
        hidden_outputs = self.activation_function(hidden_inputs)

        #calculate signals into output layer
        final_inputs = numpy.dot(self.who,hidden_outputs)
        #calculate signals emerging from final output layers
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
        pass

    #print sth for this class
    def getsth(self):
        print("input_nodes:",self.inodes)
        print("hiddenput_nodes:",self.hnodes)
        print("output_nodes:",self.onodes)
        print("learningrate:",self.lr)
        print(self.wih)
        print(self.who)
        pass
