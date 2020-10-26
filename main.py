from neural_network import neuralNetwork
import numpy
import matplotlib.pyplot as plt


#number of input,hidden and output nodes
input_nodes = 784
hidden_nodes = 100
output_nodes = 10
#learingrate为0.5
learning_rate = 0.2

#create instance of neutal network
neural  = neuralNetwork(input_nodes,hidden_nodes,
                        output_nodes,learning_rate)

#print("befor training:" ,neural.wih,neural.who)

training_data_file = open("mnist_dataset/mnist_train_100.csv",'r')
training_data_list = training_data_file.readlines()
training_data_file.close()



#将一维数组拆分成二位数组，方便绘制
#image_array = numpy.asfarray(all_values[1:]).reshape((28,28))

#plt.imshow(image_array,cmap='Greys',interpolation='None')

#开始训练
#调整输入数据
for record in training_data_list:
    # 将每一行按值拆分成数组
    all_values = record.split(',')
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #print(scaled_input)
    #设置输出数据，设置输出节点为10
    targets = numpy.zeros(output_nodes)+0.01
    targets[int(all_values[0])] =0.99
    neural.train(inputs,targets)
    pass

#测试数据


#print("after training:" ,neural.wih,neural.who)

#load the mnist test data file a list

test_data_file = open("mnist_dataset/mnist_test.csv",'r')
test_data_list = test_data_file.readlines()
test_data_file.close()


#test the neutal network
#scorecard for how well the network performs
scorecard = []
count = 0
count_sum = 0
for record in test_data_list:
    count_sum +=1
    values = record.split(',')
    #find the correct answer
    correct_label = int(values[0])
    print(correct_label,"is correct_label")
    test_inputs = (numpy.asfarray(values[1:])/255.0*0.99)+0.1
    #query the network
    test_outputs = neural.query(test_inputs)
    #the index of the highest value corresponds to the label
    label = numpy.argmax(test_outputs)
    print(label,"is network answer")
    if (correct_label == label):
        scorecard.append(1)
        count +=1
    else:
        scorecard.append(0)
        pass
    pass

print(scorecard)
scorecard_array = numpy.asarray(scorecard)
print("correct rate:" ,scorecard_array.sum()/scorecard_array.size*100,"%")




plt.show()