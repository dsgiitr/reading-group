# Distilling-the-Knowledge-in-a-Neural-Network
This is an implementation of a part of the paper "Distilling the Knowledge in a Neural Network" (https://arxiv.org/abs/1503.02531). 

Teacher network has two hidden layers with 1200 units in each layer. It is trained on MNIST with data augmentation and achieves 108 test errors.

Student network has one hidden layer with 400 units. No regularization techniques are used to train student network except weight regularization. Without distillation, it achieves 181 test errors. With distillation, the test errors reduces to 134. This demonstrates the knowledge transfer happening from teacher to student, helping the student to generalize better.

# Training and testing teacher and student network
For training teacher network, run all cells of *distill_basic_teacher.ipynb*.
For training student network, run all cells of *distill_basic_student.ipynb*.
Modify second cell of both notebooks according to the availability of GPU.
