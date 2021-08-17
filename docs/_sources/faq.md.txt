## FAQ  
### Q: Where do I find good literature about reservoir computing ? Do you have some paper or book to recommend?  

**A:** For a comparison between different RNN methods and RC, as well as a demonstration of RC's predictive power and speed:  
*Chattopadhyay, A.; Hassanzadeh, P.; Palem, K.; Subramanian, D. Data-Driven Prediction of a Multi-Scale Lorenz 96 Chaotic System Using a Hierarchy of Deep Learning Methods: Reservoir Computing, ANN, and RNN-LSTM. 2019, 1–21.*

For a quick introduction:  
*Haluszczynski, A.; Räth, C. Good and Bad Predictions: Assessing and Improving the Replication of Chaotic Attractors by Means of Reservoir Computing. Chaos 2019, 29.*  
It explains the basic math quite well and applies RC to the canonical Lorenz system. Furthermore, it discusses possibly the biggest distinguishing feature of RC, the static randomness of it's network and the resulting consequences, which is very important to understand when working with RC.

For some "classical" RC literature, the paper(s) by Jaeger et. al.:  
*Lukoševičius, M.; Jaeger, H. Reservoir Computing Approaches to Recurrent Neural Network Training. Comput. Sci. Rev. 2009, 3, 127–149.*  
It discusses the basics and the many possible variations of RC in addition to its development history. This paper is of course not quite up to date anymore, as it's 10 years old by now, but the qualitative results and ideas still hold true today.

### Q: My predictions don't agree with the real data at all! What is going on?  

**A:** The performance of the reservoir depends strongly on the hyperparameters.  
Nobody really knows how _exactly_ the prediction quality depends on the parameters, but as a start you should use a reservoir that has about 100x as many nodes as the input has dimensions.  More for real or more "complicated" data.  
For all other parameters it is advisable to just play around by hand to see what parameter (ranges) might work or to use the hyperparameter optimization algorithm of your choice. As RC is fast to train, a simple grid search is often sufficient   
 
### Q: For the exact same set of hyperparameters the prediction is sometimes amazingly accurate and at other times utterly wrong. Why?  

**A:** As the network is not trained at all, it can happen that the randomly generated network is not suited to learn the task at hand. How "good" and "bad" network topologies differ is an active research problem.  
Practically, this means that the prediction quality for a set of hyperparameters can not be determined by a single randomly generated network but instead must be calculated as e.g. the average of multiple randomly generated networks.  
Luckily, unsuccessful predictions are almost always very clearly distinguishable if the network topology is at fault, i.e. the prediction either works well or not at all. 

### Q: You said above that the network should have about 100x as many nodes as the input has dimensions, maybe more. My input is >50 dimensional and with 5000-10000 nodes the training and prediction is annoyingly slow! I thought RC was supposed to be fast, what's going on?  

**A:** The computational bottleneck of RC is a bunch of matrix multplications which, roughly, scale as O(n^3), where n is the number of nodes in the network. Therefore, just scaling up the network to accommodate larger and larger inputs doesn't work.  
Luckily there is a potential solution to this problem in the method of [local states](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.024102). This is one of the features we already have code written for, but did not yet implemented in the package.  
If you need to use RC for higher dimensional inputs _now_, it is of course always helpful to reduce the input dimensionality as much as possible, for example via an autoencoder.

### Q: In your code, I see you discard some samples at the beginning of training, why is it necessary?  

**A:** this is done for two reasons:

1\. To get rid of the transient dynamics in the reservoir, i.e. to "synchronize" the reservoir state r(t) with the trajectory, before the training starts. 
If one were to omit this step, the initial reservoir state would not correspond to the initial position on the trajectory, resulting in erroneous training results. For more details, i'd recommend reading up on the "echo state property" of RC and Echo State Networks in general.

2\. To discard any transient behavior in the training data itself. 
When trying to predict a chaotic attractor, the training data should only contain points on that attractor, otherwise the neural network learns the (irrelevant) dynamics of the system before the attractor is reached, which will decrease it's predictive quality on the attractor itself. 

Discarding the transient behavior in the data (2.) basically just amounts to data preprocessing and is not necessary for RC per se (one could always just use input data that does already start on the attractor) but as the synchronization between reservoir and trajectory (1.) is necessary, we use the "discard_steps" variable to accomplish both.

### Q: Do I need to denoise my data?  

**A:** Yes, if you work with real data it should be denoised, removing as much noise as possible without loosing too much information. It's the same as for all other machine learning techniques, really.
