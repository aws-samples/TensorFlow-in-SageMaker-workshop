# Running your TensorFlow Models in SageMaker Workshop

TensorFlow™ enables developers to quickly and easily get started with deep learning in the cloud. 
The framework has broad support in the industry and has become a popular choice for deep learning research and application development, particularly in areas such as computer vision, natural language understanding and speech translation.
You can get started on AWS with a fully-managed TensorFlow experience with Amazon SageMaker, a platform to build, train, and deploy machine learning models at scale.

## Use Machine Learning Frameworks with Amazon SageMaker
The Amazon SageMaker Python SDK provides open source APIs and containers that make it easy to train and deploy models in Amazon SageMaker with several different machine learning and deep learning frameworks. For general information about the Amazon SageMaker Python SDK, see https://sagemaker.readthedocs.io/.

You can use Amazon SageMaker to train and deploy a model using custom TensorFlow code. The Amazon SageMaker Python SDK TensorFlow estimators and models and the Amazon SageMaker open-source TensorFlow containers make writing a TensorFlow script and running it in Amazon SageMaker easier.

In this workshop you will port a working TensorFlow script to run on SageMaker and utilize some of the feature available for TensorFlow in SageMaker

The workshop is based on 5 modules:

1. [Porting a TensorFlow script to run in SageMaker using SageMaker script mode.](0_Running_TensorFlow_In_SageMaker.ipynb)
2. [Monitoring your training job using TensorBoard and Amazon CloudWatch metrics.](1_Monitoring_your_TensorFlow_scripts.ipynb)
3. [Optimizing your training job using SageMaker pipemode input.](2_Using_Pipemode_input_for_big_datasets.ipynb)
4. [Running a distributed training job.](3_Distributed_training_with_Horovod.ipynb)
5. [Deploying your trained model on Amazon SageMaker.](4_Deploying_your_TensorFlow_model.ipynb)

## License Summary

This sample code is made available under the MIT-0 license. See the LICENSE file.
