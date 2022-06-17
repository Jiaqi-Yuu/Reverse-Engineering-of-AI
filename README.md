# Reverse-Engineering-of-AI
With its commercialization, machine learning as a service (MLaaS) is becoming more and more popular, and providers are paying more attention to the privacy of models and the protection of intellectual property. Generally speaking, the machine learning service deployed on the cloud platform is a black box, where users can only obtain outputs by providing inputs to the model. The attributes of the model such as architecture, training set, training method, are concealed by provider. In this work, we devise a multi-discriminator generative adversarial network(MDGAN) to learn domain invariant features. Based on these features, we can learn a domain-free model to inversely infer the attributes of a target black-box model with unknown training data. The experiments consume a lot of computing resources, so I provide some prepared data. I only show part of this work.

# Requirement
PyTorch == 1.10.2

# Run
```python
python train.py
```

# prepared_data
We pretrained a number of models which are constructed by enumerating all posible attribute values. The details of the attributes and their values are shown in Table1. We sample 5000, 1000 from white-box models as the training set and testing set. 
The 'prepared_outputs_train(cartoon&sketch)' contains 10000 models' outputs(5000 from cartoon, 5000 from sketch). 
The inputs is 100 queries listed in 'save_query(cartoon&sketch)'
With the data prepared in advance, the training efficiency of the model is greatly improved!
![在这里插入图片描述](https://img-blog.csdnimg.cn/09d6adbbc16e4aaeadd44b30525cfe4c.jpeg#pic_center)

# future
I will provide more details of relevant work in the future.
