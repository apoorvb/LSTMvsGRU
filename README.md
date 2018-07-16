# LSTMvsGRU
In this project, I have used 3 layer LSTM and GRU models. 
I ran the same dataset on the same number of layers of both GRU and LSTM
This was done to compare their level of accuracies and loss.
2 Versions of these models were used. 
Difference between models is the output ldimension of the embedding layer.
It is usually suggested that the range of values for this layer should be between 100-300, that is why i took 2 values, <100 and >100.
output_dim gives the length of embedding vector in the vector space. It is an hyper parameter.
imdb Dataset that comes along with keras was used.


# Results
It is observed that both LSTM and GRU showed ~84% accuracies when output_dims = 8
It is observed that both LSTM and GRU showed ~89% accuracies when output_dims = 128

# Conclusion
Hnece, it can be concluded that the both models have almost the same accuracy.
Also, when output_dim of embedding layers is between 100-300, it gives better results
