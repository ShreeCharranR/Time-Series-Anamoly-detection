# Time-Series-Anamoly-detection
Autoencoders

# Timeseries anomaly detection using an Autoencode
- Uses a reconstruction convolutional autoencoder model to detect anomalies in timeseries data.
- Data -[Numenta Anomaly Benchmark(NAB)](https://www.kaggle.com/boltzmannbrain/nab) dataset. It provides artificaltim eseries data containing labeled anomalous periods of behavior. Data are ordered, timestamped, single-valued metrics.
- We will use the `art_daily_small_noise.csv` file for training and the `art_daily_jumpsup.csv` file for testing. 
- Build convolutional reconstruction autoencoder model. The model will take input of shape `(batch_size, sequence_length, num_features)` and return output of the same shape. In this case, `sequence_length` is 288 and `num_features` is 1.


## Detecting anomalies

We will detect anomalies by determining how well our model can reconstruct
the input data.


1.   Find MAE loss on training samples.
2.   Find max MAE loss value. This is the worst our model has performed trying
to reconstruct a sample. We will make this the `threshold` for anomaly
detection.
3.   If the reconstruction loss for a sample is greater than this `threshold`
value then we can infer that the model is seeing a pattern that it isn't
familiar with. We will label this sample as an `anomaly`.

![anamoly](/anamoly.PNG)
