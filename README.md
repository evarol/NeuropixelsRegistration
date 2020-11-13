# NeuropixelsRegistration
 Motion estimation and registration of Neuropixels data.
 
 To get motion estimates, run "NPregistration.m" after having provided file paths for the recording and the channel maps.
 
 Input:
 Recording file (.bin,.h5.,.continuous),
 Channel map (.mat),
 Sampling rate (hZ),
 Time bin length (seconds)
 
 Output:
 Motion estimates
 Interpolated registered data (in same input format) [Under construction]


![Demo](https://github.com/evarol/NeuropixelsRegistration/blob/master/fig1.png)

References:

Erdem Varol, Julien Boussard, Nishchal Dethe, Liam Paninski.

Decentralized motion inference and registration of Neuropixels data. 

In ICASSP 2021 (In Review)

