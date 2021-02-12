# NeuropixelsRegistration
 Motion estimation and registration of Neuropixels data.
 
 To get motion estimates, run "NPregistration.m" after having provided file paths for the recording and the channel maps.
 
 Input:
 Recording file (.bin,.h5.,.continuous),
 Channel map (.mat),
 Sampling rate (Hz),
 Time bin length (seconds)
 
 Output:
 Motion estimates
 Interpolated registered data (in same input format) [Under construction]


![Demo](https://github.com/evarol/NeuropixelsRegistration/blob/master/fig1.png)


![Demo](https://github.com/evarol/NeuropixelsRegistration/blob/master/fig1.png)

Data used in this figure is provided by International Brain Laboratory:
http://internationalbrainlab.org/data/2021-02-12_Varol


Reference:

Erdem Varol, Julien Boussard$, Nishchal Dethe, Olivier Winter, Anne Ura,The International Brain Laboratory, Anne Churchland$, Nick Steinmetz, Liam Paninski

Decentralized motion inference and registration of Neuropixels data. 

In ICASSP 2021 (To appear).

