# tensorflow

Based on mikalv simple text generator : https://gist.github.com/mikalv/3947ccf21366669ac06a01f39d7cff05

Changed the character based prediction to word based prediction.
The test was performed on TS 22.261 V16.2.0 that was converted to a txt from Word .doc
Also added a save/restore feature in order to start the training/prediction from saved data.

For training the text generator:
python3 trainTextGenV4.py -n 0 # start training from scratch
python3 trainTextGenV4.py -n 120 # start training from saved step 120

To perform a prediction you need to provide the saved model number and the batch number the seed will be from
python3 predictTextGenV4.py -n 120 -b 0 # will use the saved check point 120 and the batch 0 as seed
