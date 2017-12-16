# To create the training and test set from qrel

sh createDataset.sh #each entry in the training/test set will be like "1 FBIS3-10082 greiff ajedrecista mafioso cali capo rodriguez colombian cartel gilberto surrend"

# To convert training/test entries to the form "1 FBIS3-10243  [ -1.28749970e+01   6.57353000e-01  -2.34869000e+00  -9.16392000e-01
   #3.46384600e+00  -5.72066100e+00  -4.71916000e+00  -1.04431710e+01
#  -3.28261000e-01   7.66130400e+00  -9.80913000e-01   6.23358900e+00
 # -1.36796820e+01  -2.57439200e+00   1.88097800e+00   8.48789700e+00
 # -5.24550000e-01   6.83952900e+00   1.13805200e+00 ]

python readEmbedding-anurag.py > output1.txt

# To convert training/test entries to the form "1 FBIS3-10243 -1.28749970e+01 6.57353000e-01 -2.34869000e+00 -9.16392000e-01 3.46384600e+00 -5.72066100e+00 -4.71916000e+00 -1.04431710e+01 -3.28261000e-01 7.66130400e+00 -9.80913000e-01 6.23358900e+00 -1.36796820e+01 -2.57439200e+00 1.88097800e+00 8.48789700e+00

cat output1.txt | tr "\n" " "|sed 's/\[//g'|sed 's/\]//g'|tr -s " " > output2.txt

# Finally running the code

python Siamese-networks-medium-complete-kripa.py
