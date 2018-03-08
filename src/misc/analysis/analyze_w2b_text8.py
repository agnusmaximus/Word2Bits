import sys
import matplotlib.pyplot as plt
import re

raw_data_loss = """
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1071769.750000
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -1415233.000000
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1659229.875000
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1190172.875000
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1221467.500000
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -753892.437500
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1061762.875000
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1045094.062500
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1313231.250000
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -832586.062500
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1304812.875000
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1221771.625000
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1318932.500000
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -997424.500000
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1552922.750000
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1325409.750000
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1237996.500000
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -1158675.750000
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1644165.500000
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1328016.000000
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1128149.375000
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -1299052.250000
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1674992.250000
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1259589.500000
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -2829206.500000
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -3486937.500000
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -3421035.000000
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -2825446.000000
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1797508.500000
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -1974106.250000
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1856376.375000
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1734178.625000
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1963575.875000
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -2135621.500000
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -2099206.000000
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1917069.500000
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -2220549.250000
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -2500144.250000
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -2475755.250000
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -2193741.750000
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -2437665.000000
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -2851394.000000
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -2809894.750000
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -2421905.750000
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -2595184.500000
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -3162244.500000
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -3093206.000000
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -2589464.000000
FINAL_vectors_datasettext8_epochs25_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -679846.062500
FINAL_vectors_datasettext8_epochs25_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -1261247.000000
FINAL_vectors_datasettext8_epochs25_size1000_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1382681.125000
FINAL_vectors_datasettext8_epochs25_size1000_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -953844.500000
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1126064.625000
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -833709.625000
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1050440.875000
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1007858.375000
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1161860.875000
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -815700.312500
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1247816.500000
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1121647.250000
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1065952.375000
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -957509.500000
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1421540.750000
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1170037.375000
FINAL_vectors_datasettext8_epochs25_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -922933.437500
FINAL_vectors_datasettext8_epochs25_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -1083053.250000
FINAL_vectors_datasettext8_epochs25_size600_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1462955.250000
FINAL_vectors_datasettext8_epochs25_size600_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1085512.750000
FINAL_vectors_datasettext8_epochs25_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -803777.750000
FINAL_vectors_datasettext8_epochs25_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -1184371.000000
FINAL_vectors_datasettext8_epochs25_size800_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1434678.750000
FINAL_vectors_datasettext8_epochs25_size800_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1019785.500000
FINAL_vectors_datasettext8_epochs50_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -467860.312500
FINAL_vectors_datasettext8_epochs50_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -1195149.500000
FINAL_vectors_datasettext8_epochs50_size1000_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1284468.500000
FINAL_vectors_datasettext8_epochs50_size1000_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -842619.000000
FINAL_vectors_datasettext8_epochs50_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1048019.125000
FINAL_vectors_datasettext8_epochs50_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -817689.250000
FINAL_vectors_datasettext8_epochs50_size100_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1056793.875000
FINAL_vectors_datasettext8_epochs50_size100_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -988726.875000
FINAL_vectors_datasettext8_epochs50_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -1053391.125000
FINAL_vectors_datasettext8_epochs50_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -799736.000000
FINAL_vectors_datasettext8_epochs50_size200_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1237093.375000
FINAL_vectors_datasettext8_epochs50_size200_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1090314.750000
FINAL_vectors_datasettext8_epochs50_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -915786.125000
FINAL_vectors_datasettext8_epochs50_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -943749.437500
FINAL_vectors_datasettext8_epochs50_size400_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1377081.875000
FINAL_vectors_datasettext8_epochs50_size400_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1092144.250000
FINAL_vectors_datasettext8_epochs50_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -743415.937500
FINAL_vectors_datasettext8_epochs50_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -1053669.000000
FINAL_vectors_datasettext8_epochs50_size600_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1378394.125000
FINAL_vectors_datasettext8_epochs50_size600_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -1025477.812500
FINAL_vectors_datasettext8_epochs50_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Loss: -595856.812500
FINAL_vectors_datasettext8_epochs50_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Loss: -1132967.250000
FINAL_vectors_datasettext8_epochs50_size800_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Loss: -1336001.000000
FINAL_vectors_datasettext8_epochs50_size800_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Loss: -907234.562500
"""

raw_data_acc = """
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 21.88 % Semantic accuracy: 25.31 % Syntactic accuracy: 19.44 %
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 29.32 % Semantic accuracy: 39.06 % Syntactic accuracy: 22.37 %
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 24.72 % Semantic accuracy: 31.32 % Syntactic accuracy: 20.02 %
FINAL_vectors_datasettext8_epochs10_size1000_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 23.30 % Semantic accuracy: 27.56 % Syntactic accuracy: 20.26 %
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 31.73 % Semantic accuracy: 35.53 % Syntactic accuracy: 29.03 %
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 5.64 % Semantic accuracy: 5.97 % Syntactic accuracy: 5.40 %
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 16.60 % Semantic accuracy: 21.09 % Syntactic accuracy: 13.41 %
FINAL_vectors_datasettext8_epochs10_size100_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 19.23 % Semantic accuracy: 22.78 % Syntactic accuracy: 16.70 %
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 33.02 % Semantic accuracy: 39.58 % Syntactic accuracy: 28.35 %
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 17.27 % Semantic accuracy: 19.42 % Syntactic accuracy: 15.73 %
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 26.72 % Semantic accuracy: 34.18 % Syntactic accuracy: 21.41 %
FINAL_vectors_datasettext8_epochs10_size200_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 28.58 % Semantic accuracy: 35.48 % Syntactic accuracy: 23.67 %
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 30.60 % Semantic accuracy: 37.82 % Syntactic accuracy: 25.45 %
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 28.69 % Semantic accuracy: 38.71 % Syntactic accuracy: 21.55 %
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 30.01 % Semantic accuracy: 40.53 % Syntactic accuracy: 22.51 %
FINAL_vectors_datasettext8_epochs10_size400_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 29.42 % Semantic accuracy: 37.57 % Syntactic accuracy: 23.61 %
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 27.87 % Semantic accuracy: 34.36 % Syntactic accuracy: 23.25 %
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 31.13 % Semantic accuracy: 41.90 % Syntactic accuracy: 23.46 %
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 29.05 % Semantic accuracy: 38.16 % Syntactic accuracy: 22.56 %
FINAL_vectors_datasettext8_epochs10_size600_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 27.97 % Semantic accuracy: 35.44 % Syntactic accuracy: 22.65 %
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 24.24 % Semantic accuracy: 27.75 % Syntactic accuracy: 21.75 %
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 30.84 % Semantic accuracy: 41.67 % Syntactic accuracy: 23.12 %
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 25.58 % Semantic accuracy: 33.86 % Syntactic accuracy: 19.68 %
FINAL_vectors_datasettext8_epochs10_size800_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 24.82 % Semantic accuracy: 30.08 % Syntactic accuracy: 21.06 %
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 7.90 % Semantic accuracy: 7.61 % Syntactic accuracy: 8.11 %
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 7.80 % Semantic accuracy: 9.24 % Syntactic accuracy: 6.78 %
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 6.82 % Semantic accuracy: 6.41 % Syntactic accuracy: 7.11 %
FINAL_vectors_datasettext8_epochs1_size1000_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 7.76 % Semantic accuracy: 7.34 % Syntactic accuracy: 8.06 %
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 17.85 % Semantic accuracy: 19.57 % Syntactic accuracy: 16.64 %
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 1.54 % Semantic accuracy: 2.13 % Syntactic accuracy: 1.11 %
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 10.88 % Semantic accuracy: 11.18 % Syntactic accuracy: 10.66 %
FINAL_vectors_datasettext8_epochs1_size100_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 14.64 % Semantic accuracy: 15.99 % Syntactic accuracy: 13.68 %
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 17.81 % Semantic accuracy: 21.29 % Syntactic accuracy: 15.33 %
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 7.10 % Semantic accuracy: 8.95 % Syntactic accuracy: 5.77 %
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 12.95 % Semantic accuracy: 14.41 % Syntactic accuracy: 11.90 %
FINAL_vectors_datasettext8_epochs1_size200_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 15.92 % Semantic accuracy: 18.68 % Syntactic accuracy: 13.96 %
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 14.35 % Semantic accuracy: 17.14 % Syntactic accuracy: 12.37 %
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 9.26 % Semantic accuracy: 12.30 % Syntactic accuracy: 7.10 %
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 12.34 % Semantic accuracy: 14.16 % Syntactic accuracy: 11.05 %
FINAL_vectors_datasettext8_epochs1_size400_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 13.73 % Semantic accuracy: 15.94 % Syntactic accuracy: 12.15 %
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 11.95 % Semantic accuracy: 13.47 % Syntactic accuracy: 10.86 %
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 10.02 % Semantic accuracy: 12.06 % Syntactic accuracy: 8.57 %
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 10.53 % Semantic accuracy: 11.38 % Syntactic accuracy: 9.93 %
FINAL_vectors_datasettext8_epochs1_size600_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 11.48 % Semantic accuracy: 13.21 % Syntactic accuracy: 10.25 %
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 10.52 % Semantic accuracy: 10.64 % Syntactic accuracy: 10.43 %
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 9.23 % Semantic accuracy: 11.10 % Syntactic accuracy: 7.90 %
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 9.01 % Semantic accuracy: 9.29 % Syntactic accuracy: 8.82 %
FINAL_vectors_datasettext8_epochs1_size800_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 9.89 % Semantic accuracy: 10.32 % Syntactic accuracy: 9.59 %
FINAL_vectors_datasettext8_epochs25_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 18.10 % Semantic accuracy: 21.57 % Syntactic accuracy: 15.63 %
FINAL_vectors_datasettext8_epochs25_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 32.98 % Semantic accuracy: 46.01 % Syntactic accuracy: 23.71 %
FINAL_vectors_datasettext8_epochs25_size1000_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 24.58 % Semantic accuracy: 33.89 % Syntactic accuracy: 17.95 %
FINAL_vectors_datasettext8_epochs25_size1000_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 20.21 % Semantic accuracy: 25.32 % Syntactic accuracy: 16.56 %
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 27.48 % Semantic accuracy: 29.83 % Syntactic accuracy: 25.80 %
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 6.56 % Semantic accuracy: 7.92 % Syntactic accuracy: 5.59 %
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 15.48 % Semantic accuracy: 18.19 % Syntactic accuracy: 13.55 %
FINAL_vectors_datasettext8_epochs25_size100_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 19.42 % Semantic accuracy: 21.80 % Syntactic accuracy: 17.72 %
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 26.54 % Semantic accuracy: 29.19 % Syntactic accuracy: 24.65 %
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 17.80 % Semantic accuracy: 21.68 % Syntactic accuracy: 15.04 %
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 25.37 % Semantic accuracy: 32.78 % Syntactic accuracy: 20.08 %
FINAL_vectors_datasettext8_epochs25_size200_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 25.41 % Semantic accuracy: 30.07 % Syntactic accuracy: 22.09 %
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 23.59 % Semantic accuracy: 26.89 % Syntactic accuracy: 21.25 %
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 31.51 % Semantic accuracy: 41.40 % Syntactic accuracy: 24.47 %
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 29.16 % Semantic accuracy: 38.96 % Syntactic accuracy: 22.18 %
FINAL_vectors_datasettext8_epochs25_size400_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 26.56 % Semantic accuracy: 33.01 % Syntactic accuracy: 21.96 %
FINAL_vectors_datasettext8_epochs25_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 21.71 % Semantic accuracy: 25.86 % Syntactic accuracy: 18.76 %
FINAL_vectors_datasettext8_epochs25_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 32.51 % Semantic accuracy: 44.39 % Syntactic accuracy: 24.05 %
FINAL_vectors_datasettext8_epochs25_size600_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 28.15 % Semantic accuracy: 37.47 % Syntactic accuracy: 21.51 %
FINAL_vectors_datasettext8_epochs25_size600_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 25.51 % Semantic accuracy: 32.65 % Syntactic accuracy: 20.42 %
FINAL_vectors_datasettext8_epochs25_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 18.78 % Semantic accuracy: 21.68 % Syntactic accuracy: 16.71 %
FINAL_vectors_datasettext8_epochs25_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 31.81 % Semantic accuracy: 44.90 % Syntactic accuracy: 22.49 %
FINAL_vectors_datasettext8_epochs25_size800_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 25.32 % Semantic accuracy: 35.83 % Syntactic accuracy: 17.83 %
FINAL_vectors_datasettext8_epochs25_size800_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 22.65 % Semantic accuracy: 29.52 % Syntactic accuracy: 17.75 %
FINAL_vectors_datasettext8_epochs50_size1000_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 13.43 % Semantic accuracy: 13.93 % Syntactic accuracy: 13.07 %
FINAL_vectors_datasettext8_epochs50_size1000_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 32.59 % Semantic accuracy: 44.54 % Syntactic accuracy: 24.07 %
FINAL_vectors_datasettext8_epochs50_size1000_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 21.27 % Semantic accuracy: 29.18 % Syntactic accuracy: 15.63 %
FINAL_vectors_datasettext8_epochs50_size1000_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 17.36 % Semantic accuracy: 21.51 % Syntactic accuracy: 14.41 %
FINAL_vectors_datasettext8_epochs50_size100_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 24.82 % Semantic accuracy: 25.00 % Syntactic accuracy: 24.70 %
FINAL_vectors_datasettext8_epochs50_size100_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 7.20 % Semantic accuracy: 8.41 % Syntactic accuracy: 6.33 %
FINAL_vectors_datasettext8_epochs50_size100_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 16.41 % Semantic accuracy: 18.80 % Syntactic accuracy: 14.72 %
FINAL_vectors_datasettext8_epochs50_size100_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 18.22 % Semantic accuracy: 20.06 % Syntactic accuracy: 16.91 %
FINAL_vectors_datasettext8_epochs50_size200_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 23.67 % Semantic accuracy: 24.02 % Syntactic accuracy: 23.43 %
FINAL_vectors_datasettext8_epochs50_size200_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 19.08 % Semantic accuracy: 25.26 % Syntactic accuracy: 14.68 %
FINAL_vectors_datasettext8_epochs50_size200_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 25.10 % Semantic accuracy: 32.96 % Syntactic accuracy: 19.50 %
FINAL_vectors_datasettext8_epochs50_size200_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 24.98 % Semantic accuracy: 29.19 % Syntactic accuracy: 21.98 %
FINAL_vectors_datasettext8_epochs50_size400_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 19.57 % Semantic accuracy: 19.69 % Syntactic accuracy: 19.49 %
FINAL_vectors_datasettext8_epochs50_size400_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 31.73 % Semantic accuracy: 44.67 % Syntactic accuracy: 22.51 %
FINAL_vectors_datasettext8_epochs50_size400_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 26.98 % Semantic accuracy: 34.75 % Syntactic accuracy: 21.45 %
FINAL_vectors_datasettext8_epochs50_size400_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 26.27 % Semantic accuracy: 33.29 % Syntactic accuracy: 21.27 %
FINAL_vectors_datasettext8_epochs50_size600_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 17.06 % Semantic accuracy: 18.89 % Syntactic accuracy: 15.76 %
FINAL_vectors_datasettext8_epochs50_size600_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 35.03 % Semantic accuracy: 48.07 % Syntactic accuracy: 25.73 %
FINAL_vectors_datasettext8_epochs50_size600_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 26.60 % Semantic accuracy: 35.26 % Syntactic accuracy: 20.43 %
FINAL_vectors_datasettext8_epochs50_size600_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 22.48 % Semantic accuracy: 27.62 % Syntactic accuracy: 18.82 %
FINAL_vectors_datasettext8_epochs50_size800_neg24_window10_sample1e-4_Q0_mincount5.bin_evaluated_output Total accuracy: 14.76 % Semantic accuracy: 15.49 % Syntactic accuracy: 14.23 %
FINAL_vectors_datasettext8_epochs50_size800_neg24_window10_sample1e-4_Q1_mincount5.bin_evaluated_output Total accuracy: 34.33 % Semantic accuracy: 48.49 % Syntactic accuracy: 24.24 %
FINAL_vectors_datasettext8_epochs50_size800_neg24_window10_sample1e-4_Q2_mincount5.bin_evaluated_output Total accuracy: 23.35 % Semantic accuracy: 32.08 % Syntactic accuracy: 17.14 %
FINAL_vectors_datasettext8_epochs50_size800_neg24_window10_sample1e-4_Q4_mincount5.bin_evaluated_output Total accuracy: 20.16 % Semantic accuracy: 24.82 % Syntactic accuracy: 16.84 %
"""

def extract_name_fields(name):
    matches = re.match("FINAL_vectors_datasettext8_epochs([0-9]+)_size([0-9]+)_neg([0-9]+)_window10_sample1e-4_Q([0-9]+)_mincount5.bin_evaluated_output",
                       name)
    epochs, size, q = matches.group(1), matches.group(2), matches.group(4)
    return (int(epochs), int(size), int(q))

def extract_data(raw_data, value_field=2):
    d = []
    for line in raw_data.split("\n"):
        if line == "":
            continue
        vals = line.split()
        name, accuracy = extract_name_fields(vals[0]), float(vals[value_field])
        d.append((name, accuracy))
    return d

def plot_accuracy_vs_dimension(points_loss, points_acc, n_epochs_ran, keepqs=None):

    # Group together points of the same quantization
    unique_qs = set([d[0][2] for d in points_loss])
    unique_qs = [x for x in unique_qs if x in keepqs]
    grouped_by_qs_acc = []
    for q in unique_qs:
        grouped_by_qs_acc.append([d for d in points_acc if d[0][2] == q])
    grouped_by_qs_loss = []
    for q in unique_qs:
        grouped_by_qs_loss.append([d for d in points_loss if d[0][2] == q])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for data_acc, data_loss in zip(grouped_by_qs_acc, grouped_by_qs_loss):
        q = data_acc[0][0][2]
        q = 32 if q == 0 else q
        points_acc = [(d[0][1], d[1]) for d in data_acc]
        points_acc = sorted(points_acc, key=lambda x: x[0])
        xs = [d[0] for d in points_acc]
        ys = [d[1] for d in points_acc]
        points_loss = [(d[0][1], d[1]) for d in data_loss]
        points_loss = sorted(points_loss, key=lambda x: x[0])
        ys_loss = [d[1] for d in points_loss]
        ax1.plot(xs, ys, label="bits=" + str(q) + " (acc)", marker="o")
        ax2.plot(xs, ys_loss, label="bits=" + str(q) + " (loss)", linestyle=":", marker="o")

    # Merge legend names
    handles,labels = [],[]
    for ax in fig.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    plt.legend(handles,labels)

    ax1.grid()
    ax1.set_title("Accuracy/Loss vs Dimension, 100MB of Wikipedia, %d epochs trained" % n_epochs_ran)
    ax1.set_xlabel("Embedding Dimension")
    ax2.set_ylabel("On the Fly Training Loss")
    ax1.set_ylabel("Google Analogy Average Accuracy % (Sem + Syn)")
    fig.tight_layout()
    fig.savefig("Wiki8AccuracyVsDimensionEpochsTrained=%d.png" % n_epochs_ran)

def plot_accuracy_vs_epochs(points_loss, points_acc, dimension, keepqs=None):

    # Group together points of the same quantization
    unique_qs = set([d[0][2] for d in points_loss])
    unique_qs = [x for x in unique_qs if x in keepqs]
    grouped_by_qs_acc = []
    for q in unique_qs:
        grouped_by_qs_acc.append([d for d in points_acc if d[0][2] == q])
    grouped_by_qs_loss = []
    for q in unique_qs:
        grouped_by_qs_loss.append([d for d in points_loss if d[0][2] == q])

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    for data_acc, data_loss in zip(grouped_by_qs_acc, grouped_by_qs_loss):
        q = data_acc[0][0][2]
        q = 32 if q == 0 else q
        points_acc = [(d[0][0], d[1]) for d in data_acc]
        points_acc = sorted(points_acc, key=lambda x: x[0])
        xs = [d[0] for d in points_acc]
        ys = [d[1] for d in points_acc]
        points_loss = [(d[0][0], d[1]) for d in data_loss]
        points_loss = sorted(points_loss, key=lambda x: x[0])
        ys_loss = [d[1] for d in points_loss]
        ax1.plot(xs, ys, label="Bits=" + str(q) + " (acc)", marker="o")
        ax2.plot(xs, ys_loss, label="Bits=" + str(q) + " (loss)", linestyle=":", marker="o")

    # Merge legend names
    handles,labels = [],[]
    for ax in fig.axes:
        for h,l in zip(*ax.get_legend_handles_labels()):
            handles.append(h)
            labels.append(l)
    plt.legend(handles,labels)

    ax1.grid()
    ax1.set_title("Accuracy/Loss vs Epochs, 100MB of Wikipedia, Dimension %d" % dimension)
    ax1.set_xlabel("Epochs Trained")
    ax1.set_ylabel("Google Analogy Average Accuracy % (Sem + Syn)")
    ax2.set_ylabel("On the Fly Training Loss")
    fig.tight_layout()
    fig.savefig("Wiki8AccuracyVsEpochsTrainedDimension=%d.png" % dimension)


data_losses = extract_data(raw_data_loss)
# Losses are in negative form, so negate
data_losses = [(x[0], -x[1]) for x in data_losses]
data_accs = extract_data(raw_data_acc, value_field=3)

assert(set([x[0] for x in data_losses]) == set([x[0] for x in data_accs]))

# Plot (Accuracy vs Dimension including lines for each Q) for each epochs
unique_epochs = set([d[0][0] for d in data_losses])
for epoch in unique_epochs:
    datapoints_loss = [d for d in data_losses if d[0][0] == epoch]
    datapoints_accs = [d for d in data_accs if d[0][0] == epoch]
    plot_accuracy_vs_dimension(datapoints_loss, datapoints_accs, epoch, keepqs=[0,1])

# Plot (Accuracy vs Epochs includine lines for each Q) for each dimension
unique_dimensions = set([d[0][1] for d in data_losses])
for dimension in unique_dimensions:
    datapoints_loss = [d for d in data_losses if d[0][1] == dimension]
    datapoints_accs = [d for d in data_accs if d[0][1] == dimension]
    plot_accuracy_vs_epochs(datapoints_loss, datapoints_accs, dimension, keepqs=[0,1])
