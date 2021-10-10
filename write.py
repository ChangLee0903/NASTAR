import csv
import torch

for noise_type in ['ACVacuum_7', 'Babble_7', 'CafeRestaurant_7', 'Car_7', 'MetroSubway_7']:
    results = torch.load(f'vcb_table/results_{noise_type}.pth')
    with open(f'vcb_table/{noise_type}_new.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['method'] + [m for m in results[noise_type]['PTN'] if m != 'loss'])

        for method in ['PTN', 'DAT_one_vcb', 'DAT_full_vcb', 'TEST', 'EXTR', 'GT', 'ALL_A09', 'RETV', 'NASTAR_A09_K250']:
            writer.writerow([method] + ['{:.4f}'.format(results[noise_type][method][m])
                                        for m in results[noise_type][method] if m != 'loss'])
