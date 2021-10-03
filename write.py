import csv
import torch

results = torch.load('results.pth')

for noise_type in ['ACVacuum_7', 'Babble_7', 'CafeRestaurant_7', 'Car_7', 'MetroSubway_7']:
    with open(f'table/{noise_type}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['method'] + [m for m in results[noise_type]['PTN'] if m != 'loss'])

        for method in ['PTN', 'EXTR', 'RETV', 'ALL_A09', 'DAT_full', 'DAT_one', 'NASTAR_A09_K250', 'GT', 'TEST']:
            writer.writerow([method] + ['{:.5f}'.format(results[noise_type][method][m])
                                        for m in results[noise_type][method] if m != 'loss'])
