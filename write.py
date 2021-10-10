import csv
import torch

for noise_type in ['ACVacuum', 'Babble', 'CafeRestaurant', 'Car', 'MetroSubway']:
    results = torch.load(f'vcb_table/results_{noise_type}.pth')
    with open(f'vcb_table/{noise_type}.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(
            ['method'] + [m for m in results[noise_type]['PTN'] if m != 'loss'])

        for method in ['PTN', 'DAT_one', 'DAT_full', 'OPT', 'EXTR', 'GT', 'ALL', 'RETV', 'NASTAR']:
            writer.writerow([method] + ['{:.4f}'.format(results[noise_type][method][m])
                                        for m in results[noise_type][method] if m != 'loss'])
