import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import json
from collections import defaultdict
import pandas as pd
import os


def generate_all_aucs(cut_off=None):
    results_dir = [
        '../1st_revision_output/dblp_baby',
        '../1st_revision_output/ALL',
    ]

    for dir in results_dir:
        frames = []
        for exp in os.walk(dir):
            table_results = defaultdict(lambda: defaultdict(lambda: 0.0))

            method_labels = {
                'AA': ('aa', 'grey'),
                'RA': ('ra', 'cyan'),
                'JC': ('jc', 'orange'),
                'PA': ('pa', 'green'),
                'Content Based': ('content-based', 'blue'),
                'CF': ('icf', 'purple'),
                'Weighted RA': ('weighted_ra', 'yellow'),
                'Semantic diffusion': ('semantic_diffusion', 'pink'),
                'HetGNN': ('HetGNN', 'brown'),
                'Node2Vec': ('node2vec', 'black'),
                'Diffusion': ('diffusion', 'red'),
            }

            # plt.title('Partial ROC Curve', fontsize=20)
            plt.figure()
            if exp[0] != dir:
                year = exp[0].split('/')[-1].split('_')[1]
                print(exp[0])
                for method_name, color in method_labels.items():
                    try:
                        lines = []
                        with open(f'{exp[0]}/{color[0]}.json', 'r', encoding='UTF-8') as f:
                            for line in f:
                                lines.append(json.loads(line))
                        labels = [line[3] for line in lines]
                        print(len(labels))
                        print(len([x for x in labels if x]))
                        scores = [float(line[2]) for line in lines][:cut_off]
                        labels = labels if not cut_off else labels[:cut_off]
                        scores = scores if not scores else scores[:cut_off]

                        pos_length = len([a for a in labels if a])
                        k1, k2, threshold = roc_curve(labels, scores)
                        auc_value = auc(k1, k2)
                        plt.plot(k1, k2, color[1], linestyle='-.', label=f"{method_name}: {round(auc_value, 3)}")
                        top_precision = len([a for a in labels[:pos_length] if a]) / pos_length
                        top_1000_hit = len([a for a in labels[:1000] if a]) / 1000
                        top_100_hit = len([a for a in labels[:100] if a]) / 100
                        print(len([a for a in labels[:100] if a]), top_100_hit)

                        table_results[method_name]['AUC'] = auc_value
                        table_results[method_name]['Precision'] = top_precision
                        table_results[method_name]['top_1000_hit'] = top_1000_hit
                        table_results[method_name]['top_100_hit'] = top_100_hit
                        table_results[method_name]['year'] = float(year)
                    except Exception as e:
                        print(e)

                plt.gca().set_aspect('equal', adjustable='box')
                plt.gcf().set_size_inches(6, 6)
                plt.legend(loc='lower right', prop={'size': 10})
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.ylabel('Proportion of existing links', fontsize=10)
                plt.xlabel('Proportion of non-existing links', fontsize=10)
                pd.DataFrame.from_dict(table_results).T.to_excel(f'{exp[0]}/evaluation.xlsx')
                frames.append(pd.DataFrame.from_dict(table_results).T)
                # plt.show()
                plt.savefig(f'{exp[0]}/auc.png', dpi=1000)
        print(frames)
        all_frame = pd.concat(frames)
        all_frame['method'] = all_frame.index
        print(all_frame)
        all_frame.groupby(["year", "method"]).mean().to_excel(f'{dir}/overall_result.xlsx')


if __name__ == '__main__':
    generate_all_aucs(None)
